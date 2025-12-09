from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from collections import deque
from typing import Optional

logging.basicConfig(level=logging.INFO)


class RTSPVideoStreamWriter:
    def __init__(
        self,
        width: int,
        height: int,
        estimated_fps: int = 30,
        local_rtsp_url: Optional[str] = "rtsp://localhost:8554/live",
        remote_rtsp_url: Optional[str] = None,
    ):
        """
        Initialize the RTSP video stream writer (video-only; no audio).

        Parameters
        ----------
        width : int
            Width of the video frames.
        height : int
            Height of the video frames.
        estimated_fps : int, optional
            Estimated frames per second for the video stream, by default 15.
        local_rtsp_url : Optional[str], optional
            Local RTSP URL to stream to, by default "rtsp://localhost:8554/live".
        remote_rtsp_url : Optional[str], optional
            Remote RTSP URL to stream to, by default None.
        """
        if not local_rtsp_url and not remote_rtsp_url:
            raise ValueError(
                "At least one of local_rtsp_url or remote_rtsp_url must be provided."
            )

        self.width = int(width)
        self.height = int(height)
        self.estimated_fps = float(estimated_fps)
        self.local_rtsp_url = local_rtsp_url
        self.remote_rtsp_url = remote_rtsp_url

        # FPS measurement
        self.frame_times = deque(maxlen=60)
        self.last_fps_update = time.time()
        self.current_fps = float(estimated_fps)
        self.fps_update_interval = 5.0  # seconds

        # Throughput logs
        self._log_interval = 3.0
        self._last_log_t = time.perf_counter()
        self._frames_written = 0

        # Subprocess for ffmpeg
        self.process: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_stop = threading.Event()
        self.restart_needed = False
        self._start_process()

        # Queue and threading for frame handling
        self.frame_queue: "queue.Queue" = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._writer_thread, daemon=True)
        self.thread.start()

    def _calculate_fps(self) -> float:
        """
        Calculate actual FPS from recent frame timestamps.
        """
        if len(self.frame_times) < 2:
            return self.current_fps

        time_diffs = [
            self.frame_times[i] - self.frame_times[i - 1]
            for i in range(1, len(self.frame_times))
        ]
        if not time_diffs:
            return self.current_fps

        avg_frame_time = sum(time_diffs) / len(time_diffs)
        if avg_frame_time <= 0:
            return self.current_fps

        calculated_fps = 1.0 / avg_frame_time
        # keep within sane bounds for RTSP/x264
        return max(5.0, min(60.0, calculated_fps))

    def _should_restart_process(self, new_fps: float) -> bool:
        """
        Check if we need to restart FFmpeg due to significant FPS change.

        Parameters
        ----------
        new_fps : float
            Newly calculated FPS.

        Returns
        -------
        bool
            True if the FPS change exceeds 30%, False otherwise.
        """
        if self.current_fps <= 0:
            return True
        fps_change = abs(new_fps - self.current_fps) / self.current_fps
        return fps_change > 0.30  # >30%

    def _build_ffmpeg_command(self):
        """
        Build the FFmpeg command array with current FPS.
        Video-only (no audio) for maximum FPS and stability.
        """
        gop = max(1, int(round(self.current_fps)))  # 1-second GOP for low latency
        vbv = "4M"  # small buffer to keep latency predictable

        # Base input: raw BGR frames over stdin
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.current_fps:.3f}",
            "-use_wallclock_as_timestamps",
            "1",
            "-i",
            "-",
            "-map",
            "0:v",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-crf",
            "30",
            "-maxrate",
            vbv,
            "-bufsize",
            vbv,
            "-g",
            str(gop),
            "-x264-params",
            f"keyint={gop}:min-keyint={gop}:scenecut=0:rc-lookahead=0:ref=1:bframes=0",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "cfr",
            "-r",
            f"{self.current_fps:.3f}",
        ]

        if self.remote_rtsp_url:
            tee_arg = (
                f"[f=rtsp:rtsp_transport=tcp]{self.local_rtsp_url}"
                f"|[f=rtsp:rtsp_transport=tcp:onfail=ignore]{self.remote_rtsp_url}"
            )
            cmd += ["-f", "tee", tee_arg]
        else:
            cmd += ["-f", "rtsp", "-rtsp_transport", "tcp", self.local_rtsp_url]

        return cmd

    def _start_process(self):
        """
        Start or restart the FFmpeg subprocess.
        """
        # Clean up any old process
        if self.process:
            try:
                if self.process.stdin and not self.process.stdin.closed:
                    self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception:
                pass
            finally:
                self.process = None

        # Stop previous stderr logger if any
        self._stderr_stop.set()
        if self._stderr_thread and self._stderr_thread.is_alive():
            try:
                self._stderr_thread.join(timeout=1.0)
            except Exception:
                pass
        self._stderr_stop.clear()
        self._stderr_thread = None

        # Launch fresh
        cmd = self._build_ffmpeg_command()
        logging.info("Starting FFmpeg with FPS: %.2f", self.current_fps)
        logging.info("FFmpeg command: %s", " ".join(cmd))
        try:
            # Use a generous stdin buffer to reduce write stalls
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,  # capture for logging
                bufsize=self.width * self.height * 3 * 2,
            )
            logging.info("Started FFmpeg subprocess with PID: %s", self.process.pid)
            self.restart_needed = False

            # Start stderr logger thread
            self._stderr_thread = threading.Thread(
                target=self._stderr_logger_loop, name="ffmpeg-stderr", daemon=True
            )
            self._stderr_thread.start()

            # Reset throughput counters
            self._last_log_t = time.perf_counter()
            self._frames_written = 0

        except Exception as e:
            logging.error("Failed to start FFmpeg: %s", e)
            self.process = None
            raise

    def _stderr_logger_loop(self):
        """
        Continuously read FFmpeg stderr and log lines (helps debug exits / errors).
        """
        if not self.process or not self.process.stderr:
            return
        try:
            for raw in iter(self.process.stderr.readline, b""):
                if self._stderr_stop.is_set():
                    break
                line = raw.decode(errors="replace").rstrip()
                if line:
                    logging.error("ffmpeg> %s", line)
        except Exception as e:
            logging.debug("ffmpeg stderr logger ended: %s", e)

    def _is_process_healthy(self) -> bool:
        """
        Check if the process is running and available.
        """
        return self.process is not None and self.process.poll() is None

    def write_frame(self, frame):
        """
        Queue a frame for sending; tracks FPS and triggers restarts if needed.
        """
        if self.stop_event.is_set() or frame is None:
            return

        # Quick sanity check to avoid mismatched sizes
        try:
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                logging.warning(
                    "Frame size %dx%d != configured %dx%d; dropping",
                    w,
                    h,
                    self.width,
                    self.height,
                )
                return
        except Exception:
            # If frame isn't a numpy array or shape is missing
            return

        now = time.time()
        self.frame_times.append(now)

        # Periodically refresh FPS estimate & decide on restart
        if now - self.last_fps_update > self.fps_update_interval:
            new_fps = self._calculate_fps()
            if self._should_restart_process(new_fps):
                logging.info(
                    "FPS changed: %.1f -> %.1f, restarting FFmpeg",
                    self.current_fps,
                    new_fps,
                )
                self.current_fps = new_fps
                self.restart_needed = True
            self.last_fps_update = now

        # Keep latency low: drop oldest if we’re falling behind
        # try:
        #     while self.frame_queue.qsize() >= 3:
        #         try:
        #             self.frame_queue.get_nowait()
        #         except queue.Empty:
        #             break
        #     self.frame_queue.put_nowait(frame)
        # except queue.Full:
        #     try:
        #         self.frame_queue.get_nowait()
        #         self.frame_queue.put_nowait(frame)
        #     except queue.Empty:
        #         pass
        # except Exception as e:
        #     logging.error("Error queueing frame: %s", e)
        try:
            self.frame_queue.put(frame, block=True, timeout=0.05)
        except queue.Full:
            # Only drop oldest frame if absolutely necessary
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except (queue.Empty, queue.Full):
                pass

    def _writer_thread(self):
        """
        Thread function to write frames to FFmpeg subprocess.
        """
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Restart if requested or if FFmpeg died
            if self.restart_needed or not self._is_process_healthy():
                logging.warning("Restarting FFmpeg process...")
                self._start_process()
                if not self._is_process_healthy():
                    logging.error("Failed to restart FFmpeg process")
                    continue

            # Write raw BGR24
            try:
                # type: ignore[union-attr]
                self.process.stdin.write(frame.tobytes())
                self.process.stdin.flush()
                self._frames_written += 1
            except (BrokenPipeError, OSError) as e:
                logging.error("Pipe error writing frame: %s", e)
                self.restart_needed = True
            except Exception as e:
                logging.error("Unexpected error writing frame: %s", e)

            # Periodic throughput log
            now = time.perf_counter()
            if now - self._last_log_t >= self._log_interval:
                elapsed = now - self._last_log_t
                fps = self._frames_written / elapsed if elapsed > 0 else 0.0
                qsz = self.frame_queue.qsize()
                logging.info("[RTSP out] ~%.1f fps (queue=%d, enc=libx264)", fps, qsz)
                self._last_log_t = now
                self._frames_written = 0

        # Teardown
        if self.process:
            try:
                if self.process.stdin and not self.process.stdin.closed:
                    self.process.stdin.close()
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            except Exception:
                pass
            finally:
                self.process = None

        # Stop stderr logger
        self._stderr_stop.set()
        if self._stderr_thread and self._stderr_thread.is_alive():
            try:
                self._stderr_thread.join(timeout=1.0)
            except Exception:
                pass

    def get_current_fps(self) -> float:
        """
        Get the current measured FPS.
        """
        return self._calculate_fps()

    def stop(self):
        """
        Stop the video stream writer and clean up resources.
        """
        self.stop_event.set()
        try:
            self.thread.join(timeout=5)
        except Exception:
            pass

        if self.process:
            try:
                if self.process.stdin and not self.process.stdin.closed:
                    self.process.stdin.close()
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            except Exception:
                pass
            finally:
                self.process = None

        # Stop stderr logger
        self._stderr_stop.set()
        if self._stderr_thread and self._stderr_thread.is_alive():
            try:
                self._stderr_thread.join(timeout=1.0)
            except Exception:
                pass
