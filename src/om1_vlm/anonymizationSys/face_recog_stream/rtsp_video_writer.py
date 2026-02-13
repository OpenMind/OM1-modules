from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from collections import deque
from typing import Optional

logging.basicConfig(level=logging.INFO)


def _check_nvenc_available() -> bool:
    """Check if NVENC hardware encoder is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


class RTSPVideoStreamWriter:
    """
    RTSP video stream writer using FFmpeg with optional NVENC hardware encoding.
    """

    def __init__(
        self,
        width: int,
        height: int,
        estimated_fps: int = 30,
        local_rtsp_url: Optional[str] = "rtsp://localhost:8554/live",
        remote_rtsp_url: Optional[str] = None,
        raw_local_rtsp_url: Optional[str] = None,
        raw_remote_rtsp_url: Optional[str] = None,
        use_hwenc: bool = True,  # NEW: prefer hardware encoding
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
            Local RTSP URL to stream processed video to, by default "rtsp://localhost:8554/live".
        remote_rtsp_url : Optional[str], optional
            Remote RTSP URL to stream processed video to, by default None.
        raw_local_rtsp_url : Optional[str], optional
            Local RTSP URL to stream raw video to, by default None.
        raw_remote_rtsp_url : Optional[str], optional
            Remote RTSP URL to stream raw video to, by default None.
        use_hwenc : bool
            If True, try to use NVENC hardware encoder (Jetson/NVIDIA GPU).
            Falls back to libx264 if unavailable.
        """
        if (
            not local_rtsp_url
            and not remote_rtsp_url
            and not raw_local_rtsp_url
            and not raw_remote_rtsp_url
        ):
            raise ValueError("At least one RTSP URL must be provided.")

        self.width = int(width)
        self.height = int(height)
        self.estimated_fps = float(estimated_fps)
        self.local_rtsp_url = local_rtsp_url
        self.remote_rtsp_url = remote_rtsp_url
        self.raw_local_rtsp_url = raw_local_rtsp_url
        self.raw_remote_rtsp_url = raw_remote_rtsp_url

        # Check hardware encoder availability
        self.use_nvenc = use_hwenc and _check_nvenc_available()
        if use_hwenc and self.use_nvenc:
            logging.info("Using NVENC hardware encoder")
        elif use_hwenc:
            logging.warning("NVENC not available, falling back to libx264")

        # FPS measurement
        self.frame_times = deque(maxlen=60)
        self.last_fps_update = time.time()
        self.current_fps = float(estimated_fps)
        self.fps_update_interval = 5.0  # seconds

        # Throughput logs
        self._log_interval = 3.0
        self._last_log_t = time.perf_counter()
        self._frames_written = 0

        # Processed video subprocess
        self.process: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_stop = threading.Event()
        self.restart_needed = False

        # Raw video subprocess
        self.raw_process: Optional[subprocess.Popen] = None
        self._raw_stderr_thread: Optional[threading.Thread] = None
        self._raw_stderr_stop = threading.Event()
        self.raw_restart_needed = False

        # Start processes
        if self.local_rtsp_url or self.remote_rtsp_url:
            self._start_process()
        if self.raw_local_rtsp_url or self.raw_remote_rtsp_url:
            self._start_raw_process()

        # Queue and threading
        self.frame_queue: "queue.Queue" = queue.Queue(maxsize=3)
        self.raw_frame_queue: "queue.Queue" = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._writer_thread, daemon=True)
        self.thread.start()
        self.raw_thread = threading.Thread(target=self._raw_writer_thread, daemon=True)
        self.raw_thread.start()

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

    def _build_ffmpeg_command(self, is_raw: bool = False):
        """Build FFmpeg command with hardware or software encoding.

        Parameters
        ----------
        is_raw : bool
            If True, build command for raw video stream, otherwise for processed video.
        """
        gop = max(1, int(round(self.current_fps)))  # 1-second keyframe interval
        vbv = "2M"

        # Base input - raw BGR frames from stdin
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
            "-thread_queue_size",
            "16",  # Small input queue
            "-i",
            "-",
            "-map",
            "0:v",
        ]

        if self.use_nvenc:
            # NVENC hardware encoding (Jetson / NVIDIA GPU)
            # Optimized for lowest latency streaming
            cmd += [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p1",  # p1=fastest, p7=best quality
                "-tune",
                "ll",  # Low latency tuning
                "-profile:v",
                "baseline",  # Most compatible, lowest latency
                "-rc",
                "cbr",  # Constant bitrate for stable streaming
                "-b:v",
                vbv,
                "-maxrate",
                vbv,
                "-bufsize",
                vbv,  # 1-second buffer
                "-g",
                str(gop),  # Keyframe every 1 second
                "-bf",
                "0",  # No B-frames (lower latency)
                "-strict_gop",
                "1",  # Force fixed GOP
                "-delay",
                "0",  # Zero encoding delay
                "-zerolatency",
                "1",  # Zero latency mode
                "-pix_fmt",
                "yuv420p",
            ]
        else:
            # CPU software encoding (libx264) - fallback
            cmd += [
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-profile:v",
                "baseline",
                "-crf",
                "28",
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
            ]

        # Output settings
        cmd += [
            "-fps_mode",
            "cfr",
            "-r",
            f"{self.current_fps:.3f}",
        ]

        # RTSP output
        if is_raw:
            local_url = self.raw_local_rtsp_url
            remote_url = self.raw_remote_rtsp_url
        else:
            local_url = self.local_rtsp_url
            remote_url = self.remote_rtsp_url

        if remote_url:
            tee_arg = (
                f"[f=rtsp:rtsp_transport=tcp]{local_url}"
                f"|[f=rtsp:rtsp_transport=tcp:onfail=ignore]{remote_url}"
            )
            cmd += ["-f", "tee", tee_arg]
        else:
            cmd += ["-f", "rtsp", "-rtsp_transport", "tcp", local_url]

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
        enc_type = "NVENC" if self.use_nvenc else "libx264"
        logging.info("Starting FFmpeg (%s) with FPS: %.2f", enc_type, self.current_fps)
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
            logging.info(
                "Started FFmpeg subprocess (PID: %s, encoder: %s)",
                self.process.pid,
                enc_type,
            )
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

    def _start_raw_process(self):
        """
        Start or restart the FFmpeg subprocess for raw video stream.
        """
        # Clean up any old process
        if self.raw_process:
            try:
                if self.raw_process.stdin and not self.raw_process.stdin.closed:
                    self.raw_process.stdin.close()
                self.raw_process.terminate()
                self.raw_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.raw_process.kill()
                self.raw_process.wait()
            except Exception:
                pass
            finally:
                self.raw_process = None

        # Stop previous stderr logger if any
        self._raw_stderr_stop.set()
        if self._raw_stderr_thread and self._raw_stderr_thread.is_alive():
            try:
                self._raw_stderr_thread.join(timeout=1.0)
            except Exception:
                pass
        self._raw_stderr_stop.clear()
        self._raw_stderr_thread = None

        # Launch fresh
        cmd = self._build_ffmpeg_command(is_raw=True)
        enc_type = "NVENC" if self.use_nvenc else "libx264"
        logging.info(
            "Starting FFmpeg (RAW, %s) with FPS: %.2f", enc_type, self.current_fps
        )
        logging.info("FFmpeg RAW command: %s", " ".join(cmd))

        try:
            # Use a generous stdin buffer to reduce write stalls
            self.raw_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,  # capture for logging
                bufsize=self.width * self.height * 3 * 2,
            )
            logging.info(
                "Started FFmpeg RAW subprocess (PID: %s, encoder: %s)",
                self.raw_process.pid,
                enc_type,
            )
            self.raw_restart_needed = False

            # Start stderr logger thread
            self._raw_stderr_thread = threading.Thread(
                target=self._raw_stderr_logger_loop,
                name="ffmpeg-raw-stderr",
                daemon=True,
            )
            self._raw_stderr_thread.start()

        except Exception as e:
            logging.error("Failed to start FFmpeg RAW: %s", e)
            self.raw_process = None
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

    def _raw_stderr_logger_loop(self):
        """
        Continuously read FFmpeg RAW stderr and log lines (helps debug exits / errors).
        """
        if not self.raw_process or not self.raw_process.stderr:
            return
        try:
            for raw in iter(self.raw_process.stderr.readline, b""):
                if self._raw_stderr_stop.is_set():
                    break
                line = raw.decode(errors="replace").rstrip()
                if line:
                    logging.error("ffmpeg-raw> %s", line)
        except Exception as e:
            logging.debug("ffmpeg raw stderr logger ended: %s", e)

    def _is_process_healthy(self) -> bool:
        """
        Check if the process is running and available.
        """
        return self.process is not None and self.process.poll() is None

    def write_raw_frame(self, frame):
        """
        Queue a raw frame for sending to raw RTSP stream.
        """
        if self.stop_event.is_set() or frame is None:
            return
        if not self.raw_local_rtsp_url and not self.raw_remote_rtsp_url:
            return

        # Quick sanity check
        try:
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                logging.warning(
                    "Raw frame size %dx%d != configured %dx%d; dropping",
                    w,
                    h,
                    self.width,
                    self.height,
                )
                return
        except Exception:
            return

        try:
            # Non-blocking put with drop-oldest strategy
            if self.raw_frame_queue.full():
                try:
                    self.raw_frame_queue.get_nowait()  # Drop oldest
                except queue.Empty:
                    pass
            self.raw_frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Drop frame if still full
        except Exception as e:
            logging.error("Error queueing raw frame: %s", e)

    def write_frame(self, frame):
        """
        Queue a processed frame for sending; tracks FPS and triggers restarts if needed.
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

        try:
            # Non-blocking put with drop-oldest strategy
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()  # Drop oldest
                except queue.Empty:
                    pass
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Drop frame if still full
        except Exception as e:
            logging.error("Error queueing frame: %s", e)

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
                if self.process and self.process.stdin:
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
                enc = "nvenc" if self.use_nvenc else "x264"
                logging.info("[RTSP out] ~%.1f fps (queue=%d, enc=%s)", fps, qsz, enc)
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

    def _raw_writer_thread(self):
        """
        Thread function to write raw frames to FFmpeg subprocess.
        """
        frames_written = 0
        last_log_t = time.perf_counter()

        while not self.stop_event.is_set():
            try:
                frame = self.raw_frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Restart if requested or if FFmpeg died
            if self.raw_restart_needed or not (
                self.raw_process and self.raw_process.poll() is None
            ):
                logging.warning("Restarting FFmpeg RAW process...")
                self._start_raw_process()
                if not (self.raw_process and self.raw_process.poll() is None):
                    logging.error("Failed to restart FFmpeg RAW process")
                    continue

            # Write raw BGR24
            try:
                if self.raw_process and self.raw_process.stdin:
                    self.raw_process.stdin.write(frame.tobytes())
                    self.raw_process.stdin.flush()
                    frames_written += 1
            except (BrokenPipeError, OSError) as e:
                logging.error("Pipe error writing raw frame: %s", e)
                self.raw_restart_needed = True
            except Exception as e:
                logging.error("Unexpected error writing raw frame: %s", e)

            # Periodic throughput log
            now = time.perf_counter()
            if now - last_log_t >= self._log_interval:
                elapsed = now - last_log_t
                fps = frames_written / elapsed if elapsed > 0 else 0.0
                qsz = self.raw_frame_queue.qsize()
                enc = "nvenc" if self.use_nvenc else "x264"
                logging.info(
                    "[RTSP RAW out] ~%.1f fps (queue=%d, enc=%s)", fps, qsz, enc
                )
                last_log_t = now
                frames_written = 0

        # Teardown
        if self.raw_process:
            try:
                if self.raw_process.stdin and not self.raw_process.stdin.closed:
                    self.raw_process.stdin.close()
                self.raw_process.terminate()
                try:
                    self.raw_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.raw_process.kill()
                    self.raw_process.wait()
            except Exception:
                pass
            finally:
                self.raw_process = None

        # Stop stderr logger
        self._raw_stderr_stop.set()
        if self._raw_stderr_thread and self._raw_stderr_thread.is_alive():
            try:
                self._raw_stderr_thread.join(timeout=1.0)
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

        if self.raw_thread:
            try:
                self.raw_thread.join(timeout=5)
            except Exception:
                pass

        # Cleanup processed video process
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

        # Cleanup raw video process
        if self.raw_process:
            try:
                if self.raw_process.stdin and not self.raw_process.stdin.closed:
                    self.raw_process.stdin.close()
                self.raw_process.terminate()
                try:
                    self.raw_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.raw_process.kill()
                    self.raw_process.wait()
            except Exception:
                pass
            finally:
                self.raw_process = None

        # Stop raw stderr logger
        self._raw_stderr_stop.set()
        if self._raw_stderr_thread and self._raw_stderr_thread.is_alive():
            try:
                self._raw_stderr_thread.join(timeout=1.0)
            except Exception:
                pass
