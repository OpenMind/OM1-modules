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
        estimated_fps: int = 15,
        local_rtsp_url: Optional[str] = "rtsp://localhost:8554/live",
        remote_rtsp_url: Optional[str] = None,
    ):
        """
        Initialize the RTSP video stream writer.

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

        self.width = width
        self.height = height
        self.estimated_fps = estimated_fps
        self.local_rtsp_url = local_rtsp_url
        self.remote_rtsp_url = remote_rtsp_url

        # FPS measurement
        self.frame_times = deque(maxlen=30)
        self.last_fps_update = time.time()
        self.current_fps = estimated_fps
        self.fps_update_interval = 5.0

        # subprocess for ffmpeg
        self.process = None
        self.restart_needed = False
        self._start_process()

        # Queue and threading for frame handling
        self.frame_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._writer_thread, daemon=True)
        self.thread.start()

    def _calculate_fps(self):
        """
        Calculate actual FPS from recent frame timestamps.
        """
        if len(self.frame_times) < 2:
            return self.estimated_fps

        time_diffs = []
        for i in range(1, len(self.frame_times)):
            time_diffs.append(self.frame_times[i] - self.frame_times[i - 1])

        if time_diffs:
            avg_frame_time = sum(time_diffs) / len(time_diffs)
            if avg_frame_time > 0:
                calculated_fps = 1.0 / avg_frame_time
                return max(5, min(60, calculated_fps))

        return self.current_fps

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
        fps_change = abs(new_fps - self.current_fps) / self.current_fps
        return fps_change > 0.3

    def _build_ffmpeg_command(self):
        """
        Build the FFmpeg command array with current FPS.
        """
        cmd = [
            "ffmpeg",
            "-y",
            # Video input (raw frames)
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.current_fps),
            "-i",
            "-",
            # Map video stream
            "-map",
            "0:v",
            # Video encoding
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-b:v",
            f"{int(400 * self.current_fps / 15)}k",
            "-g",
            str(max(15, int(self.current_fps * 2))),
            "-keyint_min",
            str(max(5, int(self.current_fps / 2))),
            "-vsync",
            "cfr",
            "-r",
            str(self.current_fps),
            "-vf",
            "transpose=2",
            "-max_muxing_queue_size",
            "1024",
        ]

        if self.remote_rtsp_url:
            cmd.extend(
                [
                    "-f",
                    "tee",
                    f"[f=rtsp:rtsp_transport=tcp]{self.local_rtsp_url}|"
                    f"[f=rtsp:rtsp_transport=tcp:onfail=ignore]{self.remote_rtsp_url}",
                ]
            )
        else:
            cmd.extend(["-f", "rtsp", "-rtsp_transport", "tcp", self.local_rtsp_url])

        return cmd

    def _start_process(self):
        """
        Start or restart the FFmpeg subprocess.
        """
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

        try:
            cmd = self._build_ffmpeg_command()
            logging.info(f"Starting FFmpeg with FPS: {self.current_fps}")
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.width * self.height * 3 * 3,
            )
            logging.info(f"Started FFmpeg subprocess with PID: {self.process.pid}")
            self.restart_needed = False
        except Exception as e:
            logging.error(f"Failed to start FFmpeg: {e}")
            self.process = None
            raise

    def _is_process_healthy(self):
        """
        Check if the process is running and available.
        """
        return self.process is not None and self.process.poll() is None

    def write_frame(self, frame):
        """
        Write a video frame to the stream with FPS tracking.
        """
        if self.stop_event.is_set() or frame is None:
            return

        current_time = time.time()
        self.frame_times.append(current_time)

        if current_time - self.last_fps_update > self.fps_update_interval:
            new_fps = self._calculate_fps()

            if self._should_restart_process(new_fps):
                logging.info(
                    f"FPS changed significantly: {self.current_fps} -> {new_fps:.1f}, restarting FFmpeg"
                )
                self.current_fps = new_fps
                self.restart_needed = True

            self.last_fps_update = current_time

        try:
            while self.frame_queue.qsize() >= 3:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            self.frame_queue.put_nowait((frame, current_time))

        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait((frame, current_time))
            except queue.Empty:
                pass
        except Exception as e:
            logging.error(f"Error queueing frame: {e}")

    def _writer_thread(self):
        """
        Thread function to write frames to FFmpeg subprocess.
        """
        while not self.stop_event.is_set():
            try:
                frame_data = self.frame_queue.get(timeout=1)
                frame, _ = frame_data

                if self.restart_needed or not self._is_process_healthy():
                    logging.warning("Restarting FFmpeg process...")
                    self._start_process()

                    if not self._is_process_healthy():
                        logging.error("Failed to restart FFmpeg process")
                        continue

                try:
                    self.process.stdin.write(frame.tobytes())
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError) as e:
                    logging.error(f"Pipe error writing frame: {e}")
                    self.restart_needed = True

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Unexpected error in writer thread: {e}")

        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception:
                pass
            finally:
                self.process = None

    def get_current_fps(self):
        """
        Get the current measured FPS.
        """
        return self._calculate_fps()

    def stop(self):
        """
        Stop the video stream writer and clean up resources.
        """
        self.stop_event.set()
        self.thread.join(timeout=5)

        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception:
                pass
            finally:
                self.process = None
