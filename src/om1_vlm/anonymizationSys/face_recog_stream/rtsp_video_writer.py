from __future__ import annotations

import logging
import subprocess
import queue
import threading
from typing import Optional


logging.basicConfig(level=logging.INFO)

class RTSPVideoStreamWriter():
    def __init__(
            self,
            width: int,
            height: int,
            fps: int,
            local_rtsp_url: str = "rtsp://localhost:8554/live",
            remote_rtsp_url: Optional[str] = None,
            mic_device: str = "hw:3,0",
            mic_ac: int = 2,
        ):
        """
        Initialize the RTSP video stream writer.
        """

        if not local_rtsp_url and not remote_rtsp_url:
            raise ValueError("At least one of local_rtsp_url or remote_rtsp_url must be provided.")

        self.width = width
        self.height = height
        self.fps = fps
        self.local_rtsp_url = local_rtsp_url
        self.remote_rtsp_url = remote_rtsp_url
        self.mic_device = mic_device
        self.mic_ac = mic_ac

        # subprocess for ffmpeg
        self.process = None
        self._start_process()

        # Queue and threading for frame handling
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._writer_thread, daemon=True)
        self.thread.start()

    def _build_ffmpeg_command(self):
        """
        Build the FFmpeg command array.
        """
        cmd = [
            "ffmpeg", "-y",
            # Video input
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",

            # Audio input
            "-f", "alsa",
            "-ac", str(self.mic_ac),
            "-i", self.mic_device,

            # Video encoding
            "-map", "0:v",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-g", str(self.fps),
            "-keyint_min", str(self.fps),

            # Rotation
            "-vf", "transpose=2",

            # Audio encoding
            "-map", "1:a",
            "-c:a", "libopus",
            "-ar", "48000",
            "-ac", "2",
            "-b:a", "128k",
        ]

        if self.remote_rtsp_url:
            cmd.extend(["-f", "tee",
                       f"[f=rtsp:rtsp_transport=tcp]{self.local_rtsp_url}|"
                       f"[f=rtsp:rtsp_transport=tcp]{self.remote_rtsp_url}"])
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
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.width * self.height * 3 * 2,
            )
            logging.info(f"Started FFmpeg subprocess with PID: {self.process.pid}")
        except Exception as e:
            logging.error(f"Failed to start FFmpeg: {e}")
            self.process = None
            raise

    def _is_process_healthy(self):
        """
        Check if the process is running is available.
        """
        return self.process is not None and self.process.poll() is None

    def write_frame(self, frame):
        """
        Write a video frame to the stream.
        """
        if self.stop_event.is_set() or frame is None:
            return

        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            self.frame_queue.put_nowait(frame)

        except queue.Full:
            logging.debug("Frame queue full, dropping frame")
        except Exception as e:
            logging.error(f"Error queueing frame: {e}")

    def _writer_thread(self):
        """
        Thread function to write frames to FFmpeg subprocess.
        """
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)

                if not self._is_process_healthy():
                    logging.warning("FFmpeg process unhealthy, restarting...")
                    self._start_process()

                    if not self._is_process_healthy():
                        logging.error("Failed to restart FFmpeg process")
                        continue

                try:
                    self.process.stdin.write(frame.tobytes())
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError) as e:
                    logging.error(f"Pipe error writing frame: {e}")

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
