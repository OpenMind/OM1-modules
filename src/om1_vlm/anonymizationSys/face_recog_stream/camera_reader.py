from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Optional

import cv2

logging.basicConfig(level=logging.INFO)


class CameraReader:
    """
    Camera reader using OpenCV with a V4L2 backend.

    Capture runs in a BACKGROUND THREAD that continuously pulls the newest frame
    and keeps only the latest one. ``read_frame()`` then returns that latest
    frame without blocking. This mirrors what ``v4l2-ctl --stream-mmap`` does
    (drain continuously) and avoids the classic OpenCV problem where a
    synchronous ``cap.read()`` in the main loop serializes capture with
    processing and throttles FPS far below what the sensor can deliver.
    """

    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: int,
        rotate_90_cw: bool = False,
        threaded: bool = True,
    ):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.rotate_90_cw = rotate_90_cw
        self.threaded = threaded

        self.cap: Optional[cv2.VideoCapture] = None

        # Latest-frame handoff between the grab thread and read_frame().
        self._latest = None
        self._latest_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop = threading.Event()
        self._grab_thread: Optional[threading.Thread] = None

        self.open_camera()

        if self.threaded:
            self._grab_thread = threading.Thread(
                target=self._grab_loop, name="camera-grab", daemon=True
            )
            self._grab_thread.start()
            # Give the thread a moment to land the first frame so the first
            # read_frame() isn't None.
            self._frame_ready.wait(timeout=2.0)

    def open_camera(self):
        """Open the camera using the specified device and settings."""
        self.open_capture(self.device, self.width, self.height, int(self.fps))
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera on device {self.device}")

    def open_capture(
        self, device: str, width: int = 1280, height: int = 720, fps: int = 60
    ) -> None:
        """
        Open a UVC camera using a V4L2 pipeline.

        Parameters
        ----------
        device : str
            Video device path (e.g., '/dev/video0').
        width, height, fps : int
            Desired capture format.
        """
        if self.cap is not None:
            return
        try:
            logging.info("Opening camera with OpenCV V4L2 on %s", device)
            self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # 3 = aperture-priority AUTO exposure -> keep it, it adapts to sun.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

            self._apply_exposure_controls(device)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera device {device}")

            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or float(fps)
            logging.info(
                "Camera opened: %s (%dx%d @ %.2f fps)",
                device,
                actual_width,
                actual_height,
                actual_fps,
            )
            self.width, self.height, self.fps = actual_width, actual_height, actual_fps

        except Exception as e:
            logging.error("Error opening camera %s: %s", device, e)
            self.cap = None

    def _apply_exposure_controls(self, device: str) -> None:
        """Enable dynamic-framerate exposure."""

        def _v4l2(ctrl):
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", device, f"--set-ctrl={ctrl}"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logging.info("v4l2 set %s on %s", ctrl, device)
            except FileNotFoundError:
                logging.warning("v4l2-ctl not found; install v4l-utils")
            except subprocess.CalledProcessError as e:
                logging.warning(
                    "Failed to set %s on %s: %s",
                    ctrl,
                    device,
                    (e.stderr or e.stdout or "").strip(),
                )

        _v4l2("exposure_dynamic_framerate=1")  # over-exposure fix (Jerin)

    def _grab_loop(self):
        """Continuously read the newest frame; keep only the latest."""
        fail = 0
        while not self._stop.is_set():
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.05)
                try:
                    self.cap = None
                    self.open_capture(
                        self.device, self.width, self.height, int(self.fps or 30)
                    )
                except Exception:
                    pass
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                fail += 1
                if fail % 50 == 0:
                    logging.warning("camera grab failing (%d)", fail)
                time.sleep(0.005)
                continue
            fail = 0

            if self.rotate_90_cw:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            with self._latest_lock:
                self._latest = frame
            self._frame_ready.set()

    def is_opened(self) -> bool:
        """
        Check if the camera is opened.

        Returns
        -------
            True if the camera is opened, False otherwise.
        """
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self):
        """
        Return the most recent frame.

        Returns
        -------
        Threaded mode: non-blocking, returns the latest grabbed frame (or None
        until the first one arrives). Non-threaded mode: the original synchronous
        read, kept for compatibility.
        """
        if self.threaded:
            with self._latest_lock:
                return self._latest

        if not self.is_opened():
            logging.warning("Camera is not opened. Reopening...")
            self.open_capture(self.device, self.width, self.height, int(self.fps or 30))
        ret, frame = self.cap.read() if self.cap else (False, None)
        if not ret:
            logging.warning("Failed to read frame from camera.")
            return None
        if self.rotate_90_cw and frame is not None:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def release(self) -> None:
        """
        Release the camera resource.
        """
        self._stop.set()
        if self._grab_thread and self._grab_thread.is_alive():
            self._grab_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
