from __future__ import annotations

import logging
from typing import Optional

import cv2

logging.basicConfig(level=logging.INFO)


class CameraReader:
    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: int,
        rotate_90_cw: bool = False,
    ):
        """
        Initialize the camera reader with the specified device and settings.

        Parameters
        ----------
        device : str
            Video device path (e.g., '/dev/video0').
        width : int
            Desired frame width.
        height : int
            Desired frame height.
        fps : int
            Desired frames per second.
        rotate_90_cw : bool
            Whether to rotate frames 90 degrees clockwise.
        """
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.rotate_90_cw = rotate_90_cw

        self.cap: Optional[cv2.VideoCapture] = None
        self.open_camera()

    def open_camera(self):
        """
        Open the camera using the specified device and settings.
        """
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
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

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

    def is_opened(self) -> bool:
        """
        Check if the camera is opened.
        Returns:
            True if the camera is opened, False otherwise.
        """
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self):
        """
        Read a frame from the camera.
        Returns:
            The captured frame as a cv2.Mat object, or None if reading failed.
        """
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
        if self.cap:
            self.cap.release()
            self.cap = None
