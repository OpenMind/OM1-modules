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
        self.open_capture(self.device, self.width, self.height, self.fps)
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera on device {self.device}")

    def read_frame(self) -> Optional[cv2.Mat]:
        """
        Read a frame from the camera.
        Returns:
            The captured frame as a cv2.Mat object, or None if reading failed.
        """
        if self.cap is None or not self.cap.isOpened():
            logging.warning("Camera is not opened. Reopening...")
            self.open_capture(self.device, self.width, self.height, self.fps)

        ret, frame = self.cap.read()
        if not ret:
            logging.warning("Failed to read frame from camera.")
            return None

        return frame

    def release(self):
        """
        Release the camera resource.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def open_capture(
        self,
        device: str,
        width: int,
        height: int,
        fps: int,
    ):
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
            self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

            if self.cap.isOpened():
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                logging.info(
                    f"Opened camera {device} with MJPEG: {actual_width}x{actual_height} @ {actual_fps}fps"
                )

                self.width = actual_width
                self.height = actual_height
                self.fps = actual_fps

        except Exception as e:
            logging.error(f"Error opening camera {device}: {e}")
            self.cap = None

    def is_opened(self) -> bool:
        """
        Check if the camera is opened.
        Returns:
            True if the camera is opened, False otherwise.
        """
        return self.cap is not None and self.cap.isOpened()
