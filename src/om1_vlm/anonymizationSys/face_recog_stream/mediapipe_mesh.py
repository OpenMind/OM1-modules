from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np


class MediaPipeFaceMesh:
    """Wrapper around MediaPipe Face Mesh for detecting face landmarks."""

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
    ) -> None:
        self._mp = mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __call__(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the first face in a BGR frame and return its landmarks.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Input image in BGR format (as read by OpenCV).

        Returns
        -------
        Optional[np.ndarray]
            Detected face represented as an array of shape (N, 2) containing the
            (x, y) pixel coordinates of the landmarks, or None if no face is detected.
        """
        faces = self.all(frame_bgr)
        return faces[0] if faces else None

    def all(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        """
        Detect all faces in a BGR frame and return their landmarks.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Input image in BGR format (as read by OpenCV).

        Returns
        -------
        List[np.ndarray]
            List of detected faces, each represented as an array of shape (N, 2)
            containing the (x, y) pixel coordinates of the landmarks.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return []

        out = []
        for face in res.multi_face_landmarks:
            pts = np.array(
                [[p.x * w, p.y * h] for p in face.landmark], dtype=np.float64
            )
            out.append(pts)
        return out

    def close(self) -> None:
        """Release resources."""
        self._mesh.close()
