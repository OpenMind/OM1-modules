from typing import List, Optional

import cv2
import numpy as np
import mediapipe as mp

class MediaPipeFaceMesh:

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
        faces = self.all(frame_bgr)
        return faces[0] if faces else None

    def all(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
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
        self._mesh.close()
