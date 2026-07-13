from dataclasses import dataclass

import numpy as np

UPPER_INNER_LIP = 13
LOWER_INNER_LIP = 14
LEFT_MOUTH_CORNER = 78
RIGHT_MOUTH_CORNER = 308

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

NOSE_TIP = 1
CHIN = 152


@dataclass(frozen=True)
class LandmarkScheme:
    """Which landmark indices to read. Defaults to MediaPipe FaceMesh."""

    upper_inner_lip: int = UPPER_INNER_LIP
    lower_inner_lip: int = LOWER_INNER_LIP
    left_mouth_corner: int = LEFT_MOUTH_CORNER
    right_mouth_corner: int = RIGHT_MOUTH_CORNER
    left_eye: int = LEFT_EYE_OUTER
    right_eye: int = RIGHT_EYE_OUTER


@dataclass(frozen=True)
class MouthFeature:
    """A single frame's scale-normalized mouth measurements."""

    aperture: float  # inner-lip vertical gap / inter-ocular distance
    width_ratio: float  # mouth width / inter-ocular distance
    scale_px: float  # inter-ocular distance in pixels (proxy for face size)
    valid: bool  # False if the face was too small / degenerate to trust


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def extract_mouth_feature(
    landmarks: np.ndarray,
    scheme: LandmarkScheme = LandmarkScheme(),
    min_scale_px: float = 12.0,
) -> MouthFeature:
    """Compute a scale-normalized mouth feature from one frame of landmarks.

    Parameters
    ----------
    landmarks:
        Array of shape ``(N, 2)`` or ``(N, 3)`` in pixel coordinates. Only the
        first two columns (x, y) are used.
    scheme:
        Landmark index scheme. Defaults to MediaPipe FaceMesh.
    min_scale_px:
        If the inter-ocular distance is below this many pixels the face is too
        small/far to yield a reliable aperture, and we flag the frame invalid.
    """
    pts = np.asarray(landmarks, dtype=np.float64)[:, :2]

    eye_l = pts[scheme.left_eye]
    eye_r = pts[scheme.right_eye]
    scale = _dist(eye_l, eye_r)

    if not np.isfinite(scale) or scale < min_scale_px:
        return MouthFeature(0.0, 0.0, scale, valid=False)

    upper = pts[scheme.upper_inner_lip]
    lower = pts[scheme.lower_inner_lip]
    corner_l = pts[scheme.left_mouth_corner]
    corner_r = pts[scheme.right_mouth_corner]

    aperture = _dist(upper, lower) / scale
    width = _dist(corner_l, corner_r) / scale

    return MouthFeature(
        aperture=aperture,
        width_ratio=width,
        scale_px=scale,
        valid=True,
    )
