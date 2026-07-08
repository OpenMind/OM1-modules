"""Head pose (yaw / pitch / roll) and a pitch-aware frontality score.

Why this exists: selfie_logic.frontality() scores only yaw + roll from the 5
keypoints, so a face looking UP or DOWN still passes the enrollment gate even
though it's a poor sample for recognition. This estimates full head pose via
solveOnP on the same 5 SCRFD keypoints you already have (no new model needed),
which yields a real pitch term. An optional chin point (e.g. the lowest
2d106det landmark) sharpens pitch further.

Drop-in: `frontality(kps)` returns [0,1] like the old one, same call sites; it
just now also penalizes pitch. Falls back to the old yaw*roll estimate if
solveOnP can't run.

Coordinate notes:
  * kps are the 5 SCRFD keypoints in IMAGE-PIXEL coords, standard InsightFace
    order: [left_eye, right_eye, nose, left_mouth, right_mouth].
  * Only the MAGNITUDES of the angles matter for the gate, so left/right sign
    conventions are irrelevant here.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

# ---- generic 3D face model (arbitrary mm units), SCRFD kps order -----
_MODEL5 = np.array(
    [
        [-30.0, 30.0, -30.0],  # left eye
        [30.0, 30.0, -30.0],  # right eye
        [0.0, 0.0, 0.0],  # nose tip (reference, closest to camera)
        [-25.0, -30.0, -30.0],  # left mouth corner
        [25.0, -30.0, -30.0],  # right mouth corner
    ],
    dtype=np.float64,
)
_CHIN3D = np.array([[0.0, -65.0, -30.0]], dtype=np.float64)

# ---- frontality mapping (tune to taste) ------------------------------
YAW_LIMIT_DEG = 45.0  # |yaw|  >= this -> yaw term 0
PITCH_LIMIT_DEG = 35.0  # |pitch|>= this -> pitch term 0
ROLL_LIMIT_DEG = 35.0  # |roll| >= this -> roll term 0
# solveOnP with a generic model can report a small constant pitch on a truly
# frontal face. Measure it once (print angles on a head-on face) and set here.
PITCH_OFFSET_DEG = 0.0


def head_pose_angles(
    kps5,
    img_w: Optional[float] = None,
    img_h: Optional[float] = None,
    chin: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, float, float]]:
    """Return (yaw, pitch, roll) in degrees, or None if it can't be solved.

    kps5 : (5,2) image-pixel keypoints [Leye, Reye, nose, Lmouth, Rmouth].
    chin : optional (2,) image-pixel point (e.g. lowest 2d106det landmark,
           mapped to image coords) to better constrain pitch.
    """
    if kps5 is None or len(kps5) < 5:
        return None
    img = np.asarray([np.asarray(kps5[i][:2], dtype=np.float64) for i in range(5)])
    model = _MODEL5
    if chin is not None:
        img = np.vstack([img, np.asarray(chin[:2], dtype=np.float64)])
        model = np.vstack([_MODEL5, _CHIN3D])

    # Camera matrix. Angles are fairly insensitive to the exact focal length,
    # so if image dims are unknown we estimate focal from the inter-ocular span.
    iod = float(np.linalg.norm(img[1] - img[0])) + 1e-6
    f = float(img_w) if img_w else max(iod * 6.0, 1.0)
    cx = float(img_w) / 2.0 if img_w else float(img[:, 0].mean())
    cy = float(img_h) / 2.0 if img_h else float(img[:, 1].mean())
    cam = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    # SQPNP handles 3+ points (ITERATIVE needs 6); robust for the 5-kp case.
    flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_EPNP)
    try:
        ok, rvec, _ = cv2.solveOnP(model, img, cam, np.zeros((4, 1)), flags=flag)
    except cv2.error:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    sy = float(np.hypot(R[0, 0], R[1, 0]))
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:  # gimbal-lock fallback
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = 0.0

    # Normalize pitch/roll to the [-90,90] neighborhood (solveOnP can flip 180).
    def _wrap(a):
        if a > 90.0:
            a -= 180.0
        elif a < -90.0:
            a += 180.0
        return float(a)

    return _wrap(yaw), _wrap(pitch), _wrap(roll)


def _frontality_yaw_roll_only(kps5) -> float:
    """The original 5-kp yaw*roll estimate — used as a fallback."""
    if kps5 is None or len(kps5) < 3:
        return 0.0
    ea, eb, nose = kps5[0], kps5[1], kps5[2]
    eye_dx = abs(float(eb[0]) - float(ea[0]))
    if eye_dx < 1.0:
        return 0.0
    eye_mid_x = (float(ea[0]) + float(eb[0])) / 2.0
    nose_off = (float(nose[0]) - eye_mid_x) / eye_dx
    yaw_s = max(0.0, 1.0 - 2.0 * abs(nose_off))
    eye_dy = abs(float(eb[1]) - float(ea[1]))
    roll_s = max(0.0, 1.0 - eye_dy / eye_dx)
    return yaw_s * roll_s


def frontality_from_angles(yaw: float, pitch: float, roll: float) -> float:
    """Map head-pose angles (deg) to a [0,1] frontality. 1.0 = head-on."""
    pitch -= PITCH_OFFSET_DEG
    fy = max(0.0, 1.0 - abs(yaw) / YAW_LIMIT_DEG)
    fp = max(0.0, 1.0 - abs(pitch) / PITCH_LIMIT_DEG)
    fr = max(0.0, 1.0 - abs(roll) / ROLL_LIMIT_DEG)
    return float(fy * fp * fr)


def frontality(
    kps5,
    img_w: Optional[float] = None,
    img_h: Optional[float] = None,
    chin: Optional[np.ndarray] = None,
) -> float:
    """Pitch-aware frontality in [0,1]. Drop-in for selfie_logic.frontality().

    1.0 = head-on (yaw≈pitch≈roll≈0). Any axis going off tanks the score.
    Falls back to the old yaw*roll estimate if solveOnP can't run.
    """
    ang = head_pose_angles(kps5, img_w, img_h, chin)
    if ang is None:
        return _frontality_yaw_roll_only(kps5)
    return frontality_from_angles(*ang)


def chin_from_106(lms106_img: np.ndarray) -> Optional[np.ndarray]:
    """Chin point = lowest (max-y) 2d106det landmark, IN IMAGE-PIXEL COORDS.

    NOTE: vvad_speaker._landmarks returns crop-normalized [0,1] coords; multiply
    by the crop (w,h) and add the crop origin (x1,y1) to get image coords before
    passing here, so it lines up with the SCRFD keypoints.
    """
    if lms106_img is None or len(lms106_img) < 1:
        return None
    return np.asarray(
        lms106_img[int(np.argmax(lms106_img[:, 1]))][:2], dtype=np.float64
    )
