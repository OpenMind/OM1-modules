"""
Drawing utilities for pose keypoints and fall detection overlays.

This module provides functions to visualize COCO pose keypoints, skeleton
connections, and fall detection status on video frames.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .fall_detector import FallStatus

# COCO skeleton connections (pairs of keypoint indices)
SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # head
    (5, 6),  # shoulders
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (5, 11),
    (6, 12),  # torso sides
    (11, 12),  # hips
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
]

# Colors for skeleton limbs (BGR)
LIMB_COLORS = [
    (255, 128, 0),
    (255, 128, 0),
    (255, 128, 0),
    (255, 128, 0),  # head - orange
    (0, 255, 0),  # shoulders - green
    (255, 255, 0),
    (255, 255, 0),  # left arm - cyan
    (0, 255, 255),
    (0, 255, 255),  # right arm - yellow
    (255, 0, 255),
    (255, 0, 255),  # torso - magenta
    (255, 0, 255),  # hips
    (0, 128, 255),
    (0, 128, 255),  # left leg - orange-red
    (128, 0, 255),
    (128, 0, 255),  # right leg - purple
]

# Keypoint colors by body region
KP_COLORS = [
    (255, 128, 0),  # 0: nose
    (255, 128, 0),  # 1: left_eye
    (255, 128, 0),  # 2: right_eye
    (255, 128, 0),  # 3: left_ear
    (255, 128, 0),  # 4: right_ear
    (0, 255, 0),  # 5: left_shoulder
    (0, 255, 0),  # 6: right_shoulder
    (255, 255, 0),  # 7: left_elbow
    (0, 255, 255),  # 8: right_elbow
    (255, 255, 0),  # 9: left_wrist
    (0, 255, 255),  # 10: right_wrist
    (255, 0, 255),  # 11: left_hip
    (255, 0, 255),  # 12: right_hip
    (0, 128, 255),  # 13: left_knee
    (128, 0, 255),  # 14: right_knee
    (0, 128, 255),  # 15: left_ankle
    (128, 0, 255),  # 16: right_ankle
]


def draw_skeleton(
    img: np.ndarray,
    kps: np.ndarray,
    kp_conf_thr: float = 0.5,
    thickness: int = 2,
    radius: int = 4,
) -> np.ndarray:
    """
    Draw skeleton connections and keypoints on an image.

    Parameters
    ----------
    img : ndarray
        BGR image (modified in-place).
    kps : ndarray (17, 3)
        Keypoints [x, y, conf] for one person.
    kp_conf_thr : float
        Minimum confidence to draw a keypoint.
    thickness : int
        Line thickness for skeleton.
    radius : int
        Keypoint circle radius.

    Returns
    -------
    ndarray
        The same image for chaining.
    """
    # Draw skeleton lines first (so keypoints overlay them)
    for i, (p1_idx, p2_idx) in enumerate(SKELETON):
        if kps[p1_idx, 2] < kp_conf_thr or kps[p2_idx, 2] < kp_conf_thr:
            continue
        x1, y1 = int(kps[p1_idx, 0]), int(kps[p1_idx, 1])
        x2, y2 = int(kps[p2_idx, 0]), int(kps[p2_idx, 1])

        color = LIMB_COLORS[i] if i < len(LIMB_COLORS) else (128, 128, 128)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # Draw keypoints
    for i in range(17):
        if kps[i, 2] < kp_conf_thr:
            continue
        x, y = int(kps[i, 0]), int(kps[i, 1])
        color = KP_COLORS[i] if i < len(KP_COLORS) else (255, 255, 255)
        cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), radius, (0, 0, 0), 1, cv2.LINE_AA)  # outline

    return img


def draw_pose_box(
    img: np.ndarray,
    bbox: np.ndarray,
    label: Optional[str] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a bounding box with optional label.

    Parameters
    ----------
    img : ndarray
        BGR image (modified in-place).
    bbox : ndarray (4,) or (5,)
        Box coordinates [x1, y1, x2, y2, (conf)].
    label : str, optional
        Text label to display above box.
    color : tuple
        BGR color for box and label.
    thickness : int
        Line thickness.

    Returns
    -------
    ndarray
        The same image for chaining.
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    if label:
        fs = 0.6
        txt_thick = max(1, thickness - 1)
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, fs, txt_thick
        )
        tx, ty = x1, max(th + 4, y1 - 6)

        # Background rectangle
        cv2.rectangle(
            img,
            (tx - 1, ty - th - baseline - 2),
            (tx + tw + 1, ty + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            img,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            color,
            txt_thick,
            cv2.LINE_AA,
        )

    return img


def draw_fall_status(
    img: np.ndarray,
    status: FallStatus,
    draw_box: bool = True,
    draw_label: bool = True,
) -> np.ndarray:
    """
    Draw fall detection status on an image.

    Parameters
    ----------
    img : ndarray
        BGR image (modified in-place).
    status : FallStatus
        Fall detection result.
    draw_box : bool
        Whether to draw bounding box.
    draw_label : bool
        Whether to draw status label.

    Returns
    -------
    ndarray
        The same image for chaining.
    """
    if status.is_fallen:
        color = (0, 0, 255)  # red for fallen
        label = f"FALLEN CONF: ({status.confidence:.2f})"
    else:
        color = (0, 255, 0)  # green for standing
        label = f"OK FALLEN CONF: ({status.confidence:.2f})"

    bbox = np.array(status.bbox)

    if draw_box:
        thickness = 3 if status.is_fallen else 2
        draw_pose_box(img, bbox, label if draw_label else None, color, thickness)

    # Draw warning overlay for fallen detection
    if status.is_fallen:
        x1, y1, x2, y2 = map(int, bbox)
        # Semi-transparent red overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

    return img


def draw_pose_overlays(
    img: np.ndarray,
    dets: Optional[np.ndarray],
    keypoints: Optional[np.ndarray],
    fall_statuses: Optional[List[FallStatus]] = None,
    draw_skeleton: bool = True,
    draw_boxes: bool = True,
    draw_fall_status: bool = True,
    kp_conf_thr: float = 0.5,
) -> np.ndarray:
    """
    Draw all pose-related overlays on an image.

    Parameters
    ----------
    img : ndarray
        BGR image (modified in-place).
    dets : ndarray (N, 5), optional
        Pose detections [x1, y1, x2, y2, conf].
    keypoints : ndarray (N, 17, 3), optional
        Keypoints for each detection.
    fall_statuses : list[FallStatus], optional
        Fall detection results aligned with detections.
    draw_skeleton : bool
        Whether to draw skeleton lines and keypoints.
    draw_boxes : bool
        Whether to draw bounding boxes.
    draw_fall_status : bool
        Whether to draw fall status overlays.
    kp_conf_thr : float
        Minimum keypoint confidence for drawing.

    Returns
    -------
    ndarray
        The same image for chaining.
    """
    if dets is None or len(dets) == 0:
        return img

    H, W = img.shape[:2]
    thickness = max(1, int(round(min(H, W) / 400.0)))
    radius = max(2, int(round(min(H, W) / 200.0)))

    for i in range(len(dets)):
        # Draw fall status first (includes semi-transparent overlay)
        if draw_fall_status and fall_statuses and i < len(fall_statuses):
            status = fall_statuses[i]
            if status.is_fallen:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            if draw_boxes:
                label = f"{'FALLEN' if status.is_fallen else 'OK'} ({status.confidence:.2f})"
                draw_pose_box(
                    img,
                    dets[i],
                    label,
                    color,
                    thickness=3 if status.is_fallen else thickness,
                )

            if status.is_fallen:
                x1, y1, x2, y2 = map(int, dets[i][:4])
                overlay = img.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        elif draw_boxes:
            conf = dets[i, 4] if dets.shape[1] > 4 else 0.0
            draw_pose_box(img, dets[i], f"person {conf:.2f}", (255, 200, 0), thickness)

        # Draw skeleton on top
        if draw_skeleton and keypoints is not None and i < len(keypoints):
            # Use the module-level function to avoid name collision
            _draw_skeleton_internal(img, keypoints[i], kp_conf_thr, thickness, radius)

    return img


def _draw_skeleton_internal(
    img: np.ndarray,
    kps: np.ndarray,
    kp_conf_thr: float,
    thickness: int,
    radius: int,
) -> None:
    """
    Internal skeleton drawing helper
    """
    # Draw skeleton lines
    for i, (p1_idx, p2_idx) in enumerate(SKELETON):
        if kps[p1_idx, 2] < kp_conf_thr or kps[p2_idx, 2] < kp_conf_thr:
            continue
        x1, y1 = int(kps[p1_idx, 0]), int(kps[p1_idx, 1])
        x2, y2 = int(kps[p2_idx, 0]), int(kps[p2_idx, 1])
        color = LIMB_COLORS[i] if i < len(LIMB_COLORS) else (128, 128, 128)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # Draw keypoints
    for i in range(17):
        if kps[i, 2] < kp_conf_thr:
            continue
        x, y = int(kps[i, 0]), int(kps[i, 1])
        color = KP_COLORS[i] if i < len(KP_COLORS) else (255, 255, 255)
        cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), radius, (0, 0, 0), 1, cv2.LINE_AA)


def draw_fall_alert(
    img: np.ndarray,
    fall_count: int,
    margin: int = 10,
) -> np.ndarray:
    """Draw a global fall alert banner in upper-right corner.

    Parameters
    ----------
    img : ndarray
        BGR image (modified in-place).
    fall_count : int
        Number of people detected as fallen.
    margin : int
        Margin from edge of image.

    Returns
    -------
    ndarray
        The same image for chaining.
    """
    if fall_count == 0:
        return img

    H, W = img.shape[:2]
    text = f"FALL DETECTED: {fall_count} person(s)"
    fs = 1.0
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thickness)

    # Position in upper-right corner
    x = W - tw - margin - 10
    y = margin + th + 5

    # Red background
    cv2.rectangle(img, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 200), -1)
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return img
