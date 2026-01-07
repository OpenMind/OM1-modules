# face_recog_stream/fall_detector.py
"""
Fall detection using human pose keypoints.

This module analyzes COCO-format pose keypoints to determine if a person
has fallen. Designed for deployment on moving robots (e.g., Unitree quadrupeds)
where traditional static camera assumptions don't apply.

Detection Strategy:
1. Body orientation: Check if torso is horizontal (shoulders near hip level)
2. Keypoint positions: Compare vertical positions of upper/lower body
3. Aspect ratio: Fallen people tend to have wider-than-tall bounding boxes
4. Confidence filtering: Only use keypoints with sufficient visibility

COCO Keypoint indices used:
    5: left_shoulder,  6: right_shoulder
   11: left_hip,      12: right_hip
   13: left_knee,     14: right_knee
   15: left_ankle,    16: right_ankle
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Keypoint indices
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16
KP_NOSE = 0

# Upper body keypoints: nose, eyes, ears, shoulders, elbows, wrists
UPPER_BODY_KPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Lower body keypoints: hips, knees, ankles
LOWER_BODY_KPS = [11, 12, 13, 14, 15, 16]


@dataclass
class FallStatus:
    """Fall detection result for a single person.

    Attributes
    ----------
    is_fallen : bool
        Whether the person is detected as fallen.
    confidence : float
        Confidence score of fall detection (0-1).
    reason : str
        Human-readable explanation of detection.
    bbox : tuple
        Bounding box (x1, y1, x2, y2).
    track_id : int
        Tracking ID if available, else -1.
    """

    is_fallen: bool
    confidence: float
    reason: str
    bbox: Tuple[float, float, float, float]
    track_id: int = -1


class FallDetector:
    """Analyze pose keypoints to detect fallen people.

    Parameters
    ----------
    horizontal_ratio_thr : float
        Torso angle threshold (shoulder-hip slope). Values closer to 0
        indicate horizontal orientation. Default 0.4.
    height_ratio_thr : float
        Ratio of (shoulder_y - ankle_y) / bbox_height. When shoulders
        are near or below ankle level, indicates fall. Default 0.3.
    aspect_ratio_thr : float
        Bounding box width/height ratio. Fallen people typically have
        ratio > 1.0. Default 1.2.
    kp_conf_thr : float
        Minimum keypoint visibility to use in calculations. Default 0.5.
    temporal_frames : int
        Number of frames to consider for temporal smoothing. Default 5.
    fall_frame_ratio : float
        Fraction of recent frames that must indicate fall. Default 0.6.
    """

    def __init__(
        self,
        horizontal_ratio_thr: float = 0.2,
        height_ratio_thr: float = 0.3,
        aspect_ratio_thr: float = 1.2,
        kp_conf_thr: float = 0.5,
        temporal_frames: int = 5,
        fall_frame_ratio: float = 0.5,
    ):
        self.horizontal_ratio_thr = float(horizontal_ratio_thr)
        self.height_ratio_thr = float(height_ratio_thr)
        self.aspect_ratio_thr = float(aspect_ratio_thr)
        self.kp_conf_thr = float(kp_conf_thr)
        self.temporal_frames = int(temporal_frames)
        self.fall_frame_ratio = float(fall_frame_ratio)

        # Per-detection temporal history: track_id -> deque of (timestamp, is_fallen)
        self._history: Dict[int, Deque[Tuple[float, bool]]] = {}
        self._lock = threading.Lock()
        self._next_id = 0

    def _get_valid_kp(self, kps: np.ndarray, idx: int) -> Optional[Tuple[float, float]]:
        """Get keypoint (x, y) if confidence exceeds threshold.

        Parameters
        ----------
        kps : ndarray (17, 3)
            Keypoints array [x, y, conf].
        idx : int
            Keypoint index.

        Returns
        -------
        tuple or None
            (x, y) if valid, else None.
        """
        if kps[idx, 2] >= self.kp_conf_thr:
            return float(kps[idx, 0]), float(kps[idx, 1])
        return None

    def _count_visible_keypoints(self, kps: np.ndarray) -> Tuple[int, int, int]:
        """Count visible keypoints in upper body, lower body, and total.

        Parameters
        ----------
        kps : ndarray (17, 3)
            Keypoints array [x, y, conf].

        Returns
        -------
        tuple
            (upper_count, lower_count, total_count)
        """
        upper_count = sum(
            1 for idx in UPPER_BODY_KPS if kps[idx, 2] >= self.kp_conf_thr
        )
        lower_count = sum(
            1 for idx in LOWER_BODY_KPS if kps[idx, 2] >= self.kp_conf_thr
        )
        total_count = upper_count + lower_count
        return upper_count, lower_count, total_count

    def _get_midpoint(
        self, kps: np.ndarray, idx1: int, idx2: int
    ) -> Optional[Tuple[float, float]]:
        """Get midpoint between two keypoints if both are valid.

        Parameters
        ----------
        kps : ndarray (17, 3)
            Keypoints array.
        idx1, idx2 : int
            Keypoint indices.

        Returns
        -------
        tuple or None
            Midpoint (x, y) if both valid, else None.
        """
        p1 = self._get_valid_kp(kps, idx1)
        p2 = self._get_valid_kp(kps, idx2)
        if p1 is None or p2 is None:
            return None
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def _check_horizontal_torso(self, kps: np.ndarray) -> Tuple[bool, float, str]:
        """Check if torso is horizontal (fallen orientation).

        Checks both sides of torso (left shoulder-hip, right shoulder-hip)
        and the midpoint line. If any is horizontal, considers it fallen.

        Returns
        -------
        tuple
            (is_horizontal, confidence, reason)
        """
        results = []

        # Check left side: left_shoulder to left_hip
        l_shoulder = self._get_valid_kp(kps, KP_L_SHOULDER)
        l_hip = self._get_valid_kp(kps, KP_L_HIP)
        if l_shoulder is not None and l_hip is not None:
            dx = abs(l_shoulder[0] - l_hip[0])
            dy = abs(l_shoulder[1] - l_hip[1])
            if dx > 1:  # avoid division issues
                slope = dy / (dx + 1e-6)
                if slope < self.horizontal_ratio_thr:
                    conf = 1.0 - (slope / self.horizontal_ratio_thr)
                    results.append(
                        (True, conf, f"left torso horizontal (slope={slope:.2f})")
                    )
                else:
                    results.append(
                        (False, 0.0, f"left torso vertical (slope={slope:.2f})")
                    )

        # Check right side: right_shoulder to right_hip
        r_shoulder = self._get_valid_kp(kps, KP_R_SHOULDER)
        r_hip = self._get_valid_kp(kps, KP_R_HIP)
        if r_shoulder is not None and r_hip is not None:
            dx = abs(r_shoulder[0] - r_hip[0])
            dy = abs(r_shoulder[1] - r_hip[1])
            if dx > 1:
                slope = dy / (dx + 1e-6)
                if slope < self.horizontal_ratio_thr:
                    conf = 1.0 - (slope / self.horizontal_ratio_thr)
                    results.append(
                        (True, conf, f"right torso horizontal (slope={slope:.2f})")
                    )
                else:
                    results.append(
                        (False, 0.0, f"right torso vertical (slope={slope:.2f})")
                    )

        # Check midpoint: shoulder_mid to hip_mid (if both sides partially visible)
        shoulder_mid = self._get_midpoint(kps, KP_L_SHOULDER, KP_R_SHOULDER)
        hip_mid = self._get_midpoint(kps, KP_L_HIP, KP_R_HIP)
        if shoulder_mid is not None and hip_mid is not None:
            dx = abs(shoulder_mid[0] - hip_mid[0])
            dy = abs(shoulder_mid[1] - hip_mid[1])
            if dx > 1:
                slope = dy / (dx + 1e-6)
                if slope < self.horizontal_ratio_thr:
                    conf = 1.0 - (slope / self.horizontal_ratio_thr)
                    results.append(
                        (True, conf, f"mid torso horizontal (slope={slope:.2f})")
                    )
                else:
                    results.append(
                        (False, 0.0, f"mid torso vertical (slope={slope:.2f})")
                    )

        if not results:
            return False, 0.0, "insufficient keypoints for torso check"

        # If any torso line is horizontal, return the one with highest confidence
        horizontal_results = [r for r in results if r[0]]
        if horizontal_results:
            best = max(horizontal_results, key=lambda x: x[1])
            return best

        # All checks say vertical - return one with lowest slope (closest to horizontal)
        return results[0]

    def _check_height_inversion(
        self, kps: np.ndarray, bbox: np.ndarray
    ) -> Tuple[bool, float, str]:
        """Check if upper body is at or below lower body level.

        Returns
        -------
        tuple
            (is_inverted, confidence, reason)
        """
        shoulder_mid = self._get_midpoint(kps, KP_L_SHOULDER, KP_R_SHOULDER)

        # Get lowest valid lower-body keypoint
        lower_kps = []
        for idx in [KP_L_ANKLE, KP_R_ANKLE, KP_L_KNEE, KP_R_KNEE]:
            p = self._get_valid_kp(kps, idx)
            if p is not None:
                lower_kps.append(p[1])  # y coordinate

        if shoulder_mid is None or not lower_kps:
            return False, 0.0, "insufficient keypoints for height check"

        shoulder_y = shoulder_mid[1]
        lowest_lower_y = max(lower_kps)  # max y = lowest point

        bbox_height = bbox[3] - bbox[1]
        if bbox_height < 10:
            return False, 0.0, "bbox too small"

        # Normalized difference: positive means shoulders below normal
        # In image coords, higher y = lower position
        height_diff = (shoulder_y - lowest_lower_y) / bbox_height

        if height_diff > -self.height_ratio_thr:
            # Shoulders are at or below leg level
            conf = min(1.0, (height_diff + self.height_ratio_thr) / 0.5)
            return True, conf, f"height inverted (diff={height_diff:.2f})"

        return False, 0.0, f"normal height (diff={height_diff:.2f})"

    def _check_aspect_ratio(self, bbox: np.ndarray) -> Tuple[bool, float, str]:
        """Check if bounding box aspect ratio indicates lying down.

        Returns
        -------
        tuple
            (is_wide, confidence, reason)
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        if h < 10:
            return False, 0.0, "bbox too small"

        ratio = w / h

        if ratio > self.aspect_ratio_thr:
            conf = min(1.0, (ratio - self.aspect_ratio_thr) / 0.5)
            return True, conf, f"wide bbox (ratio={ratio:.2f})"

        return False, 0.0, f"normal bbox (ratio={ratio:.2f})"

    def _update_history(self, track_id: int, is_fallen: bool) -> bool:
        """Update temporal history and return smoothed result.

        Parameters
        ----------
        track_id : int
            Detection/track identifier.
        is_fallen : bool
            Current frame detection result.

        Returns
        -------
        bool
            Temporally smoothed fall status.
        """
        now = time.time()

        with self._lock:
            if track_id not in self._history:
                self._history[track_id] = deque(maxlen=self.temporal_frames)

            hist = self._history[track_id]
            hist.append((now, is_fallen))

            # Clean old entries (>2 seconds old)
            while hist and (now - hist[0][0]) > 2.0:
                hist.popleft()

            if len(hist) < 2:
                return is_fallen

            # Count fall detections in recent frames
            fall_count = sum(1 for _, f in hist if f)
            fall_ratio = fall_count / len(hist)

            return fall_ratio >= self.fall_frame_ratio

    def detect_single(
        self,
        bbox: np.ndarray,
        kps: np.ndarray,
        track_id: int = -1,
    ) -> FallStatus:
        """Analyze one person's pose for fall detection.

        Parameters
        ----------
        bbox : ndarray (4,) or (5,)
            Bounding box [x1, y1, x2, y2, (conf)].
        kps : ndarray (17, 3)
            Keypoints [x, y, visibility].
        track_id : int
            Optional tracking ID for temporal smoothing.

        Returns
        -------
        FallStatus
            Detection result with confidence and explanation.
        """
        bbox = bbox[:4].astype(np.float32)

        # Count visible keypoints
        upper_count, lower_count, total_count = self._count_visible_keypoints(kps)

        # Require at least 3 visible keypoints to make any determination
        if total_count < 3:
            return FallStatus(
                is_fallen=False,
                confidence=0.0,
                reason=f"insufficient keypoints (total={total_count}, need>=3)",
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                track_id=track_id,
            )

        # Run torso and height checks (these already handle missing keypoints)
        h_fallen, h_conf, h_reason = self._check_horizontal_torso(kps)
        v_fallen, v_conf, v_reason = self._check_height_inversion(kps, bbox)

        # Only run aspect ratio check if we have sufficient upper AND lower body visibility
        # Requires at least 2 upper body AND 2 lower body keypoints
        a_fallen, a_conf, a_reason = (
            False,
            0.0,
            "aspect ratio skipped (need upper>=2 & lower>=2)",
        )
        if upper_count >= 2 and lower_count >= 3:
            a_fallen, a_conf, a_reason = self._check_aspect_ratio(bbox)
            signals = [
                (h_fallen, h_conf, 0.4, h_reason),
                (v_fallen, v_conf, 0.4, v_reason),
                (a_fallen, a_conf, 0.2, a_reason),
            ]

        # Combine signals (only include aspect ratio if it was actually checked)
        else:
            # Redistribute weights when aspect ratio is included
            signals = [
                (h_fallen, h_conf, 0.5, h_reason),  # horizontal torso: weight 0.5
                (v_fallen, v_conf, 0.5, v_reason),  # height inversion: weight 0.5
            ]

        weighted_sum = sum(s[0] * s[1] * s[2] for s in signals)
        total_weight = sum(s[2] for s in signals if s[1] > 0)

        if total_weight > 0:
            combined_conf = weighted_sum / total_weight
        else:
            combined_conf = 0.0

        is_fallen_raw = combined_conf > 0.5

        # Build reason string
        active_reasons = [s[3] for s in signals if s[0]]
        reason = "; ".join(active_reasons) if active_reasons else "normal"

        # Temporal smoothing
        if track_id >= 0:
            is_fallen = self._update_history(track_id, is_fallen_raw)
        else:
            # Assign temporary ID for this detection
            with self._lock:
                temp_id = self._next_id
                self._next_id += 1
            is_fallen = self._update_history(temp_id, is_fallen_raw)

        return FallStatus(
            is_fallen=is_fallen,
            confidence=combined_conf,
            reason=reason,
            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            track_id=track_id,
        )

    def detect_batch(
        self,
        dets: np.ndarray,
        keypoints: np.ndarray,
        track_ids: Optional[List[int]] = None,
    ) -> List[FallStatus]:
        """Analyze multiple people for fall detection.

        Parameters
        ----------
        dets : ndarray (N, 5)
            Detections [x1, y1, x2, y2, conf].
        keypoints : ndarray (N, 17, 3)
            Keypoints for each detection.
        track_ids : list[int], optional
            Tracking IDs for temporal smoothing.

        Returns
        -------
        list[FallStatus]
            Detection results for each person.
        """
        if dets is None or len(dets) == 0:
            return []

        results = []
        for i in range(len(dets)):
            tid = track_ids[i] if track_ids and i < len(track_ids) else -1
            status = self.detect_single(dets[i], keypoints[i], track_id=tid)
            results.append(status)

        return results

    def cleanup_old_tracks(self, active_ids: Optional[List[int]] = None) -> None:
        """Remove stale tracking history.

        Parameters
        ----------
        active_ids : list[int], optional
            Currently active track IDs. If provided, removes all others.
        """
        now = time.time()
        with self._lock:
            if active_ids is not None:
                active_set = set(active_ids)
                stale = [k for k in self._history if k not in active_set]
            else:
                # Remove tracks not updated in 5 seconds
                stale = [
                    k
                    for k, v in self._history.items()
                    if not v or (now - v[-1][0]) > 5.0
                ]

            for k in stale:
                del self._history[k]
