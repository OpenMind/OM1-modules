"""
Unified tracker for face identities and fall detection status.

Combines face recognition results with fall detection to provide:
- Who is present now
- Who has fallen (with identity if face is matched)
- Historical statistics for both
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

# Head keypoint indices for matching (COCO format)
HEAD_KP_INDICES = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear


@dataclass
class FallInfo:
    """Fall detection info for one person."""

    identity: Optional[str]  # Matched face name or None
    is_fallen: bool


def _expand_bbox(bbox, margin: float):
    """Expand bbox by margin fraction."""
    x1, y1, x2, y2 = bbox[:4]
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin, h * margin
    return (x1 - mx, y1 - my, x2 + mx, y2 + my)


def _point_in_bbox(px: float, py: float, bbox) -> bool:
    """Check if point is inside bbox."""
    return bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]


def match_falls_to_faces(
    fall_statuses: List,
    pose_keypoints: Optional[np.ndarray],
    face_bboxes: Optional[np.ndarray],
    face_names: List[Optional[str]],
    kp_conf_thr: float = 0.5,
    bbox_margin: float = 0.25,
) -> List[FallInfo]:
    """Match fall detections to face identities using head keypoints.

    Method:
    - Get head keypoints (nose, eyes, ears) from each pose
    - Expand face bbox by margin (default 25%)
    - Check if head keypoints fall inside expanded face bbox
    - Assign identity from matched face

    Returns
    -------
    List[FallInfo]
        Fall info for each detected person with matched identity.
    """
    if not fall_statuses:
        return []

    results = []
    used_faces = set()

    for f_idx, status in enumerate(fall_statuses):
        matched_identity = None

        # Try to match using head keypoints
        if (
            pose_keypoints is not None
            and f_idx < len(pose_keypoints)
            and face_bboxes is not None
            and len(face_bboxes) > 0
        ):
            kps = pose_keypoints[f_idx]

            # Get valid head keypoints
            head_points = []
            for kp_idx in HEAD_KP_INDICES:
                if kps[kp_idx, 2] >= kp_conf_thr:
                    head_points.append((float(kps[kp_idx, 0]), float(kps[kp_idx, 1])))

            if head_points:
                best_face_idx = -1
                best_match_count = 0

                for face_idx in range(len(face_bboxes)):
                    if face_idx in used_faces:
                        continue

                    expanded = _expand_bbox(face_bboxes[face_idx], bbox_margin)
                    match_count = sum(
                        1 for px, py in head_points if _point_in_bbox(px, py, expanded)
                    )

                    if match_count > best_match_count:
                        best_match_count = match_count
                        best_face_idx = face_idx

                if best_face_idx >= 0 and best_match_count >= 1:
                    used_faces.add(best_face_idx)
                    if best_face_idx < len(face_names) and face_names[best_face_idx]:
                        matched_identity = face_names[best_face_idx]

        results.append(
            FallInfo(
                identity=matched_identity,
                is_fallen=status.is_fallen,
            )
        )

    return results


# ------------------------------- Who Tracker ------------------------------- #
class WhoTracker:
    """Tracks identities seen now and over a short lookback window."""

    def __init__(self, lookback_sec: float = 10.0):
        self.lookback_sec = float(lookback_sec)
        # (ts, names[], fall_infos[])
        self._events: Deque[Tuple[float, List[str], List[FallInfo]]] = deque(maxlen=300)
        self._last_now: List[str] = []
        self._last_falls: List[FallInfo] = []
        self._lock = threading.Lock()

    def update_now(
        self,
        names: List[Optional[str]],
        fall_infos: Optional[List[FallInfo]] = None,
    ) -> None:
        """Update current identities; names may include 'unknown' or None."""
        now_ts = time.time()
        flat: List[str] = [n for n in names if n is not None]
        falls = fall_infos or []

        with self._lock:
            self._last_now = flat
            self._last_falls = falls
            self._events.append((now_ts, flat, falls))
            cutoff = now_ts - self.lookback_sec
            while self._events and self._events[0][0] < cutoff:
                self._events.popleft()

    def snapshot(self, recent_sec: Optional[float] = None) -> Dict:
        """
        Summarize who is here now and (for recent) use max-per-frame semantics.

        EX: {"server_ts": 1761692303.4755309, "recent_sec": 4.0, "now": ["wendy"],
            "unknown_now": 0, "frames_recent": 57, "frames_with_unknown": 0,
            "recent_name_frames": {"wendy": 57}, "unknown_recent": 0}

        Parameters
        ----------
        recent_sec : Optional[float]
            Lookback window in seconds for recent stats. If None, uses self.lookback_sec.
        """
        with self._lock:
            now_list = list(self._last_now)
            now_falls = list(self._last_falls)

            if recent_sec is None:
                recent_sec = self.lookback_sec
            cutoff = time.time() - float(recent_sec)
            recent_data: List[Tuple[float, List[str], List[FallInfo]]] = [
                (ts, names, falls) for ts, names, falls in self._events if ts >= cutoff
            ]

        def is_named(x: str) -> bool:
            return bool(x) and x != "unknown"

        # Latest frame breakdown
        seen_in_now = set()
        now_named: List[str] = []
        for n in now_list:
            if is_named(n) and n not in seen_in_now:
                seen_in_now.add(n)
                now_named.append(n)
        now_unknown = sum(1 for n in now_list if n == "unknown")

        # Windowed (frames-based) stats
        frames_recent = len(recent_data)
        frames_with_unknown = 0
        unknown_recent_peak = 0
        recent_name_frames: Dict[str, int] = {}

        for ts, frame_names, frame_falls in recent_data:
            # Per-frame known set (dedup within the frame)
            kset = {n for n in frame_names if is_named(n)}
            for k in kset:
                recent_name_frames[k] = recent_name_frames.get(k, 0) + 1

            # Unknown presence and peak (per frame)
            ucount = sum(1 for n in frame_names if n == "unknown")
            if ucount > 0:
                frames_with_unknown += 1
                if ucount > unknown_recent_peak:
                    unknown_recent_peak = ucount

        # Current frame fall detection
        fallen_now = []
        fallen_unknown_now = 0
        for fall in now_falls:
            if fall.is_fallen:
                if fall.identity and fall.identity != "unknown":
                    if fall.identity not in fallen_now:
                        fallen_now.append(fall.identity)
                else:
                    fallen_unknown_now += 1

        # Recent frames fall detection
        frames_with_fall = 0
        fallen_recent: Dict[str, Dict] = {}

        for ts, frame_names, frame_falls in recent_data:
            frame_has_fall = False

            for fall in frame_falls:
                if not fall.is_fallen:
                    continue

                frame_has_fall = True
                identity = fall.identity if fall.identity else "unknown"

                if identity not in fallen_recent:
                    fallen_recent[identity] = {
                        "fallen_frames": 0,
                        "total_frames": 0,
                    }

                fallen_recent[identity]["fallen_frames"] += 1

            if frame_has_fall:
                frames_with_fall += 1

        # Set total_frames for each identity
        for identity, stats in fallen_recent.items():
            total = recent_name_frames.get(identity, stats["fallen_frames"])
            stats["total_frames"] = total

        return {
            "server_ts": time.time(),
            "recent_sec": float(recent_sec),
            "now": now_named,
            "unknown_now": int(now_unknown),
            "frames_recent": int(frames_recent),
            "frames_with_unknown": int(frames_with_unknown),
            "recent_name_frames": recent_name_frames,
            "unknown_recent": int(unknown_recent_peak),
            "fallen_now": fallen_now,
            "fallen_now_count": len(fallen_now) + fallen_unknown_now,
            "fallen_unknown_now": fallen_unknown_now,
            "fallen_recent": fallen_recent,
            "frames_with_fall": frames_with_fall,
            "alert": len(fallen_now) + fallen_unknown_now > 0,
        }
