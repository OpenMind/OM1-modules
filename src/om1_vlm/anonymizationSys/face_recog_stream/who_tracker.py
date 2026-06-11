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

# Tier constants — kept in sync with selfie_logic.recognize_from_sims to avoid
# magic strings in this module.
TIER_CONFIDENT = "confident"
TIER_TENTATIVE = "tentative"
TIER_UNCERTAIN = "uncertain"


class WhoTracker:
    """Tracks face identities seen now and over a short lookback window.

    Inputs (via :meth:`update_now`) carry an optional tier per identity:
      - ``"confident"`` — high-confidence match (top1-top2 margin large)
      - ``"tentative"`` — likely match but borderline; caller should verify
      - ``"uncertain"`` / ``None`` — treated as unidentified for naming

    The single-frame fields (``now``, ``unknown_now``) are unchanged so
    legacy callers (fall-detection mapping, drawing) keep working. The new
    tier-aware fields (``confident_now``, ``tentative_now``) drive the
    FacePresence prompt and any robot-behavior gating.

    Temporal voting note:
      Per-track multi-frame voting is performed UPSTREAM in
      :class:`face_tracker.FaceTracker` (3-frame majority + tier). By the
      time names reach WhoTracker, they're already voted. This class
      doesn't re-aggregate across frames — it just maintains a sliding
      window for the ``recent_name_frames`` debug stats.
    """

    def __init__(self, lookback_sec: float = 10.0):
        self.lookback_sec = float(lookback_sec)
        # (ts, names[], tiers[], fall_infos[])
        # `tiers[i]` is the tier of `names[i]` ("confident"/"tentative"/
        # "uncertain" or None for legacy callers).
        self._events: Deque[Tuple[float, List[str], List[str], List[FallInfo]]] = deque(
            maxlen=300
        )
        self._last_now: List[str] = []
        self._last_tiers: List[str] = []
        self._last_falls: List[FallInfo] = []
        self._lock = threading.Lock()

    def update_now(
        self,
        names: List[Optional[str]],
        tiers: Optional[List[Optional[str]]] = None,
        fall_infos: Optional[List[FallInfo]] = None,
    ) -> None:
        """Update the current per-frame state.

        Parameters
        ----------
        names : list[str | None]
            One entry per visible face. ``"unknown"`` for unidentified faces,
            a label like ``"wendy"`` for identified ones. ``None`` entries
            are dropped.
        tiers : list[str | None], optional
            Parallel to ``names``. Each entry is one of
            ``"confident"``, ``"tentative"``, ``"uncertain"``, or ``None``
            (treated as "uncertain" for confident/tentative aggregation).
            If omitted, all identified names are treated as ``"confident"``
            for back-compat with callers that don't yet pass tier info.
        fall_infos : list[FallInfo], optional
            Fall-detection results aligned with ``names``.
        """
        now_ts = time.time()
        flat_names: List[str] = []
        flat_tiers: List[str] = []
        for i, n in enumerate(names):
            if n is None:
                continue
            flat_names.append(n)
            if tiers is not None and i < len(tiers) and tiers[i] is not None:
                flat_tiers.append(str(tiers[i]))
            else:
                # Legacy caller: assume confident for named entries, uncertain
                # for unknowns. Matches old behavior where any named entry was
                # treated as a positive identification.
                flat_tiers.append(TIER_CONFIDENT if n != "unknown" else TIER_UNCERTAIN)
        falls = fall_infos or []

        with self._lock:
            self._last_now = flat_names
            self._last_tiers = flat_tiers
            self._last_falls = falls
            self._events.append((now_ts, flat_names, flat_tiers, falls))
            cutoff = now_ts - self.lookback_sec
            while self._events and self._events[0][0] < cutoff:
                self._events.popleft()

    def snapshot(self, recent_sec: Optional[float] = None) -> Dict:
        """Summarize who is here now and over the recent window.

        Fields
        ------
        Per-frame (current frame only):
          now : list[str]
              Identified names in the latest frame (any tier). Kept for
              back-compat with fall-detection mapping. Flickers if face
              recognition is uncertain at frame boundaries.
          confident_now : list[str]
              Latest-frame names with tier "confident". Use this when you
              want the robot to greet by name without verification.
          tentative_now : list[str]
              Latest-frame names with tier "tentative". Use this to prompt
              user verification ("Are you Wendy?").
          unknown_now : int
              Count of unidentified faces in the latest frame.

        Windowed (over ``recent_sec`` seconds):
          frames_recent : int
          recent_name_frames : dict[str, int]
              How many recent frames each identified name appeared in
              (counted ONCE per frame, any tier). Kept for the debug UI.
          confident_frames : dict[str, int]
              Same but counts only frames where the tier was "confident".
          tentative_frames : dict[str, int]
              Same but counts only frames where the tier was "tentative".
          unknown_recent : int
              Peak per-frame unknown count over the window.

        Plus all existing fall-detection fields (unchanged).
        """
        with self._lock:
            now_list = list(self._last_now)
            now_tiers = list(self._last_tiers)
            now_falls = list(self._last_falls)

            if recent_sec is None:
                recent_sec = self.lookback_sec
            cutoff = time.time() - float(recent_sec)
            recent_data: List[Tuple[float, List[str], List[str], List[FallInfo]]] = [
                (ts, names, tiers, falls)
                for ts, names, tiers, falls in self._events
                if ts >= cutoff
            ]

        def is_named(x: str) -> bool:
            return bool(x) and x != "unknown"

        # ----- Latest-frame breakdown -----
        seen_in_now: set = set()
        now_named: List[str] = []
        confident_now: List[str] = []
        tentative_now: List[str] = []
        for name, tier in zip(now_list, now_tiers):
            if not is_named(name) or name in seen_in_now:
                continue
            seen_in_now.add(name)
            now_named.append(name)
            if tier == TIER_CONFIDENT:
                confident_now.append(name)
            elif tier == TIER_TENTATIVE:
                tentative_now.append(name)
        now_unknown = sum(1 for n in now_list if n == "unknown")

        # ----- Windowed (frames-based) stats -----
        frames_recent = len(recent_data)
        frames_with_unknown = 0
        unknown_recent_peak = 0
        recent_name_frames: Dict[str, int] = {}
        confident_frames: Dict[str, int] = {}
        tentative_frames: Dict[str, int] = {}

        for _ts, frame_names, frame_tiers, _frame_falls in recent_data:
            seen_frame: set = set()
            seen_confident: set = set()
            seen_tentative: set = set()
            for name, tier in zip(frame_names, frame_tiers):
                if not is_named(name):
                    continue
                # Dedup within a single frame — one face can only count once.
                seen_frame.add(name)
                if tier == TIER_CONFIDENT:
                    seen_confident.add(name)
                elif tier == TIER_TENTATIVE:
                    seen_tentative.add(name)
            for n in seen_frame:
                recent_name_frames[n] = recent_name_frames.get(n, 0) + 1
            for n in seen_confident:
                confident_frames[n] = confident_frames.get(n, 0) + 1
            for n in seen_tentative:
                tentative_frames[n] = tentative_frames.get(n, 0) + 1

            ucount = sum(1 for x in frame_names if x == "unknown")
            if ucount > 0:
                frames_with_unknown += 1
                if ucount > unknown_recent_peak:
                    unknown_recent_peak = ucount

        # ----- Fall detection (unchanged) -----
        fallen_now: List[str] = []
        fallen_unknown_now = 0
        for fall in now_falls:
            if fall.is_fallen:
                if fall.identity and fall.identity != "unknown":
                    if fall.identity not in fallen_now:
                        fallen_now.append(fall.identity)
                else:
                    fallen_unknown_now += 1

        frames_with_fall = 0
        fallen_recent: Dict[str, Dict] = {}
        for _ts, _frame_names, _frame_tiers, frame_falls in recent_data:
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
        for identity, stats in fallen_recent.items():
            stats["total_frames"] = recent_name_frames.get(
                identity, stats["fallen_frames"]
            )

        return {
            "server_ts": time.time(),
            "recent_sec": float(recent_sec),
            # Per-frame
            "now": now_named,
            "confident_now": confident_now,
            "tentative_now": tentative_now,
            "unknown_now": int(now_unknown),
            # Windowed
            "frames_recent": int(frames_recent),
            "frames_with_unknown": int(frames_with_unknown),
            "recent_name_frames": recent_name_frames,
            "confident_frames": confident_frames,
            "tentative_frames": tentative_frames,
            "unknown_recent": int(unknown_recent_peak),
            # Fall detection (unchanged)
            "fallen_now": fallen_now,
            "fallen_now_count": len(fallen_now) + fallen_unknown_now,
            "fallen_unknown_now": fallen_unknown_now,
            "fallen_recent": fallen_recent,
            "frames_with_fall": frames_with_fall,
            "alert": len(fallen_now) + fallen_unknown_now > 0,
        }
