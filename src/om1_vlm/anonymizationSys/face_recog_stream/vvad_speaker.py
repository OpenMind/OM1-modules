import logging
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mediapipe_mesh import MediaPipeFaceMesh
from .mouth_features import LandmarkScheme, MouthFeature, extract_mouth_feature
from .speaking_vad import DetectorConfig, SpeakingDetector

log = logging.getLogger(__name__)


class VVADScorer:
    """
    Per-track visual speaking detector.

    push_frame(), evict(), score_track(), score_window(), resolve_speaking(),
    resolve_window(), and the `speaking_thr` attribute.
    """

    def __init__(
        self,
        speaking_thr: float = 0.5,
        window_sec: float = 1.0,
        buffer_sec: float = 6.0,
        min_window_samples: int = 4,
        fps: float = 15.0,
    ):
        self.speaking_thr = float(speaking_thr)
        self.window_sec = float(window_sec)
        self.buffer_sec = max(float(buffer_sec), self.window_sec)
        self.fps = float(fps)
        self._min_window_samples = int(min_window_samples)

        log.info(
            "VVAD(speaking_detector) landmark backend = MediaPipeFaceMesh, "
            "run per-frame (thr=%.2f, window=%.1fs, buffer=%.1fs, fps=%.1f)",
            self.speaking_thr,
            self.window_sec,
            self.buffer_sec,
            self.fps,
        )

        self._scheme = LandmarkScheme()
        self._lock = threading.Lock()

        self._mesh: Optional[MediaPipeFaceMesh] = None
        self._mesh_capacity = 0

        self._live: Dict[int, SpeakingDetector] = {}
        self._hist: Dict[int, deque] = {}

    def _new_detector(self, fps: Optional[float] = None) -> SpeakingDetector:
        cfg = DetectorConfig(
            fps=float(fps) if fps else self.fps,
            window_sec=self.window_sec,
        )
        return SpeakingDetector(cfg)

    def _get_mesh(self, capacity: int) -> MediaPipeFaceMesh:
        capacity = max(capacity, 1)
        with self._lock:
            if self._mesh is None or self._mesh_capacity < capacity:
                if self._mesh is not None:
                    try:
                        self._mesh.close()
                    except Exception:
                        pass
                self._mesh = MediaPipeFaceMesh(max_num_faces=capacity)
                self._mesh_capacity = capacity
            return self._mesh

    @staticmethod
    def _match_faces_to_tracks(
        faces: List[np.ndarray],
        track_boxes: List[Tuple[int, Tuple[float, float, float, float]]],
    ) -> Dict[int, np.ndarray]:
        """
        Assign each detected face's landmarks to its nearest track by
        center distance — greedy, globally-nearest-pair-first, each track
        claims at most one face.
        """
        assigned: Dict[int, np.ndarray] = {}
        if not faces or not track_boxes:
            return assigned

        centers = [
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0) for _tid, (x1, y1, x2, y2) in track_boxes
        ]
        pairs = []  # (dist_sq, face_idx, track_idx)
        for fi, lm in enumerate(faces):
            cx, cy = float(lm[:, 0].mean()), float(lm[:, 1].mean())
            for ti, (bcx, bcy) in enumerate(centers):
                d = (cx - bcx) ** 2 + (cy - bcy) ** 2
                pairs.append((d, fi, ti))
        pairs.sort(key=lambda p: p[0])

        used_faces, used_tracks = set(), set()
        for _d, fi, ti in pairs:
            if fi in used_faces or ti in used_tracks:
                continue
            used_faces.add(fi)
            used_tracks.add(ti)
            assigned[track_boxes[ti][0]] = faces[fi]
        return assigned

    # buffer
    def push_frame(
        self,
        frame_bgr: np.ndarray,
        track_boxes: List[Tuple[int, Tuple[float, float, float, float]]],
    ) -> None:
        """Feed one frame for ALL currently-visible tracks at once.

        Parameters
        ----------
        frame_bgr : the full camera frame (BGR), NOT a per-face crop.
        track_boxes : list of (track_id, (x1, y1, x2, y2)) for every track
            active this frame.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return

        faces: List[np.ndarray] = []
        if track_boxes:
            mesh = self._get_mesh(len(track_boxes))
            faces = mesh.all(frame_bgr)
        assigned = self._match_faces_to_tracks(faces, track_boxes)

        now = time.time()
        for track_id, _bbox in track_boxes:
            lm = assigned.get(track_id)
            feat: Optional[MouthFeature] = None
            if lm is not None:
                raw_feat = extract_mouth_feature(lm, scheme=self._scheme)
                if raw_feat.valid:
                    feat = raw_feat

            with self._lock:
                det = self._live.get(track_id)
                if det is None:
                    det = self._new_detector()
                    self._live[track_id] = det
                det.update(feat)

                hist = self._hist.get(track_id)
                if hist is None:
                    hist = deque()
                    self._hist[track_id] = hist
                hist.append((now, feat))
                cutoff = now - self.buffer_sec
                while hist and hist[0][0] < cutoff:
                    hist.popleft()

    def evict(self, alive_track_ids=None) -> None:
        """Drop buffered state for tracks not in alive_track_ids (or all if None)."""
        with self._lock:
            if alive_track_ids is None:
                self._live.clear()
                self._hist.clear()
                if self._mesh is not None:
                    try:
                        self._mesh.close()
                    except Exception:
                        pass
                    self._mesh = None
                    self._mesh_capacity = 0
                return
            alive = set(int(t) for t in alive_track_ids)
            for tid in [t for t in self._live if t not in alive]:
                del self._live[tid]
            for tid in [t for t in self._hist if t not in alive]:
                del self._hist[tid]

    def score_track(self, track_id: int) -> Optional[float]:
        """Current ("right now") speaking score for this track."""
        with self._lock:
            det = self._live.get(track_id)
        return float(det.state.score) if det is not None else None

    def is_speaking(self, track_id: int) -> bool:
        """Current ("right now") hysteresis-debounced speaking state."""
        with self._lock:
            det = self._live.get(track_id)
        return bool(det.state.speaking) if det is not None else False

    def score_window(
        self, track_id: int, win_start: float, win_end: float
    ) -> Optional[float]:
        """Score an explicit [start, end] epoch span by replaying the
        buffered features for that span through a fresh detector (so the
        result depends only on that window, independent of live state).
        """
        with self._lock:
            hist = self._hist.get(track_id)
            span = (
                [(ts, f) for ts, f in hist if win_start <= ts <= win_end]
                if hist
                else []
            )
        span = [(ts, f) for ts, f in span if f is not None]
        if len(span) < self._min_window_samples:
            return None

        dur = span[-1][0] - span[0][0]
        fps = (len(span) - 1) / dur if dur > 1e-3 else self.fps
        det = self._new_detector(fps=fps)
        state = None
        for _, feat in span:
            state = det.update(feat)
        return float(state.score) if state is not None else None

    def resolve_speaking(
        self, candidate_track_ids: List[int]
    ) -> Tuple[Optional[int], Dict[int, float]]:
        """Score each candidate track (current state) and return the top
        speaker (if above threshold) with all scores.
        """
        scores: Dict[int, float] = {}
        for tid in candidate_track_ids:
            s = self.score_track(int(tid))
            if s is not None:
                scores[int(tid)] = s
        if not scores:
            return None, scores
        best = max(scores, key=lambda tid: scores[tid])
        return (best if scores[best] >= self.speaking_thr else None), scores

    def resolve_window(
        self, candidate_track_ids: List[int], win_start: float, win_end: float
    ) -> Dict[int, float]:
        """Score each candidate track over the given [win_start, win_end] span."""
        scores: Dict[int, float] = {}
        for tid in candidate_track_ids:
            s = self.score_window(int(tid), win_start, win_end)
            if s is not None:
                scores[int(tid)] = s
        return scores
