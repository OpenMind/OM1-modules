"""
Face tracker with BoTSORT + low-frequency face recognition.

Architecture:
  Every frame:  SCRFD detect → BoTSORT track (persistent IDs)
  Low freq:     For unidentified tracks → AdaFace embed → vote → assign identity
  Once matched: Identity sticks to track ID until track is lost

Usage:
    tracker = FaceTracker(arc=arc_engine)
    tracker.set_gallery(feats, labels)
    # Per frame:
    results = tracker.update(frame, dets, kpss)
    # results = [TrackResult(track_id=1, bbox=..., name="wendy (0.72)", ...), ...]
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class TrackResult:
    """Per-track output each frame."""

    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    conf: float
    name: Optional[str] = None  # display name (e.g. "wendy (0.72)" or "wendy? (0.62)")
    sim: float = 0.0  # best similarity score
    is_known: bool = False  # True only after vote confirmed


@dataclass
class _TrackIdentity:
    """Internal identity state for one track."""

    track_id: int
    status: str = "new"  # new → identifying → identified / unknown
    name: Optional[str] = None
    sim: float = 0.0
    # Embedding votes collected during identification
    vote_names: List[str] = field(default_factory=list)
    vote_sims: List[float] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    # Timing
    first_seen: float = 0.0
    last_recog_time: float = 0.0
    frames_seen: int = 0
    recog_attempts: int = 0


class FaceTracker:
    """BoTSORT face tracker with low-frequency recognition and multi-frame voting.

    Parameters
    ----------
    arc : TRTFaceRecognition
        AdaFace TRT engine for embedding extraction.
    recog_interval : float
        Seconds between recognition attempts per track (default 0.3).
    vote_frames : int
        Recognition frames to collect before voting (default 3).
    vote_threshold : float
        Fraction of votes needed to confirm identity (default 0.5).
    max_recog_attempts : int
        Max attempts before marking as "unknown" (default 10).
    sim_thr : float
        Cosine similarity threshold for match (default 0.4).
    track_buffer : int
        BoTSORT frames to keep lost tracks alive (default 60).
    arc_max_bs : int
        Max batch size for AdaFace inference (default 8).
    """

    def __init__(
        self,
        arc,
        *,
        recog_interval: float = 0.3,
        vote_frames: int = 3,
        vote_threshold: float = 0.5,
        max_recog_attempts: int = 10,
        sim_thr: float = 0.4,
        track_buffer: int = 60,
        arc_max_bs: int = 8,
        det_conf: float = 0.5,
        re_identify_interval: float = 3.0,
    ):
        self.arc = arc
        self.recog_interval = float(recog_interval)
        self.vote_frames = int(vote_frames)
        self.vote_threshold = float(vote_threshold)
        self.max_recog_attempts = int(max_recog_attempts)
        self.sim_thr = float(sim_thr)
        self.arc_max_bs = int(arc_max_bs)
        self.re_identify_interval = float(re_identify_interval)

        # Gallery (set via set_gallery)
        self._gal_feats: Optional[np.ndarray] = None
        self._gal_labels: List[str] = []

        # BoTSORT tracker — align thresholds with SCRFD det_conf
        self._det_conf = float(det_conf)
        self._track_buffer = int(track_buffer)
        self._tracker = self._init_tracker(self._track_buffer, det_conf=self._det_conf)

        # Track identity state: track_id → _TrackIdentity
        self._identities: Dict[int, _TrackIdentity] = {}

        # Track IDs seen this frame (for cleanup)
        self._active_ids: set = set()

    @staticmethod
    def _init_tracker(track_buffer: int, det_conf: float = 0.5):
        """Initialize BoTSORT tracker with thresholds aligned to detection confidence."""
        from boxmot import BoTSORT

        high_thresh = max(det_conf, 0.3)
        new_thresh = max(det_conf, 0.3)
        low_thresh = max(det_conf * 0.5, 0.05)
        log.info(
            "BoTSORT thresholds: high=%.2f, new=%.2f, low=%.2f (det_conf=%.2f)",
            high_thresh,
            new_thresh,
            low_thresh,
            det_conf,
        )
        return BoTSORT(
            model_weights=None,
            device="cuda",
            fp16=False,
            track_high_thresh=high_thresh,
            track_low_thresh=low_thresh,
            new_track_thresh=new_thresh,
            track_buffer=track_buffer,
            match_thresh=0.85,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=False,
        )

    def set_gallery(self, feats: Optional[np.ndarray], labels: List[str]) -> None:
        """Update gallery embeddings and labels."""
        self._gal_feats = feats
        self._gal_labels = list(labels)

    def set_sim_thr(self, thr: float) -> None:
        """Update similarity threshold at runtime."""
        self.sim_thr = float(thr)

    def update(
        self,
        frame: np.ndarray,
        dets: np.ndarray,
        kpss: Optional[np.ndarray] = None,
    ) -> List[TrackResult]:
        """Process one frame: track + optionally recognize unidentified faces.

        Parameters
        ----------
        frame : ndarray (H, W, 3)
            BGR image.
        dets : ndarray (N, 5)
            SCRFD detections [x1, y1, x2, y2, conf].
        kpss : ndarray (N, 5, 2) or None
            5-point landmarks per detection.

        Returns
        -------
        list[TrackResult]
            One entry per tracked face.
        """
        now = time.time()
        H, W = frame.shape[:2]

        # Run BoTSORT
        tracks = self._run_tracker(dets, frame)

        # Match tracks to SCRFD detections (for landmarks)
        track_det_map = self._match_tracks_to_dets(tracks, dets, kpss)

        # Build results + collect tracks needing recognition
        self._active_ids = set()
        results: List[TrackResult] = []
        need_recog: List[Tuple[int, np.ndarray, Optional[np.ndarray]]] = []

        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            conf = float(track[5]) if len(track) > 5 else 0.0

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            self._active_ids.add(track_id)

            # Get or create identity state
            if track_id not in self._identities:
                self._identities[track_id] = _TrackIdentity(
                    track_id=track_id, first_seen=now
                )

            ident = self._identities[track_id]
            ident.frames_seen += 1

            # Re-identify: if "unknown" or low-confidence "identified" and enough
            # time has passed, reset voting state so recognition retries.
            # Handles the "walking closer" scenario where face gets clearer over time.
            if (
                self.re_identify_interval > 0
                and (now - ident.last_recog_time) >= self.re_identify_interval
            ):
                if ident.status == "unknown":
                    log.debug(
                        "Track %d: re-identify (was unknown, %.1fs since last attempt)",
                        track_id,
                        now - ident.last_recog_time,
                    )
                    ident.status = "new"
                    ident.vote_names.clear()
                    ident.vote_sims.clear()
                    ident.embeddings.clear()
                    ident.recog_attempts = 0
                elif ident.status == "identified" and ident.sim < self.sim_thr + 0.1:
                    # Low-confidence match — re-verify when face is clearer
                    log.debug(
                        "Track %d: re-verify '%s' (sim=%.3f, marginal)",
                        track_id,
                        ident.name,
                        ident.sim,
                    )
                    ident.status = "new"
                    ident.vote_names.clear()
                    ident.vote_sims.clear()
                    ident.embeddings.clear()
                    ident.recog_attempts = 0

            # Should we run recognition for this track?
            needs_recog = (
                ident.status in ("new", "identifying")
                and self._gal_feats is not None
                and self._gal_feats.size > 0
                and (now - ident.last_recog_time) >= self.recog_interval
                and ident.recog_attempts < self.max_recog_attempts
            )

            if needs_recog:
                det_kps = track_det_map.get(track_id, {}).get("kps")
                need_recog.append((track_id, np.array([x1, y1, x2, y2]), det_kps))

            # Build display name based on current status
            name_display, is_known, sim = self._get_display_name(ident)

            results.append(
                TrackResult(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    conf=conf,
                    name=name_display,
                    sim=sim,
                    is_known=is_known,
                )
            )

        # Batch recognition
        if need_recog and self.arc is not None:
            log.debug(f"Running recognition for {len(need_recog)} tracks")
            self._run_recognition_batch(frame, need_recog, now)
        elif need_recog:
            log.warning(f"need_recog={len(need_recog)} but arc is None")

        # Cleanup stale tracks
        self._cleanup_stale()

        return results

    def _get_display_name(
        self, ident: _TrackIdentity
    ) -> Tuple[Optional[str], bool, float]:
        """Get display name for a track based on its identity status.

        Returns (name_display, is_known, sim).
        """
        if ident.status == "identified" and ident.name:
            # Confirmed identity
            return f"{ident.name} ({ident.sim:.2f})", True, ident.sim

        elif ident.status == "identifying" and ident.vote_names:
            # Voting in progress — show temporary best guess with "?"
            real = [n for n in ident.vote_names if n != "__unknown__"]
            if real:
                top_name = max(set(real), key=real.count)
                top_sim = max(
                    s
                    for n, s in zip(ident.vote_names, ident.vote_sims)
                    if n == top_name
                )
                return f"{top_name}? ({top_sim:.2f})", False, top_sim
            return "unknown", False, 0.0

        elif ident.status == "unknown":
            return "unknown", False, 0.0

        else:
            # New track, no recognition yet
            return "unknown", False, 0.0

    def _run_tracker(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Run BoTSORT. Returns (M, 7) [x1,y1,x2,y2,track_id,conf,cls]."""
        if dets is None or len(dets) == 0:
            empty = np.empty((0, 6))
            tracks = self._tracker.update(empty, frame)
            return tracks if len(tracks) > 0 else np.empty((0, 7))

        # SCRFD (N,5) → BoTSORT (N,6) [x1,y1,x2,y2,conf,cls]
        cls_col = np.zeros((dets.shape[0], 1), dtype=np.float32)
        dets_6 = np.hstack([dets, cls_col])

        tracks = self._tracker.update(dets_6, frame)
        return tracks if len(tracks) > 0 else np.empty((0, 7))

    def _match_tracks_to_dets(
        self,
        tracks: np.ndarray,
        dets: np.ndarray,
        kpss: Optional[np.ndarray],
    ) -> Dict[int, dict]:
        """Match tracked boxes back to SCRFD detections to recover landmarks.

        Returns {track_id: {"det_idx": int, "kps": ndarray(5,2) or None}}
        """
        result = {}
        if tracks is None or len(tracks) == 0 or dets is None or len(dets) == 0:
            return result

        track_boxes = tracks[:, :4]
        det_boxes = dets[:, :4]

        for t_idx in range(len(tracks)):
            track_id = int(tracks[t_idx, 4])
            tb = track_boxes[t_idx]
            best_iou, best_d = 0.0, -1

            for d_idx in range(len(det_boxes)):
                db = det_boxes[d_idx]
                xx1 = max(tb[0], db[0])
                yy1 = max(tb[1], db[1])
                xx2 = min(tb[2], db[2])
                yy2 = min(tb[3], db[3])
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                area_t = (tb[2] - tb[0]) * (tb[3] - tb[1])
                area_d = (db[2] - db[0]) * (db[3] - db[1])
                iou = inter / (area_t + area_d - inter + 1e-9)

                if iou > best_iou:
                    best_iou = iou
                    best_d = d_idx

            if best_d >= 0 and best_iou > 0.3:
                entry = {"det_idx": best_d}
                entry["kps"] = (
                    kpss[best_d] if kpss is not None and best_d < len(kpss) else None
                )
                result[track_id] = entry

        return result

    # Recognition

    def _run_recognition_batch(
        self,
        frame: np.ndarray,
        need_recog: List[Tuple[int, np.ndarray, Optional[np.ndarray]]],
        now: float,
    ) -> None:
        """Run AdaFace on unidentified tracks and update votes."""
        from .adaface import warp_face_by_5p

        crops: List[np.ndarray] = []
        crop_track_ids: List[int] = []
        crop_bboxes: List[np.ndarray] = []

        for track_id, bbox, kps in need_recog:
            try:
                x1, y1, x2, y2 = bbox
                if kps is not None:
                    crop = warp_face_by_5p(frame, kps, 112)
                else:
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue
                    crop = cv2.resize(face, (112, 112))
                crops.append(crop)
                crop_track_ids.append(track_id)
                crop_bboxes.append(bbox)
            except Exception:
                continue

        if not crops:
            return

        # Batch inference
        all_feats = []
        for i in range(0, len(crops), self.arc_max_bs):
            batch = crops[i : i + self.arc_max_bs]
            vecs = self.arc.infer(batch)
            all_feats.append(vecs)

        if not all_feats:
            return
        feats = (
            np.concatenate(all_feats, axis=0) if len(all_feats) > 1 else all_feats[0]
        )

        # Match against gallery
        if self._gal_feats is None or self._gal_feats.size == 0:
            return

        S = feats @ self._gal_feats.T
        best_j = np.argmax(S, axis=1)
        best_v = S[np.arange(S.shape[0]), best_j]

        for idx, track_id in enumerate(crop_track_ids):
            ident = self._identities.get(track_id)
            if ident is None:
                continue

            ident.last_recog_time = now
            ident.recog_attempts += 1
            ident.status = "identifying"

            sim_val = float(best_v[idx])
            label_idx = int(best_j[idx])

            ident.embeddings.append(feats[idx])

            log.debug(
                f"Track {ident.track_id} bbox={crop_bboxes[idx][:4]}: sim={sim_val:.3f} best='{self._gal_labels[label_idx]}' → "
                f"{'MATCH' if sim_val >= self.sim_thr else 'NO MATCH'}"
            )
            if sim_val >= self.sim_thr:
                ident.vote_names.append(self._gal_labels[label_idx])
                ident.vote_sims.append(sim_val)
            else:
                ident.vote_names.append("__unknown__")
                ident.vote_sims.append(sim_val)

            # Check if ready to decide
            if (
                len(ident.vote_names) >= self.vote_frames
                or ident.recog_attempts >= self.max_recog_attempts
            ):
                self._finalize_identity(ident)

    def _finalize_identity(self, ident: _TrackIdentity) -> None:
        """Decide identity from collected votes using majority voting."""
        if not ident.vote_names:
            ident.status = "unknown"
            ident.name = None
            return

        real_votes = [n for n in ident.vote_names if n != "__unknown__"]

        if not real_votes:
            ident.status = "unknown"
            ident.name = None
            log.debug(f"Track {ident.track_id}: all votes unknown")
            return

        counter = Counter(real_votes)
        best_name, best_count = counter.most_common(1)[0]
        total_votes = len(ident.vote_names)
        ratio = best_count / total_votes

        if ratio >= self.vote_threshold:
            winning_sims = [
                s for n, s in zip(ident.vote_names, ident.vote_sims) if n == best_name
            ]
            avg_sim = sum(winning_sims) / len(winning_sims) if winning_sims else 0.0

            ident.status = "identified"
            ident.name = best_name
            ident.sim = avg_sim
            log.info(
                f"Track {ident.track_id}: confirmed '{best_name}' "
                f"(votes={best_count}/{total_votes}, sim={avg_sim:.3f})"
            )
        else:
            ident.status = "unknown"
            ident.name = None
            log.debug(
                f"Track {ident.track_id}: no majority ({best_name} {best_count}/{total_votes})"
            )

    def _cleanup_stale(self) -> None:
        """Remove identity entries for tracks no longer active."""
        stale = [tid for tid in self._identities if tid not in self._active_ids]
        for tid in stale:
            del self._identities[tid]

    def get_track_identities(self) -> Dict[int, dict]:
        """Get current identity state for all active tracks."""
        return {
            tid: {
                "name": ident.name,
                "status": ident.status,
                "sim": ident.sim,
                "frames_seen": ident.frames_seen,
                "recog_attempts": ident.recog_attempts,
                "votes": len(ident.vote_names),
            }
            for tid, ident in self._identities.items()
            if tid in self._active_ids
        }

    def reset(self) -> None:
        """Reset all tracking state."""
        self._identities.clear()
        self._active_ids.clear()
        if self._tracker is not None:
            self._tracker = self._init_tracker(
                self._track_buffer, det_conf=self._det_conf
            )
