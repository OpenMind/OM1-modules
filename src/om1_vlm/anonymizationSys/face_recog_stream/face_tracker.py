"""
Face tracker with BoTSORT + low-frequency, tier-aware face recognition.

Architecture:
  Every frame:  SCRFD detect → BoTSORT track (persistent IDs)
  Low freq:     For unidentified tracks → AdaFace embed → margin-aware
                recognition → multi-frame voting → assign identity
  Once matched: Identity sticks to track ID until track is lost.

Recognition tiers (see selfie_logic.recognize_from_sims):
  - "confident": top1 - top2 margin >= strict_margin → trust this match
  - "tentative": top1 - top2 margin >= loose_margin → likely but ask to verify
  - "uncertain": below loose_margin or top1 below sim_thr → "unknown" vote

Voting decision (see _finalize_identity):
  1. If any name has >= vote_threshold fraction of CONFIDENT votes → identified
     with tier="confident".
  2. Else if any name has >= vote_threshold fraction of CONFIDENT+TENTATIVE
     votes → identified with tier="tentative" (caller should verify with user).
  3. Otherwise → unknown.

Usage:
    tracker = FaceTracker(arc=arc_engine, sim_thr=0.45,
                          strict_margin=0.08, loose_margin=0.04)
    tracker.set_gallery(feats, labels)
    # Per frame:
    results = tracker.update(frame, dets, kpss)
    # results = [TrackResult(track_id=1, raw_name="wendy", tier="confident", ...), ...]
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import selfie_logic as sl

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrackResult:
    """Per-track output each frame.

    Fields
    ------
    track_id : int
        Persistent BoTSORT id; survives short occlusions.
    bbox : (x1, y1, x2, y2)
        Face bounding box in image coordinates.
    conf : float
        BoTSORT detection confidence for this frame.
    raw_name : str | None
        Plain identity label, e.g. ``"wendy"``, or ``None`` if not yet voted.
        Use this for downstream tracking (WhoTracker, fall match, etc.).
    name : str | None
        Display string with score, e.g. ``"wendy (0.72)"`` or ``"wendy? (0.62)"``.
        Use this for on-frame overlays/drawing.
    sim : float
        Average cosine similarity over the winning votes (0 if unknown).
    tier : str
        ``"confident"`` / ``"tentative"`` / ``"uncertain"``. Drives downstream
        UX (e.g. whether the robot greets by name or asks to verify).
    uuid : str | None
        Stable identity ID from the gallery (UUID4 hex), or ``None`` if the
        track is unidentified. Names can change via ``set_name``; UUIDs
        cannot. Downstream layers (HttpAPI, audit log) should reference
        identities by UUID rather than name.
    is_known : bool
        True only when ``tier == "confident"``. Kept for back-compat with
        callers (e.g. blur logic) that previously had only a boolean signal.
    """

    track_id: int
    bbox: Tuple[int, int, int, int]
    conf: float
    raw_name: Optional[str] = None
    name: Optional[str] = None
    sim: float = 0.0
    tier: str = sl.TIER_UNCERTAIN
    uuid: Optional[str] = None
    is_known: bool = False


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------


@dataclass
class _TrackIdentity:
    """Internal identity state accumulated for one BoTSORT track.

    Lifecycle: new → identifying → (identified | unknown)
    On re_identify_interval, "unknown" or low-confidence "identified" tracks
    are reset back to "new" so recognition can retry on a clearer frame.
    """

    track_id: int
    status: str = "new"  # new | identifying | identified | unknown

    # Decided identity (after _finalize_identity)
    name: Optional[str] = None
    uuid: Optional[str] = None  # gallery UUID of the winning name (None if unknown)
    sim: float = 0.0
    tier: str = sl.TIER_UNCERTAIN

    # Votes collected during the identifying phase.
    # Length-aligned lists: vote_names[i] is the predicted name for vote i,
    # vote_uuids[i] is the gallery UUID that name maps to (parallel; needed
    # because multiple UUIDs can share a name), vote_tiers[i] is the tier
    # of that prediction, vote_sims[i] is the top-1 sim. "unknown" entries
    # are kept (count toward total votes) but excluded from naming decisions
    # in _finalize_identity.
    vote_names: List[str] = field(default_factory=list)
    vote_uuids: List[Optional[str]] = field(default_factory=list)
    vote_tiers: List[str] = field(default_factory=list)
    vote_sims: List[float] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)

    # Timing & bookkeeping
    first_seen: float = 0.0
    last_recog_time: float = 0.0
    frames_seen: int = 0
    recog_attempts: int = 0
    # When >0, marks when this track first became (or stayed) unknown.
    # Used by the auto-enroll system to time how long a face has been
    # unknown.
    unknown_since: float = 0.0
    # ISO timestamp of the gallery's ``last_seen_at`` at the moment this
    # track first identified to a known UUID. Captured BEFORE any
    # touch_last_seen call from this session, so it reflects the
    # PREVIOUS sighting of this UUID — not "right now".
    # Stays sticky for the lifetime of the track session: even as the
    # gallery's last_seen_at is updated every frame, this field doesn't
    # change. That gives the LLM consistent "we've met before" / "newcomer"
    # signal throughout a single conversation.
    # None until the first confident identification.
    session_last_seen_iso: Optional[str] = None

    # The last UUID this track was CONFIDENTLY identified as. Persists across
    # re-identify resets (unlike `uuid`, which gets cleared when a track goes
    # unknown), so we can detect when one continuous track flips from one
    # confident identity to a different one — the trigger for auto-merging
    # fragmented identities (e.g. a frontal vs a profile enrollment).
    last_confident_uuid: Optional[str] = None


# ---------------------------------------------------------------------------
# FaceTracker
# ---------------------------------------------------------------------------


class FaceTracker:
    """BoTSORT face tracker with tier-aware recognition and multi-frame voting.

    Recognition flow per frame:

      1. BoTSORT assigns a track_id to each SCRFD detection (persistent IDs).
      2. For tracks in ``"new"`` or ``"identifying"`` state that haven't been
         re-checked within ``recog_interval`` seconds, queue them for AdaFace
         embedding extraction.
      3. Embed in a single batched pass (one TRT call for all queued tracks).
      4. For each track, run :func:`selfie_logic.recognize_from_sims` to get
         ``(name, tier, top1_sim, top2_sim)``. Push that into the track's
         vote history.
      5. Once a track has ``vote_frames`` votes (or hit ``max_recog_attempts``),
         call :meth:`_finalize_identity` to decide.

    Parameters
    ----------
    arc : TRTFaceRecognition
        AdaFace TRT engine for embedding extraction.
    recog_interval : float
        Seconds between recognition attempts per track (default 0.3).
    vote_frames : int
        Number of votes to collect before deciding identity (default 3).
    vote_threshold : float
        Fraction of votes a single name needs to win (default 0.5).
    max_recog_attempts : int
        Hard cap on recognition attempts per track (default 10).
    sim_thr : float
        Top-1 cosine threshold below which a vote is "uncertain" (default 0.4).
    strict_margin : float
        ``top1 - top2`` required for a vote to be "confident" (default 0.08).
    loose_margin : float
        ``top1 - top2`` required for a vote to be at least "tentative"
        (default 0.04). Must be < ``strict_margin``.
    track_buffer : int
        BoTSORT frames to keep lost tracks alive (default 60).
    arc_max_bs : int
        AdaFace TRT max batch size (default 8).
    det_conf : float
        SCRFD detection confidence; passed to BoTSORT for tier alignment.
    re_identify_interval : float
        Seconds before retrying recognition on "unknown" or marginal tracks.
        0 disables retry (default 3.0).
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
        strict_margin: float = 0.08,
        loose_margin: float = 0.04,
        track_buffer: int = 60,
        arc_max_bs: int = 8,
        det_conf: float = 0.5,
        re_identify_interval: float = 3.0,
        on_unknown_sample: Optional[
            "Callable[[int, np.ndarray, np.ndarray, Tuple[int,int,int,int], str, Optional[np.ndarray], float], Optional[str]]"
        ] = None,
        on_confident_sample: Optional[
            "Callable[[int, str, np.ndarray, np.ndarray, float], None]"
        ] = None,
    ):
        self.arc = arc
        self.recog_interval = float(recog_interval)
        self.vote_frames = int(vote_frames)
        self.vote_threshold = float(vote_threshold)
        self.max_recog_attempts = int(max_recog_attempts)
        self.sim_thr = float(sim_thr)
        self.strict_margin = float(strict_margin)
        self.loose_margin = float(loose_margin)
        self.arc_max_bs = int(arc_max_bs)
        self.re_identify_interval = float(re_identify_interval)
        # Callback fired once per recognition pass per track, regardless of
        # tier. Signature: (track_id, aligned_crop, embedding, bbox, tier).
        # Used by AutoEnroller to buffer unknown faces and drop committed
        # ones — keeping face_tracker unaware of storage concerns.
        self.on_unknown_sample = on_unknown_sample
        # Callback fired only on confident per-frame identifications.
        # Signature: (track_id, uuid, aligned_crop, embedding, sim).
        # Used by SampleRefreshManager for continuous-learning sample
        # enrichment. Guard A (sim threshold) is the caller's responsibility.
        self.on_confident_sample = on_confident_sample
        # Fired when one continuous track's CONFIDENT identity changes from one
        # UUID to a different one. Signature: (track_id, old_uuid, new_uuid).
        # Set as a plain attribute (like on_unknown_sample) by run.py, wired to
        # the auto-merge handler. None = feature off.
        self.on_identity_flip: Optional[Callable[[int, str, str], None]] = None
        # Vision-only "who is speaking" scorer (set by run.py). Optional.
        self.vvad = None

        # Gallery (set via set_gallery). All three arrays are parallel —
        # row i of _gal_feats has name _gal_labels[i] and UUID _gal_uuids[i].
        self._gal_feats: Optional[np.ndarray] = None
        self._gal_labels: List[str] = []
        self._gal_uuids: List[str] = []

        # BoTSORT tracker — align thresholds with SCRFD det_conf
        self._det_conf = float(det_conf)
        self._track_buffer = int(track_buffer)
        self._tracker = self._init_tracker(self._track_buffer, det_conf=self._det_conf)

        # Track identity state: track_id → _TrackIdentity
        self._identities: Dict[int, _TrackIdentity] = {}

        # Track IDs seen this frame (for cleanup)
        self._active_ids: set = set()

        # Current frame faces (for status queries)
        self._current_faces: list = []

    # ------------------------------------------------------------------
    # Setup / runtime config
    # ------------------------------------------------------------------

    @staticmethod
    def _init_tracker(track_buffer: int, det_conf: float = 0.5):
        """Initialize BoTSORT with thresholds aligned to detection confidence."""
        from boxmot import BotSort

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
        return BotSort(
            reid_weights=None,
            device="cuda",
            half=False,
            track_high_thresh=high_thresh,
            track_low_thresh=low_thresh,
            new_track_thresh=new_thresh,
            track_buffer=track_buffer,
            match_thresh=0.85,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=False,
        )

    def set_gallery(
        self,
        feats: Optional[np.ndarray],
        labels: List[str],
        uuids: Optional[List[str]] = None,
    ) -> None:
        """Update gallery centroids, names, and UUIDs (called after enrollment).

        ``uuids`` is optional for back-compat — if omitted, an empty string is
        used as a placeholder per row. Pass real UUIDs when using
        ``UUIDGallery.get_centroids()`` so downstream layers can reference
        identities by stable ID.

        Two side-effects on cached track identities (``self._identities``):

        1. **Stale invalidation** — if a track's cached UUID no longer exists
           in the new gallery (it got deleted), reset that track to "new"
           status so it re-enters recognition on the next frame. Without this,
           a track identified as "sean" before sean got deleted would keep
           displaying "sean" until the BoTSORT track itself died.

        2. **Name propagation** — if a track's cached UUID still exists but
           its name was changed (e.g. anon_xxx → sean via /set_name), update
           the cached ``ident.name`` in place. Without this, the displayed
           label would lag the rename by up to ``re_identify_interval`` seconds
           (the next recognition cycle), confusing demo viewers who expect
           rename to update the video immediately.
        """
        self._gal_feats = feats
        self._gal_labels = list(labels)
        if uuids is None:
            self._gal_uuids = [""] * len(labels)
        else:
            if len(uuids) != len(labels):
                raise ValueError(
                    f"uuids ({len(uuids)}) and labels ({len(labels)}) must be same length"
                )
            self._gal_uuids = list(uuids)

        # Build uuid → current_name map for O(1) lookups
        valid_uuids: Dict[str, str] = {}
        for u, n in zip(self._gal_uuids, self._gal_labels):
            if u:
                valid_uuids[u] = n

        stale_count = 0
        renamed_count = 0
        for tid, ident in self._identities.items():
            # A confidently-remembered UUID that no longer exists (e.g. it was
            # the loser of an auto-merge) must be forgotten, else it could
            # spuriously re-trigger a flip. Do this regardless of current uuid.
            if (
                ident.last_confident_uuid
                and ident.last_confident_uuid not in valid_uuids
            ):
                ident.last_confident_uuid = None
            if not ident.uuid:
                continue
            if ident.uuid not in valid_uuids:
                # (1) Stale: the UUID this track was identified to is gone.
                log.info(
                    "Track %d: identity %r (uuid=%s) removed from gallery — resetting",
                    tid,
                    ident.name,
                    ident.uuid,
                )
                ident.status = "new"
                ident.name = None
                ident.uuid = None
                ident.sim = 0.0
                ident.tier = sl.TIER_UNCERTAIN
                ident.vote_names.clear()
                ident.vote_uuids.clear()
                ident.vote_tiers.clear()
                ident.vote_sims.clear()
                ident.embeddings.clear()
                ident.recog_attempts = 0
                # Don't reset first_seen / frames_seen / unknown_since —
                # those describe the track itself, not the identity.
                stale_count += 1
            else:
                # (2) UUID still valid — check if name changed (rename via
                # /set_name, or claim via /selfie merging anon into named).
                new_name = valid_uuids[ident.uuid]
                # ident.name might be None for tracks that were anon (no name
                # finalized yet); coerce both sides to plain strings for
                # comparison. "" (anon) → "" (still anon) is a no-op;
                # "" → "sean" is the rename / claim case.
                cached_name = ident.name or ""
                if cached_name != new_name:
                    log.info(
                        "Track %d: uuid=%s name updated %r → %r (live propagation)",
                        tid,
                        ident.uuid,
                        cached_name,
                        new_name,
                    )
                    # Update cached name AND any votes for this UUID so the
                    # display rebuild picks up the change immediately. Note:
                    # we do NOT touch sim / tier / status — only the name.
                    ident.name = new_name if new_name else None
                    for i, vu in enumerate(ident.vote_uuids):
                        if vu == ident.uuid:
                            ident.vote_names[i] = new_name
                    renamed_count += 1
        if stale_count > 0 or renamed_count > 0:
            log.info(
                "set_gallery: reset=%d renamed=%d (%d valid UUIDs)",
                stale_count,
                renamed_count,
                len(valid_uuids),
            )

    def set_thresholds(
        self,
        *,
        sim_thr: Optional[float] = None,
        strict_margin: Optional[float] = None,
        loose_margin: Optional[float] = None,
    ) -> None:
        """Hot-update recognition thresholds. ``None`` means "leave as-is"."""
        if sim_thr is not None:
            self.sim_thr = float(sim_thr)
        if strict_margin is not None:
            self.strict_margin = float(strict_margin)
        if loose_margin is not None:
            self.loose_margin = float(loose_margin)

    def set_sim_thr(self, thr: float) -> None:
        """Back-compat shim — prefer ``set_thresholds(sim_thr=thr)``."""
        self.sim_thr = float(thr)

    # ------------------------------------------------------------------
    # Main per-frame entry point
    # ------------------------------------------------------------------

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
            5-point landmarks per detection (used for alignment).

        Returns
        -------
        list[TrackResult]
            One entry per active track this frame.
        """
        now = time.time()
        H, W = frame.shape[:2]

        # Run BoTSORT
        tracks = self._run_tracker(dets, frame)

        # Match tracks back to SCRFD detections (for landmarks)
        track_det_map = self._match_tracks_to_dets(tracks, dets, kpss)

        # Build results + collect tracks that still need recognition
        self._active_ids = set()
        results: List[TrackResult] = []
        need_recog: List[Tuple[int, np.ndarray, Optional[np.ndarray], float]] = []

        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            conf = float(track[5]) if len(track) > 5 else 0.0

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            self._active_ids.add(track_id)

            # Feed the speaking-detector buffer. Pad the (tight) SCRFD box so the
            # crop keeps the full mouth/chin context the VVAD model expects;
            # clipping the mouth squashes the speaking score. Recognition still
            # uses the original tight box below.
            if self.vvad is not None:
                bw, bh = x2 - x1, y2 - y1
                px = int(bw * 0.20)
                pyt = int(bh * 0.15)  # a little on top
                pyb = int(bh * 0.30)  # more on bottom (mouth/chin)
                cx1, cy1 = max(0, x1 - px), max(0, y1 - pyt)
                cx2, cy2 = min(W, x2 + px), min(H, y2 + pyb)
                self.vvad.push(
                    track_id,
                    frame[cy1:cy2, cx1:cx2],
                    kps=track_det_map.get(track_id, {}).get("kps"),
                )

            # Get or create identity state
            if track_id not in self._identities:
                self._identities[track_id] = _TrackIdentity(
                    track_id=track_id, first_seen=now
                )

            ident = self._identities[track_id]
            ident.frames_seen += 1

            # Track how long this face has been unknown (for auto-enroll later)
            if ident.status == "identified" and ident.tier == sl.TIER_CONFIDENT:
                ident.unknown_since = 0.0
            elif ident.unknown_since == 0.0:
                ident.unknown_since = now

            # Re-identify policy: if state is stale, reset votes so recognition
            # retries on what is hopefully a clearer frame.
            self._maybe_reset_for_reidentify(ident, now)

            # Decide whether this track needs another recognition attempt.
            # NOTE: we intentionally do NOT require a non-empty gallery here.
            # Even with zero identities, we still want to embed the face and
            # forward it to AutoEnroller (the empty-gallery branch below sets
            # tier=uncertain and calls _fire_unknown_callback). Excluding empty
            # gallery here would silently break auto-enrollment on a fresh
            # install.
            needs_recog = (
                ident.status in ("new", "identifying")
                and (now - ident.last_recog_time) >= self.recog_interval
                and ident.recog_attempts < self.max_recog_attempts
            )

            if needs_recog:
                det_kps = track_det_map.get(track_id, {}).get("kps")
                need_recog.append((track_id, np.array([x1, y1, x2, y2]), det_kps, conf))

            # Build display + raw name based on current status/tier
            raw_name, display_name, is_known, sim, tier, uuid = self._build_display(
                ident
            )

            results.append(
                TrackResult(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    conf=conf,
                    raw_name=raw_name,
                    name=display_name,
                    sim=sim,
                    tier=tier,
                    uuid=uuid,
                    is_known=is_known,
                )
            )

        # Batched recognition for tracks queued this frame
        if need_recog and self.arc is not None:
            log.debug("Running recognition for %d tracks", len(need_recog))
            self._run_recognition_batch(frame, need_recog, now)
        elif need_recog:
            log.warning("need_recog=%d but arc is None", len(need_recog))

        # Cleanup tracks that BoTSORT stopped reporting
        self._cleanup_stale()

        if self.vvad is not None:
            self.vvad.evict(self._active_ids)

        # Build the by-area sorted face list used by /who and unknown capture
        self._current_faces = self._build_current_faces(results)

        return results

    # ------------------------------------------------------------------
    # Re-identify / display
    # ------------------------------------------------------------------

    def _maybe_reset_for_reidentify(self, ident: _TrackIdentity, now: float) -> None:
        """Reset votes if it's been a while since last recognition attempt.

        Two trigger cases:
          - status == "unknown": maybe the face has gotten clearer since.
          - status == "identified" but tier == "tentative" or sim is marginal:
            try to upgrade to "confident" when conditions improve.

        Interaction with AutoEnroller
        -----------------------------
        While ``status == "unknown"`` (between finalize and re-identify
        firing), no recognition runs for this track, so AutoEnroller stops
        receiving observe() calls. This is acceptable: AutoEnroller's buffer
        from before finalization is still valid (samples don't age inside
        the buffer), and re-identify will resume observations within
        ``re_identify_interval`` seconds (default 3s) — well before
        AutoEnroller's ``stale_track_sec`` (default 5s) would GC the buffer.
        """
        if self.re_identify_interval <= 0:
            return
        if (now - ident.last_recog_time) < self.re_identify_interval:
            return

        should_reset = False
        if ident.status == "unknown":
            should_reset = True
            log.debug(
                "Track %d: re-identify (was unknown, %.1fs since last attempt)",
                ident.track_id,
                now - ident.last_recog_time,
            )
        elif ident.status == "identified" and ident.tier == sl.TIER_TENTATIVE:
            should_reset = True
            log.debug(
                "Track %d: re-verify '%s' (tentative, sim=%.3f)",
                ident.track_id,
                ident.name,
                ident.sim,
            )
        elif ident.status == "identified" and ident.sim < self.sim_thr + 0.1:
            # Confident but marginal sim — re-verify when face gets clearer.
            should_reset = True
            log.debug(
                "Track %d: re-verify '%s' (marginal sim=%.3f)",
                ident.track_id,
                ident.name,
                ident.sim,
            )

        if should_reset:
            ident.status = "new"
            ident.vote_names.clear()
            ident.vote_uuids.clear()
            ident.vote_tiers.clear()
            ident.vote_sims.clear()
            ident.embeddings.clear()
            ident.recog_attempts = 0

    @staticmethod
    def _anon_label(uuid: Optional[str]) -> str:
        """Short, stable, human-readable label for an anonymous (un-named) UUID.

        Example: '73d0a41a63aa4594...' → 'anon_73d0a4'

        The 6 hex chars give ~16.7M possible labels — collision-free for any
        realistic gallery size. The label is deterministic: same UUID always
        maps to the same label, even across restarts. The LLM can recognize
        this as "we've met but I don't know your name yet" and prompt for one.
        """
        if not uuid:
            return "unknown"
        return f"anon_{uuid[:6]}"

    def _build_display(
        self, ident: _TrackIdentity
    ) -> Tuple[Optional[str], Optional[str], bool, float, str, Optional[str]]:
        """Return ``(raw_name, display_name, is_known, sim, tier, uuid)``.

        ``raw_name`` is the plain label for downstream consumers (no score).
        ``display_name`` is the formatted string for on-frame drawing.
        ``is_known`` is True only when tier is confident.
        ``uuid`` is the gallery UUID for stable identity references (None
        when no identification has been made yet).
        """
        # Confident identification with a name
        if (
            ident.status == "identified"
            and ident.tier == sl.TIER_CONFIDENT
            and ident.name
        ):
            return (
                ident.name,
                f"{ident.name} ({ident.sim:.2f})",
                True,
                ident.sim,
                sl.TIER_CONFIDENT,
                ident.uuid,
            )

        # Confident identification but the UUID has no name yet (auto-enrolled,
        # awaiting /set_name). Surface a stable short label so the LLM and
        # operator can distinguish "we've met before" from "complete stranger".
        if (
            ident.status == "identified"
            and ident.tier == sl.TIER_CONFIDENT
            and not ident.name
            and ident.uuid
        ):
            label = self._anon_label(ident.uuid)
            return (
                label,
                f"{label} ({ident.sim:.2f})",
                True,
                ident.sim,
                sl.TIER_CONFIDENT,
                ident.uuid,
            )

        # Tentative identification with a name — show with "?" suffix
        if (
            ident.status == "identified"
            and ident.tier == sl.TIER_TENTATIVE
            and ident.name
        ):
            return (
                ident.name,
                f"{ident.name}? ({ident.sim:.2f})",
                False,
                ident.sim,
                sl.TIER_TENTATIVE,
                ident.uuid,
            )

        # Tentative identification for an anonymous UUID (matched something
        # in gallery but with shaky confidence, and that UUID has no name).
        if (
            ident.status == "identified"
            and ident.tier == sl.TIER_TENTATIVE
            and not ident.name
            and ident.uuid
        ):
            label = self._anon_label(ident.uuid)
            return (
                label,
                f"{label}? ({ident.sim:.2f})",
                False,
                ident.sim,
                sl.TIER_TENTATIVE,
                ident.uuid,
            )

        # Voting still in progress — surface a tentative best guess if we have one
        if ident.status == "identifying" and ident.vote_names:
            real = [
                (n, u, s)
                for n, u, t, s in zip(
                    ident.vote_names,
                    ident.vote_uuids,
                    ident.vote_tiers,
                    ident.vote_sims,
                )
                if n != "unknown"
            ]
            if real:
                top_name = max(
                    {n for n, _, _ in real},
                    key=lambda x: sum(1 for n, _, _ in real if n == x),
                )
                top_sim = max(s for n, _, s in real if n == top_name)
                # Best-guess UUID = most-frequent UUID among votes for top_name
                cand_uuids = [u for n, u, _ in real if n == top_name and u is not None]
                top_uuid = (
                    Counter(cand_uuids).most_common(1)[0][0] if cand_uuids else None
                )
                # Anonymous UUID (no name yet) → use stable short label
                display_label = top_name if top_name else self._anon_label(top_uuid)
                return (
                    display_label,
                    f"{display_label}? ({top_sim:.2f})",
                    False,
                    top_sim,
                    sl.TIER_TENTATIVE,
                    top_uuid,
                )

        # New or unknown — no identity available
        return (None, "unknown", False, 0.0, sl.TIER_UNCERTAIN, None)

    def _build_current_faces(self, results: List[TrackResult]) -> list:
        """Sort the per-track results into a by-area face list.

        Used by /who and the auto-enroll candidate selector. "Largest face"
        is a cheap proxy for "closest to camera = most likely the addressee".
        """
        faces = []
        for r in results:
            x1, y1, x2, y2 = r.bbox
            area = (x2 - x1) * (y2 - y1)
            # Look up the snapshotted session-start last_seen for this
            # track. Falls back to None for tracks not yet confidently
            # identified (e.g. "unknown" status, or first few frames).
            ident = self._identities.get(r.track_id)
            session_last_seen = ident.session_last_seen_iso if ident else None
            faces.append(
                {
                    "name": r.raw_name if r.raw_name else "unknown",
                    "uuid": r.uuid,
                    "tier": r.tier,
                    "bbox": r.bbox,
                    "area": area,
                    "track_id": r.track_id,
                    "session_last_seen_iso": session_last_seen,
                }
            )
        return sorted(faces, key=lambda f: f["area"], reverse=True)

    # ------------------------------------------------------------------
    # BoTSORT plumbing
    # ------------------------------------------------------------------

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

        Returns ``{track_id: {"det_idx": int, "kps": ndarray(5,2) or None}}``.
        """
        result: Dict[int, dict] = {}
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
                entry: dict = {"det_idx": best_d}
                entry["kps"] = (
                    kpss[best_d] if kpss is not None and best_d < len(kpss) else None
                )
                result[track_id] = entry

        return result

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def _run_recognition_batch(
        self,
        frame: np.ndarray,
        need_recog: List[Tuple[int, np.ndarray, Optional[np.ndarray], float]],
        now: float,
    ) -> None:
        """Embed unidentified tracks in one TRT call and update their votes."""
        from .adaface import warp_face_by_5p

        crops: List[np.ndarray] = []
        crop_track_ids: List[int] = []
        crop_bboxes: List[np.ndarray] = []
        crop_kpss: List[Optional[np.ndarray]] = []
        crop_confs: List[float] = []

        for track_id, bbox, kps, conf in need_recog:
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
                crop_kpss.append(kps)
                crop_confs.append(conf)
            except Exception:
                continue

        if not crops:
            return

        # Batched AdaFace inference
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

        if self._gal_feats is None or self._gal_feats.size == 0:
            # Even with empty gallery, we want to give the auto-enroller a
            # chance to observe (everyone is "uncertain" in this state, but
            # we still need the embedding-buffering path to fire).
            for idx, track_id in enumerate(crop_track_ids):
                ident = self._identities.get(track_id)
                if ident is None:
                    continue
                ident.last_recog_time = now
                ident.recog_attempts += 1
                ident.status = "identifying"
                ident.embeddings.append(feats[idx])
                ident.vote_names.append("unknown")
                ident.vote_uuids.append(None)
                ident.vote_tiers.append(sl.TIER_UNCERTAIN)
                ident.vote_sims.append(0.0)
                self._fire_unknown_callback(
                    track_id,
                    crops[idx],
                    feats[idx],
                    tuple(crop_bboxes[idx]),
                    sl.TIER_UNCERTAIN,
                )
                if (
                    len(ident.vote_names) >= self.vote_frames
                    or ident.recog_attempts >= self.max_recog_attempts
                ):
                    self._finalize_identity(ident)
            return

        # Single matmul vs. gallery centroids: (B, dim) @ (dim, N_id) = (B, N_id)
        S = feats @ self._gal_feats.T

        for idx, track_id in enumerate(crop_track_ids):
            ident = self._identities.get(track_id)
            if ident is None:
                continue

            ident.last_recog_time = now
            ident.recog_attempts += 1
            ident.status = "identifying"

            name, tier, s1, s2 = sl.recognize_from_sims(
                S[idx],
                self._gal_labels,
                min_sim=self.sim_thr,
                strict_margin=self.strict_margin,
                loose_margin=self.loose_margin,
            )

            # Map name → UUID using the top-1 row of S. recognize_from_sims
            # returns the name of the best-scoring row; for tier "uncertain"
            # the name is "unknown" so there's no UUID to attach.
            if name != "unknown" and self._gal_uuids:
                top_idx = int(np.argmax(S[idx]))
                vote_uuid: Optional[str] = self._gal_uuids[top_idx]
            else:
                vote_uuid = None

            ident.embeddings.append(feats[idx])
            ident.vote_names.append(name)
            ident.vote_uuids.append(vote_uuid)
            ident.vote_tiers.append(tier)
            ident.vote_sims.append(s1)

            log.debug(
                "Track %d bbox=%s: top1=%s/%.3f top2=%.3f margin=%.3f → tier=%s uuid=%s",
                track_id,
                tuple(crop_bboxes[idx]),
                name,
                s1,
                s2,
                s1 - s2,
                tier,
                vote_uuid[:8] + "..." if vote_uuid else "-",
            )

            # Notify auto-enroller (it decides what to do based on tier)
            self._fire_unknown_callback(
                track_id,
                crops[idx],
                feats[idx],
                tuple(crop_bboxes[idx]),
                tier,
                crop_kpss[idx],
                crop_confs[idx],
            )

            # Notify refresh manager on confident identifications. Refresh
            # manager applies its own guards (rate-limit, diversity) before
            # actually folding the sample into the gallery. We pass the
            # specific UUID this frame matched (vote_uuid) — could differ
            # from the track's finalized UUID if the track's voting hasn't
            # converged yet, but for confident per-frame matches the
            # difference is rarely material.
            if (
                tier == sl.TIER_CONFIDENT
                and vote_uuid
                and self.on_confident_sample is not None
            ):
                try:
                    self.on_confident_sample(
                        track_id,
                        vote_uuid,
                        crops[idx],
                        feats[idx],
                        s1,
                    )
                except Exception as e:
                    log.warning("on_confident_sample callback raised: %s", e)

            # Decide as soon as we have enough votes or hit the cap
            if (
                len(ident.vote_names) >= self.vote_frames
                or ident.recog_attempts >= self.max_recog_attempts
            ):
                self._finalize_identity(ident)

    def _fire_unknown_callback(
        self,
        track_id: int,
        crop: np.ndarray,
        embedding: np.ndarray,
        bbox: Tuple[int, int, int, int],
        tier: str,
        kps: Optional[np.ndarray] = None,
        conf: float = 1.0,
    ) -> None:
        """Forward a recognition observation to the auto-enroll callback.

        The callback receives every tier (not just uncertain) — AutoEnroller
        uses non-uncertain observations to clear its per-track buffer when a
        formerly-unknown face becomes identified.
        """
        if self.on_unknown_sample is None:
            return
        try:
            self.on_unknown_sample(track_id, crop, embedding, bbox, tier, kps, conf)
        except Exception as e:
            log.warning("on_unknown_sample callback raised: %s", e)

    def _finalize_identity(self, ident: _TrackIdentity) -> None:
        """Decide a track's identity from collected (name, tier) votes.

        Strategy:
          - Tally CONFIDENT votes by name.
          - Tally TENTATIVE votes by name (these EXCLUDE confident; they're a
            separate weaker bucket).
          - If any name's confident-fraction >= vote_threshold → identified
            confident.
          - Else if any name's (confident + tentative)-fraction >= threshold
            → identified tentative.
          - Else → unknown.

        This means a track with mixed evidence (some confident, some tentative
        for the same name) is preferentially called confident — confident
        votes are strictly stronger than tentative ones.
        """
        now = time.time()
        total_votes = len(ident.vote_names)

        if total_votes == 0:
            self._set_unknown(ident, now, reason="no_votes")
            return

        confident_counts: Counter = Counter()
        loose_counts: Counter = Counter()  # confident + tentative for each name
        for name, tier in zip(ident.vote_names, ident.vote_tiers):
            if name == "unknown":
                continue
            if tier == sl.TIER_CONFIDENT:
                confident_counts[name] += 1
                loose_counts[name] += 1
            elif tier == sl.TIER_TENTATIVE:
                loose_counts[name] += 1
            # tier == "uncertain" never carries a real name (name == "unknown")

        # Try confident first
        if confident_counts:
            best_name, best_count = confident_counts.most_common(1)[0]
            if best_count / total_votes >= self.vote_threshold:
                self._set_identified(
                    ident,
                    best_name,
                    sl.TIER_CONFIDENT,
                    now,
                    votes_for_name=best_count,
                    total_votes=total_votes,
                )
                return

        # Fall back to tentative (still requires the same vote_threshold of
        # loose evidence — we just don't demand confident-tier evidence)
        if loose_counts:
            best_name, best_count = loose_counts.most_common(1)[0]
            if best_count / total_votes >= self.vote_threshold:
                self._set_identified(
                    ident,
                    best_name,
                    sl.TIER_TENTATIVE,
                    now,
                    votes_for_name=best_count,
                    total_votes=total_votes,
                )
                return

        self._set_unknown(ident, now, reason="no_majority")

    def _set_identified(
        self,
        ident: _TrackIdentity,
        name: str,
        tier: str,
        now: float,
        *,
        votes_for_name: int,
        total_votes: int,
    ) -> None:
        """Apply an identified decision to a track and log it."""
        winning_sims = [
            s for n, s in zip(ident.vote_names, ident.vote_sims) if n == name
        ]
        avg_sim = sum(winning_sims) / len(winning_sims) if winning_sims else 0.0

        # Pick the UUID associated with the winning name. Most-frequent UUID
        # among the votes that voted for `name` — robust to the rare case
        # where the same name maps to two distinct UUIDs over the vote window
        # (e.g. two enrollments of the same person; both centroids matched at
        # different votes).
        winning_uuids = [
            u
            for n, u in zip(ident.vote_names, ident.vote_uuids)
            if n == name and u is not None
        ]
        winning_uuid: Optional[str] = None
        if winning_uuids:
            winning_uuid = Counter(winning_uuids).most_common(1)[0][0]

        ident.status = "identified"
        ident.tier = tier
        ident.name = name
        ident.uuid = winning_uuid
        ident.sim = avg_sim
        ident.unknown_since = (
            0.0 if tier == sl.TIER_CONFIDENT else (ident.unknown_since or now)
        )

        # --- identity-flip detection (auto-merge trigger) ---
        # If this same continuous track was previously CONFIDENT on a
        # different UUID, the two UUIDs are very likely the same person whose
        # poses got separate enrollments (frontal vs profile). Fire the flip
        # callback so the outer layer can merge them. Only confident->confident
        # transitions count; we reuse the tier already decided above.
        if tier == sl.TIER_CONFIDENT and winning_uuid:
            prev = ident.last_confident_uuid
            if prev and prev != winning_uuid and self.on_identity_flip is not None:
                try:
                    self.on_identity_flip(ident.track_id, prev, winning_uuid)
                except Exception as e:
                    log.warning("on_identity_flip callback raised: %s", e)
            ident.last_confident_uuid = winning_uuid

        log.info(
            "Track %d: %s '%s' uuid=%s (votes=%d/%d, sim=%.3f)",
            ident.track_id,
            tier,
            name,
            (winning_uuid[:8] + "...") if winning_uuid else "-",
            votes_for_name,
            total_votes,
            avg_sim,
        )

    def _set_unknown(self, ident: _TrackIdentity, now: float, *, reason: str) -> None:
        """Mark a track as unknown (no majority found)."""
        ident.status = "unknown"
        ident.tier = sl.TIER_UNCERTAIN
        ident.name = None
        ident.uuid = None
        if ident.unknown_since == 0:
            ident.unknown_since = now
        log.debug("Track %d: unknown (%s)", ident.track_id, reason)

    # ------------------------------------------------------------------
    # Cleanup / introspection
    # ------------------------------------------------------------------

    def _cleanup_stale(self) -> None:
        """Drop identity state for tracks BoTSORT no longer reports."""
        stale = [tid for tid in self._identities if tid not in self._active_ids]
        for tid in stale:
            del self._identities[tid]

    def get_track_identities(self) -> Dict[int, dict]:
        """Snapshot of current identity state per active track (for /who etc)."""
        return {
            tid: {
                "name": ident.name,
                "uuid": ident.uuid,
                "status": ident.status,
                "tier": ident.tier,
                "sim": ident.sim,
                "frames_seen": ident.frames_seen,
                "recog_attempts": ident.recog_attempts,
                "votes": len(ident.vote_names),
                "session_last_seen_iso": ident.session_last_seen_iso,
            }
            for tid, ident in self._identities.items()
            if tid in self._active_ids
        }

    def maybe_set_session_last_seen(self, track_id: int, iso: Optional[str]) -> bool:
        """If this track has no ``session_last_seen_iso`` yet, set it.

        Called by the on_confident_sample wiring in run.py the first time
        a track is confidently identified, BEFORE touching the gallery's
        last_seen_at. After that, subsequent confident matches in the same
        track session preserve the snapshotted value (sticky).

        Returns True if set this call, False if already set (or unknown
        track).
        """
        ident = self._identities.get(track_id)
        if ident is None:
            return False
        if ident.session_last_seen_iso is not None:
            return False
        ident.session_last_seen_iso = iso
        return True

    def get_session_last_seen(self, track_id: int) -> Optional[str]:
        """Read the sticky session-start last_seen for a track, or None."""
        ident = self._identities.get(track_id)
        if ident is None:
            return None
        return ident.session_last_seen_iso

    def get_faces(self) -> list:
        """All faces in the current frame, sorted by bbox area descending.

        Each entry: ``{name, tier, bbox, area, track_id}``.
        ``name`` is ``"unknown"`` if the track has no identification (or is
        only tentative — callers check ``tier`` separately).
        """
        return self._current_faces

    def get_unknowns(self) -> list:
        """Faces currently unidentified, with how long they've been unknown."""
        now = time.time()
        unknowns = []
        for face in self._current_faces:
            if face["name"] != "unknown":
                continue
            tid = face["track_id"]
            ident = self._identities.get(tid)
            if ident is None or ident.unknown_since <= 0:
                continue
            unknowns.append(
                {
                    "track_id": tid,
                    "bbox": list(face["bbox"]),
                    "area": face["area"],
                    "unknown_duration": round(now - ident.unknown_since, 1),
                }
            )
        return unknowns

    def reset(self) -> None:
        """Reset all tracking state (e.g. after gallery wipe between venues)."""
        self._identities.clear()
        self._active_ids.clear()
        self._current_faces.clear()
        if self._tracker is not None:
            self._tracker = self._init_tracker(
                self._track_buffer, det_conf=self._det_conf
            )
