"""
Auto-enrollment: silently build a UUID for a face that keeps showing up unknown.

How it works
------------
1. Per recognition pass, face_tracker hands us (track_id, aligned_crop,
   embedding, sim_top1) for tracks that came back ``"uncertain"``. We do
   no GPU work here — embedding is already computed upstream.

2. For each track, we maintain a small buffer of recent samples (default
   max_buffer=5). Samples must pass a per-frame quality bar (frontality,
   sharpness, face size) — bad frames are dropped immediately.

3. When a track's buffer hits ``n_samples_required`` AND the pairwise
   cosine similarity between buffered samples is tight enough
   (``min_pairwise_sim``, default 0.6) — meaning they all look like the
   same person, not a flicker between two visitors — we commit a new
   UUID with name="" (anonymous). The LLM can name it later via
   ``/set_name``.

4. Tracks that disappear (no more frames) get their buffer dropped after
   ``stale_track_sec``. Tracks that get identified by face_tracker also
   get dropped (no need to enroll them).

Why this exists
---------------
Right now, enrollment requires the LLM to ask "what's your name?" and
call /selfie. For drop-in visitors who never speak, we'd never enroll
them. With auto-enroll, the gallery quietly grows; the LLM only has to
*name* the identities, not create them. Naming is then a snappy O(1)
metadata write instead of a multi-frame collection loop.

Threading model
---------------
Same as gallery.py — all methods should be called from the main thread
(via the recognition callback inside face_tracker.update). No internal
locking. Maintains its own per-track buffer; doesn't touch shared state
except by calling gallery methods.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from . import selfie_logic as sl

log = logging.getLogger(__name__)


# Tier name from selfie_logic — duplicated here as a string constant so this
# module doesn't import face_tracker / selfie_logic (cleaner dependency graph).
_TIER_UNCERTAIN = "uncertain"


# ---------------------------------------------------------------------------
# Per-track buffer
# ---------------------------------------------------------------------------


@dataclass
class _TrackBuffer:
    """Rolling buffer of recent samples for one unknown track.

    We keep aligned crops AND their embeddings — embeddings for the
    tightness/novelty checks, crops for the eventual commit to the gallery.

    Timestamps
    ----------
    first_seen : first time we created a buffer for this track. Drives the
        ``min_unknown_sec`` gate.
    last_seen : most recent ``observe()`` call for this track (whether or
        not the sample was buffered). Drives ``gc_stale`` — if no observe
        has fired in ``stale_track_sec``, the track is gone and we drop
        the buffer.

    Note: there's no ``last_buffered`` because sample-spacing is controlled
    by embedding-based novelty (not time). See AutoEnroller.observe() and
    the ``novelty_thr`` parameter for the rationale.
    """

    track_id: int
    crops: Deque[np.ndarray] = field(default_factory=deque)
    embeddings: Deque[np.ndarray] = field(default_factory=deque)
    first_seen: float = 0.0
    last_seen: float = 0.0
    # True after we've successfully enrolled this track — prevents double
    # enrollment if more samples arrive before face_tracker picks up the new
    # gallery centroid and starts identifying this track as "confident".
    committed_uuid: Optional[str] = None


# ---------------------------------------------------------------------------
# AutoEnroller
# ---------------------------------------------------------------------------


class AutoEnroller:
    """Watches unknown tracks and silently enrolls them when confident.

    Parameters
    ----------
    gallery : UUIDGallery
        Where new identities are created. Must be writable from the calling
        thread (i.e. the main thread when this is wired via run_job_sync).
    n_samples_required : int, default 3
        How many quality samples must accumulate before we even consider
        enrolling. Lower → faster enrollment, more risk of bad identities.
    max_buffer : int, default 5
        Maximum samples kept per track. When full, oldest is dropped. Should
        be ≥ n_samples_required.
    min_pairwise_sim : float, default 0.55
        Required pairwise cosine similarity (worst pair) across the buffer
        before enrollment. Filters out tracks that "flicker" between two
        people (BoTSORT ID switch on similar-looking visitors). Conservative
        starting value; tune up if you see misenrollments.
    min_unknown_sec : float, default 1.5
        Track must have been visible/unknown for at least this long before
        enrollment fires. Keeps us from enrolling people who just walked
        through the frame.
    min_face_pixels : int, default 80
        Minimum bbox short side. Below this, samples are skipped — they're
        too small to be reliable.
    capture_interval_sec : float, default 0.3
        Minimum time between buffered samples for the same track. Prevents
        the buffer from filling with near-duplicates from consecutive frames.
    stale_track_sec : float, default 5.0
        Drop a track's buffer if no observe() in this long. Saves memory on
        tracks that walked off.
    merge_thr : float, default 0.50
        Before creating a NEW UUID, check whether the buffer's mean embedding
        is close enough to any existing UUID's centroid. If max cosine sim
        across the gallery >= ``merge_thr``, append the samples to that
        existing UUID instead of creating a new one.

        Why this exists: ``SIM_THR`` is the per-frame recognition floor —
        below it, a face is labeled "unknown" even if it's a known person
        in bad lighting (top1=0.52 but margin=0.20 → "uncertain" because
        0.52 < SIM_THR=0.55, but actually it's clearly wendy). Without this
        merge step, AutoEnroller would create a duplicate UUID for wendy
        every time she walks into a different lighting condition.

        The buffer mean is more reliable than any single frame's embedding
        (averaging suppresses noise), so this check uses the WHOLE buffer's
        centroid against the gallery — typically higher than the per-frame
        top1, often crossing into the "definitely this known person" zone.

        Set lower than SIM_THR. Setting to 0 disables the merge-on-commit
        path (always creates new UUIDs, original behavior).
    source_tag : str, default "auto_enroll"
        ``source`` value to write into metadata.json for traceability.
    on_committed : callable, optional
        Optional callback ``on_committed(uuid, track_id, sample_count)``
        fired immediately after a successful enrollment. Useful for the
        main loop to refresh face_tracker's gallery snapshot.

    Why no per-sample novelty/consistency check
    -------------------------------------------
    Earlier iterations had two embedding-based gates here:
      - novelty (reject if sim_to_mean > 0.97): tried to avoid buffering
        near-duplicate frames
      - consistency (reject if sim_to_mean < 0.50): tried to catch
        BoTSORT track-id swaps mid-buffer

    Problem with novelty: a person standing naturally still produces
    sim ≈ 0.95-0.99 between consecutive frames. Setting novelty_thr
    permissively (0.97) starves the buffer in the still case; setting
    it loosely (0.99) makes the gate near-meaningless. The "window"
    (sim < novelty_thr AND sim > consistency_thr) often excludes
    perfectly valid same-person frames.

    Problem with consistency vs tightness: the per-sample consistency
    check is mostly redundant with the at-commit tightness check
    (``min_pairwise_sim``). Both reject "buffer contains samples of
    multiple people".

    Resolution: skip both per-sample checks. Accept all quality samples
    into the buffer; rely on tightness at commit time. Within-pose
    averaging from 3+ similar samples still yields a more robust
    centroid than any single sample. Cross-pose diversity comes from
    repeat encounters via the ``merge_thr`` path, not from one session.
    """

    def __init__(
        self,
        gallery,
        *,
        n_samples_required: int = 3,
        max_buffer: int = 5,
        min_pairwise_sim: float = 0.55,
        min_unknown_sec: float = 1.0,
        min_face_pixels: int = 30,
        stale_track_sec: float = 5.0,
        merge_thr: float = 0.50,
        source_tag: str = "auto_enroll",
        on_committed: Optional[Callable[[str, int, int], None]] = None,
        min_conf: float = 0.0,
        min_frontality: float = 0.0,
    ):
        self.gallery = gallery
        self.n_required = int(n_samples_required)
        self.max_buffer = max(int(max_buffer), self.n_required)
        self.min_pairwise_sim = float(min_pairwise_sim)
        self.min_unknown_sec = float(min_unknown_sec)
        self.min_face_pixels = int(min_face_pixels)
        self.stale_track_sec = float(stale_track_sec)
        self.merge_thr = float(merge_thr)
        self.source_tag = str(source_tag)
        self.on_committed = on_committed
        # Dedicated auto-enroll quality gates, independent of the /selfie
        # thresholds. 0.0 = gate disabled.
        self.min_conf = float(min_conf)
        self.min_frontality = float(min_frontality)

        # track_id → buffer
        self._buffers: Dict[int, _TrackBuffer] = {}

        # Stats (for /who debug)
        self.enrolled_count = 0
        self.merged_count = 0  # NEW: tracks merge-on-commit decisions
        self.rejected_count = 0

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def observe(
        self,
        track_id: int,
        aligned_crop: np.ndarray,
        embedding: np.ndarray,
        bbox: Tuple[int, int, int, int],
        tier: str,
        kps: Optional[np.ndarray] = None,
        conf: float = 1.0,
    ) -> Optional[str]:
        """Feed one frame's recognition result for one track.

        Called from face_tracker._run_recognition_batch for EVERY track that
        ran recognition this batch (not just unknown ones — passing
        identified tracks lets us drop their buffers if they were previously
        unknown).

        Parameters
        ----------
        track_id : int
            BoTSORT track id. Stable across frames for the same person.
        aligned_crop : ndarray (112, 112, 3) BGR
            Same warped crop that was fed to AdaFace. Reused for storage.
        embedding : ndarray (512,) float32, L2-normalized
            AdaFace output for ``aligned_crop``.
        bbox : (x1, y1, x2, y2)
            Face bbox in image coordinates. Used for the size gate.
        tier : str
            Recognition tier from selfie_logic.recognize_from_sims.
            We only buffer samples when tier is "uncertain"; tracks that
            became identified get their buffer cleared.

        Returns
        -------
        Optional[str]
            If this observation triggered an enrollment, returns the new
            UUID. Otherwise None. Most calls return None.
        """
        now = time.time()

        # Track became identified or tentative → drop any buffer, no enroll
        if tier != _TIER_UNCERTAIN:
            self._buffers.pop(track_id, None)
            return None

        # Quality / size gate happens FIRST — don't even create a buffer
        # entry for samples that fail the bar. Avoids leaking empty entries.
        x1, y1, x2, y2 = bbox
        if min(x2 - x1, y2 - y1) < self.min_face_pixels:
            return None

        # Dedicated auto-enroll quality gate (independent of /selfie): require
        # a confident, frontal detection before buffering, so only clean
        # frontal faces enter the gallery. Does NOT affect what the robot can
        # SEE/track — the global DETECTION_CONFIDENCE stays low; this only
        # gates enrollment. 0.0 thresholds = gate off.
        if self.min_conf > 0.0 and conf < self.min_conf:
            return None
        if self.min_frontality > 0.0 and kps is not None:
            if sl.frontality(kps) < self.min_frontality:
                return None

        # Get or create buffer
        buf = self._buffers.get(track_id)
        if buf is None:
            buf = _TrackBuffer(
                track_id=track_id,
                first_seen=now,
                last_seen=now,
            )
            self._buffers[track_id] = buf

        # Always mark the track as recently seen — keeps gc_stale honest
        # even when the sample below gets rate-limited away.
        buf.last_seen = now

        # Already enrolled this track — wait for face_tracker to start
        # identifying it as "confident". Don't enroll again.
        if buf.committed_uuid is not None:
            return None

        # No per-sample novelty/consistency check. We accept every quality
        # sample and rely on min_pairwise_sim (the tightness check at commit
        # time) to catch BoTSORT track-id swaps. See the class docstring
        # for the rationale.

        # Buffer this sample (drop oldest if at capacity)
        buf.crops.append(aligned_crop.copy())
        buf.embeddings.append(embedding.copy())
        if len(buf.crops) > self.max_buffer:
            buf.crops.popleft()
            buf.embeddings.popleft()

        # Eligibility checks
        if len(buf.embeddings) < self.n_required:
            return None
        if (now - buf.first_seen) < self.min_unknown_sec:
            return None

        # Pairwise tightness check
        worst = self._min_pairwise_sim(list(buf.embeddings))
        if worst < self.min_pairwise_sim:
            log.debug(
                "auto-enroll track=%d skipped: pairwise=%.3f < %.3f",
                track_id,
                worst,
                self.min_pairwise_sim,
            )
            self.rejected_count += 1
            # Don't drop the buffer — maybe later samples improve cohesion.
            # But cap how much we wait by trimming oldest if we exceeded buffer.
            return None

        # All gates passed → commit
        uuid = self._commit(buf)
        return uuid

    def force_commit(self, track_id: int, name: str = "") -> Optional[str]:
        """Commit a track's buffer immediately, even if gates aren't satisfied.

        Used by /selfie when the LLM has just learned the person's name but
        auto-enroll hasn't fired yet for this track. The selfie collection
        loop itself produces fresh samples; this method drains whatever
        auto-enroll has already buffered.

        Returns the UUID, or None if the track has no buffer or only 0
        samples (nothing to commit).
        """
        buf = self._buffers.get(track_id)
        if buf is None or not buf.crops:
            return None
        if buf.committed_uuid is not None:
            # Already committed — caller can use that UUID
            return buf.committed_uuid
        uuid = self._commit(buf, name=name, forced=True)
        return uuid

    def drop_track(self, track_id: int) -> None:
        """Explicitly drop a track's buffer (e.g. when track was identified
        as a known person by a different path).
        """
        self._buffers.pop(track_id, None)

    def gc_stale(self) -> int:
        """Remove buffers for tracks that haven't been observed recently.

        Returns the number of buffers dropped. Called once per frame (cheap)
        from the main loop, OR every few seconds from a slower timer.
        """
        now = time.time()
        cutoff = now - self.stale_track_sec
        stale = [tid for tid, buf in self._buffers.items() if buf.last_seen < cutoff]
        for tid in stale:
            self._buffers.pop(tid, None)
        return len(stale)

    def try_commit_pending(self) -> List[str]:
        """Scan all active buffers and commit any that pass all gates.

        WHY THIS EXISTS
        ---------------
        AutoEnroller's commit logic in ``observe()`` is event-driven — it
        only runs when face_tracker calls observe() with a new sample.
        But after face_tracker finalizes a track as ``status="unknown"``,
        it stops calling observe() until re-identify fires (≥1s later).

        Without this method, a buffer that accumulated enough samples
        BEFORE the track went "unknown" would just sit there waiting:
          - sample_count gate: PASSED (≥ n_required)
          - tightness gate:    PASSED (samples are consistent)
          - time gate:         MAY pass during the dead zone

        but no observe() is being called to trigger the gate check. So
        the UUID creation gets delayed by up to re_identify_interval
        seconds, even though every condition has been met.

        This method, called periodically from the main loop, polls all
        buffers and commits any that pass all gates RIGHT NOW. Solves
        the "system knows it's a stranger but hasn't enrolled them yet"
        latency.

        Idempotent — already-committed buffers are skipped. Returns the
        list of UUIDs created during this call (usually empty, sometimes
        1+ if multiple unknown tracks ripened at once).

        Cheap: typically iterates over a handful of buffers, each gate
        check is O(1) or O(n²) on small embedding lists (n ≤ max_buffer).
        Safe to call every frame, though every ~1s is enough.
        """
        if not self._buffers:
            return []
        now = time.time()
        committed: List[str] = []
        for track_id, buf in list(self._buffers.items()):
            # Skip already-committed buffers
            if buf.committed_uuid is not None:
                continue
            # Sample count gate
            if len(buf.embeddings) < self.n_required:
                continue
            # Time gate
            if (now - buf.first_seen) < self.min_unknown_sec:
                continue
            # Tightness gate
            worst = self._min_pairwise_sim(list(buf.embeddings))
            if worst < self.min_pairwise_sim:
                # Tight enough samples haven't accumulated; leave for now.
                # If the track is gone, gc_stale will drop the buffer later.
                continue
            # All gates pass → commit
            uuid = self._commit(buf)
            if uuid:
                committed.append(uuid)
        if committed:
            log.info(
                "try_commit_pending: committed %d pending buffer(s)",
                len(committed),
            )
        return committed

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Diagnostic snapshot for /who or a debug endpoint."""
        return {
            "enrolled_total": self.enrolled_count,
            "merged_total": self.merged_count,
            "rejected_total": self.rejected_count,
            "active_buffers": len(self._buffers),
            "buffers": {
                tid: {
                    "samples_buffered": len(buf.crops),
                    "first_seen_ago": round(time.time() - buf.first_seen, 1),
                    "committed_uuid": buf.committed_uuid,
                }
                for tid, buf in self._buffers.items()
            },
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _commit(
        self,
        buf: _TrackBuffer,
        name: str = "",
        forced: bool = False,
    ) -> str:
        """Commit a buffered track into the gallery.

        Decision path:
          1. Compute the buffer's mean embedding (more reliable than any
             single frame — averaging suppresses noise).
          2. Compare against ALL existing centroids. If the best match's
             cosine sim >= ``merge_thr``, append samples to that UUID
             (merge path). This catches the "known person in bad lighting
             showed up as 'uncertain' per-frame, but their buffer mean is
             clearly them" case.
          3. Otherwise, create a brand-new UUID.

        Forced commits (from /set_name) ALWAYS create a new UUID with the
        given name — the LLM has named this specific track, so honor the
        intent even if the embedding is close to an existing identity.
        (If the user has a twin in the gallery, forced commit is the right
        choice; the LLM/operator made the call.)

        Sets ``buf.committed_uuid`` so further observe() calls for this
        track don't try to enroll again.
        """
        # 1. Try merging into an existing UUID (unless forced — see docstring)
        target_uuid: Optional[str] = None
        merged_into_name: str = ""
        merged_sim: float = 0.0
        if not forced and self.merge_thr > 0.0:
            target_uuid, merged_into_name, merged_sim = self._find_merge_target(buf)

        try:
            if target_uuid is None:
                # 2a. Create new UUID
                uuid = self.gallery.create(
                    name=name,
                    source=("auto_enroll_forced" if forced else self.source_tag),
                    first_track_id=int(buf.track_id),
                    buffered_samples=len(buf.crops),
                )
            else:
                # 2b. Merge into existing UUID
                uuid = target_uuid

            saved = self.gallery.add_samples(
                uuid,
                list(buf.crops),
                embeddings=list(buf.embeddings),
            )
        except Exception as e:
            log.exception("auto-enroll commit failed for track %d: %s", buf.track_id, e)
            return ""

        buf.committed_uuid = uuid
        if target_uuid is None:
            self.enrolled_count += 1
            log.info(
                "auto-enroll track=%d → NEW uuid=%s (samples=%d, forced=%s, name=%r)",
                buf.track_id,
                uuid,
                len(saved),
                forced,
                name,
            )
        else:
            self.merged_count += 1
            log.info(
                "auto-enroll track=%d → MERGED into uuid=%s name=%r (samples=%d, sim=%.3f)",
                buf.track_id,
                uuid,
                merged_into_name,
                len(saved),
                merged_sim,
            )

        if self.on_committed is not None:
            try:
                self.on_committed(uuid, buf.track_id, len(saved))
            except Exception as e:
                log.warning("on_committed callback raised: %s", e)
        return uuid

    def _find_merge_target(
        self,
        buf: _TrackBuffer,
    ) -> Tuple[Optional[str], str, float]:
        """Check if buf.embeddings' mean matches an existing ANON UUID strongly
        enough to fold the new samples in. Returns ``(uuid, name, sim)`` of the
        best ANON match if sim >= merge_thr, else ``(None, "", 0.0)``.

        IMPORTANT: only ANON UUIDs (name="") are valid merge targets.
        Named identities (sean, alice, ...) are NEVER auto-merged into,
        because face embedding sim is unreliable enough that a passing
        stranger could mistakenly contaminate a named person's identity.
        Concrete example:
            sean has UUID A (name="sean") in gallery.
            stranger walks in, looks vaguely like sean.
            buffer mean sim to A's centroid = 0.52 (>= merge_thr=0.50).
            WITHOUT this guard: stranger's samples get folded into sean's UUID.
                                → sean's centroid is polluted, sean's recognition
                                  suffers for everyone going forward.
            WITH this guard:    A is named → skipped. Create new anon UUID
                                instead. LLM/operator can later confirm via
                                /gallery/find_similar + /gallery/merge if
                                they're really the same person.

        Why use the buffer mean instead of per-frame embeddings:
            The per-frame recognition pass already saw each individual
            embedding and concluded "uncertain" (otherwise this track
            wouldn't have buffered). But the MEAN of N tight samples is a
            cleaner signal than any single frame — noise averages out,
            identity signal stays. So the mean often crosses the merge
            threshold even when no single frame did.

        Example of legitimate auto-merge:
            anon UUID X was created last session (face seen but not named).
            Same person walks in again, gets buffered to a new track.
            Buffer mean sim against X's centroid = 0.65.
            X has name="" (anon) → merge into X (samples folded in).
        """
        if not buf.embeddings:
            return None, "", 0.0
        try:
            feats, names, uuids = self.gallery.get_centroids()
        except Exception as e:
            log.warning("auto-enroll: could not read gallery centroids: %s", e)
            return None, "", 0.0
        if feats is None or feats.shape[0] == 0:
            # Empty gallery — nothing to merge with
            return None, "", 0.0

        # Mean of buffer embeddings, L2-normalized
        buf_mean = np.mean(np.stack(buf.embeddings, axis=0), axis=0)
        n = float(np.linalg.norm(buf_mean))
        if n < 1e-12:
            return None, "", 0.0
        buf_mean = (buf_mean / n).astype(np.float32)

        # Cosine sims against all gallery centroids
        sims = feats @ buf_mean
        # Sort descending so we can pick the best ANON match (not just best
        # overall, which might be a named UUID we must NOT merge into).
        order = np.argsort(-sims)

        for idx in order:
            j = int(idx)
            cand_sim = float(sims[j])
            cand_name = names[j]
            cand_uuid = uuids[j]
            if cand_sim < self.merge_thr:
                # Beyond this point all candidates are below threshold —
                # nothing left to consider.
                break
            if cand_name and cand_name.strip():
                # Named UUID — skip. Even if this is the closest match, we
                # refuse to auto-merge into a named identity. Log so it's
                # visible in demo what got rejected.
                log.debug(
                    "auto-enroll merge_target SKIP named uuid=%s name=%r sim=%.3f "
                    "(named UUIDs never auto-merged into; will create new anon)",
                    cand_uuid,
                    cand_name,
                    cand_sim,
                )
                continue
            # Anonymous match — valid merge target
            log.info(
                "auto-enroll merge_target FOUND anon uuid=%s sim=%.3f "
                "(sample fold-in)",
                cand_uuid,
                cand_sim,
            )
            return cand_uuid, cand_name, cand_sim

        return None, "", 0.0

    @staticmethod
    def _min_pairwise_sim(embeddings: List[np.ndarray]) -> float:
        """Return the WORST pairwise cosine sim across the buffer.

        Using the worst pair (not the average) is intentional — one bad
        sample in a tight cluster shouldn't enroll, even if every other
        pair is fine. The threshold acts as a sanity guard against
        BoTSORT track-id mix-ups.
        """
        if len(embeddings) < 2:
            return 1.0  # vacuously tight
        # Embeddings are already L2-normalized; dot product == cosine.
        M = np.stack(embeddings, axis=0)
        sims = M @ M.T
        # Zero the diagonal so we don't pick up self-similarity
        n = sims.shape[0]
        sims[np.arange(n), np.arange(n)] = 1.0
        # We want the WORST off-diagonal element.
        off_diag = sims[~np.eye(n, dtype=bool)]
        return float(off_diag.min())
