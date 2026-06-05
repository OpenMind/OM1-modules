"""Continuous-refresh manager for the face gallery.

When a track is confidently identified, the recognized embedding may add
new feature diversity to the known UUID's centroid — e.g. a different
angle, lighting, or expression. Folding these into the gallery makes
recognition more robust over time.

Guards (defense against polluting a named identity with misattributions):

  A. Match confidence — caller verifies tier == confident AND sim is well
     above ``sim_thr`` (sim_thr + ``min_extra_confidence``) before calling.
  B. Per-UUID rate limit — at most one refresh attempt per
     ``min_refresh_interval_sec`` per UUID. Prevents a single session
     from flooding a UUID with near-identical samples and prevents the
     refresh path from dominating CPU.
  C. Diversity check — applied inside ``gallery.refresh_sample``: the new
     sample's sim to centroid must be in ``[min_diversity_sim,
     max_diversity_sim]``. Skips near-duplicates and likely
     misidentifications.

The gallery itself enforces the FIFO sample cap (``max_samples_per_uuid``)
so disk doesn't grow unboundedly even with permissive guard settings.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)


class SampleRefreshManager:
    """Stateful helper that decides when to call ``gallery.refresh_sample``.

    Holds the per-UUID rate-limit state (Guard B). Guard A (sim threshold)
    is a constructor parameter; Guard C (diversity) is delegated to the
    gallery.

    Wire into the main loop after recognition runs. For each track that's
    been confidently identified this frame::

        if refresh_mgr.should_refresh(uuid, sim, now):
            refresh_mgr.observe(uuid, crop, embedding, now)
    """

    def __init__(
        self,
        gallery,
        *,
        sim_thr: float = 0.55,
        min_extra_confidence: float = 0.10,
        min_refresh_interval_sec: float = 10.0,
        min_diversity_sim: float = 0.65,
        max_diversity_sim: float = 0.92,
    ) -> None:
        self.gallery = gallery
        # Guard A threshold: only refresh when sim >= sim_thr + min_extra
        self.refresh_thr = float(sim_thr) + float(min_extra_confidence)
        # Guard B: min seconds between refresh attempts per UUID
        self.min_refresh_interval_sec = float(min_refresh_interval_sec)
        # Guard C parameters (passed through to gallery.refresh_sample)
        self.min_diversity_sim = float(min_diversity_sim)
        self.max_diversity_sim = float(max_diversity_sim)

        # Per-UUID monotonic timestamp of the LAST refresh ATTEMPT (success
        # or skip — once we try, we wait the interval before trying again).
        # This prevents a long-lived track from spamming the gallery.
        self._last_attempt: Dict[str, float] = {}

        # Counters for /gallery/auto_enroll_status-style introspection.
        self.attempts = 0
        self.refreshed = 0
        self.skipped_low_sim = 0  # Guard A failure
        self.skipped_rate_limit = 0  # Guard B failure
        self.skipped_diversity = 0  # Guard C failure (from gallery)

    def set_sim_thr(self, sim_thr: float, min_extra_confidence: float = 0.10) -> None:
        """Hot-update Guard A threshold (e.g. when ``/config`` changes sim_thr)."""
        self.refresh_thr = float(sim_thr) + float(min_extra_confidence)

    def should_refresh(self, uuid: str, sim: float, now: float) -> bool:
        """Cheap pre-check before computing whether to actually call
        ``gallery.refresh_sample``. Returns True only if Guards A + B pass.

        Guard C is checked inside ``observe``/``gallery.refresh_sample``
        because it needs the embedding (more expensive).
        """
        if not uuid:
            return False
        # Guard A
        if sim < self.refresh_thr:
            return False
        # Guard B
        last = self._last_attempt.get(uuid, 0.0)
        if (now - last) < self.min_refresh_interval_sec:
            return False
        return True

    def observe(
        self,
        uuid: str,
        crop: np.ndarray,
        embedding: np.ndarray,
        now: float,
        sim: Optional[float] = None,
    ) -> bool:
        """Run a refresh attempt. Returns True if a sample was actually added,
        False otherwise. Always marks the UUID as "tried at ``now``" for
        rate-limiting (so a flood of low-quality observations doesn't
        bypass Guard B).
        """
        if not uuid:
            return False
        self.attempts += 1
        self._last_attempt[uuid] = now

        # Re-check Guard A here too in case caller bypassed should_refresh
        if sim is not None and sim < self.refresh_thr:
            self.skipped_low_sim += 1
            return False

        try:
            added = self.gallery.refresh_sample(
                uuid,
                crop,
                embedding,
                min_diversity_sim=self.min_diversity_sim,
                max_diversity_sim=self.max_diversity_sim,
            )
        except Exception as e:
            log.warning("refresh observe: gallery.refresh_sample raised: %s", e)
            return False

        if added:
            self.refreshed += 1
        else:
            # gallery.refresh_sample logs the specific reason
            self.skipped_diversity += 1
        return added

    def snapshot(self) -> dict:
        """Diagnostic snapshot for /gallery/refresh_status (or debug logs)."""
        return {
            "refresh_thr": round(self.refresh_thr, 3),
            "min_refresh_interval_sec": self.min_refresh_interval_sec,
            "diversity_range": [
                round(self.min_diversity_sim, 3),
                round(self.max_diversity_sim, 3),
            ],
            "attempts": self.attempts,
            "refreshed": self.refreshed,
            "skipped_low_sim": self.skipped_low_sim,
            "skipped_rate_limit": self.skipped_rate_limit,
            "skipped_diversity": self.skipped_diversity,
            "tracked_uuids": len(self._last_attempt),
        }
