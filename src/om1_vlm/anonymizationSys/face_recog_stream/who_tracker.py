from __future__ import annotations

import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple


# ------------------------------- Who Tracker ------------------------------- #
class WhoTracker:
    """Tracks identities seen now and over a short lookback window."""

    def __init__(self, lookback_sec: float = 10.0):
        self.lookback_sec = float(lookback_sec)
        self._events: deque[Tuple[float, List[str]]] = deque(
            maxlen=300
        )  # (ts, names[])
        self._last_now: List[str] = []
        self._lock = threading.Lock()

    def update_now(self, names: List[Optional[str]]) -> None:
        """Update current identities; names may include 'unknown' or None."""
        now_ts = time.time()
        flat: List[str] = [n for n in names if n is not None]
        with self._lock:
            self._last_now = flat
            self._events.append((now_ts, flat))
            cutoff = now_ts - self.lookback_sec
            while self._events and self._events[0][0] < cutoff:
                self._events.popleft()

    def snapshot(self, recent_sec: Optional[float] = None) -> Dict:
        """Summarize who is here now and (for recent) use max-per-frame semantics.
        EX: {"server_ts": 1761692303.4755309, "recent_sec": 4.0, "now": ["wendy"], 
            "unknown_now": 0, "frames_recent": 57, "frames_with_unknown": 0, 
            "recent_name_frames": {"wendy": 57}, "unknown_recent": 0}
        """
        with self._lock:
            now_list = list(self._last_now)
            if recent_sec is None:
                recent_sec = self.lookback_sec
            cutoff = time.time() - float(recent_sec)
            recent_frames: List[List[str]] = [
                names for ts, names in self._events if ts >= cutoff
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
        frames_recent = len(recent_frames)
        frames_with_unknown = 0
        unknown_recent_peak = 0
        recent_name_frames: Dict[str, int] = {}

        for frame_names in recent_frames:
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

        return {
            "server_ts": time.time(),
            "recent_sec": float(recent_sec),
            "now": now_named,
            "unknown_now": int(now_unknown),
            "frames_recent": int(frames_recent),
            "frames_with_unknown": int(frames_with_unknown),
            "recent_name_frames": recent_name_frames,
            "unknown_recent": int(unknown_recent_peak),
        }
