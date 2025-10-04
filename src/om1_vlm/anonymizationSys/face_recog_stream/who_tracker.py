from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple


# ------------------------------- Who Tracker ------------------------------- #
class WhoTracker:
    """Tracks identities seen now and over a short lookback window."""

    def __init__(self, lookback_sec: float = 2.0):
        self.lookback_sec = float(lookback_sec)
        self._events: deque[Tuple[float, List[str]]] = deque(
            maxlen=300
        )  # (ts, names[])
        self._last_now: List[str] = []
        self._lock = threading.Lock()

    def update_now(self, names: List[Optional[str]]) -> None:
        """Update current identities; names may include 'unknown' or None."""
        now = time.time()
        flat: List[str] = [n for n in names if n is not None]
        with self._lock:
            self._last_now = flat
            self._events.append((now, flat))
            cutoff = now - self.lookback_sec
            while self._events and self._events[0][0] < cutoff:
                self._events.popleft()

    def snapshot(self, recent_sec: Optional[float] = None) -> Dict:
        """Summarize who is here now and within recent_sec."""
        with self._lock:
            now_list = list(self._last_now)
            if recent_sec is None:
                recent_sec = self.lookback_sec
            cutoff = time.time() - float(recent_sec)
            recent: List[str] = []
            for ts, names in self._events:
                if ts >= cutoff:
                    recent.extend(names)

        def summarize(seq: List[str]) -> Dict[str, int]:
            return dict(Counter(seq)) if seq else {}

        def is_named(x: str) -> bool:
            return x and (x != "unknown")

        now_named = [n for n in now_list if is_named(n)]
        rec_named = [n for n in recent if is_named(n)]
        now_unknown = sum(1 for n in now_list if n == "unknown")
        rec_unknown = sum(1 for n in recent if n == "unknown")

        return {
            "now": now_named,
            "now_counts": summarize(now_named),
            "recent_sec": recent_sec,
            "recent_counts": summarize(rec_named),
            "unknown_now": now_unknown,
            "unknown_recent": rec_unknown,
        }
