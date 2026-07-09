"""vvad_overlay.py — transient "who is speaking" badge for the video overlay.

Decoupled from the per-detection renderer (draw.py): at the /selfie trigger,
http_api calls flag(...) with the speaking faces' bboxes; the frame loop calls
draw(...) once per frame. Badges auto-expire after `ttl` seconds, so the marker
only shows right around the moment someone gives their name — no persistent
detection. Multiple speakers => multiple badges. The chosen (enrolled) face is
green; any other simultaneous speakers are amber.

State is module-level (one camera/stream), guarded by a lock since flag() runs
on the selfie thread and draw() on the frame-loop thread.
"""

from __future__ import annotations

import threading
import time
from typing import List, Tuple

import cv2

_lock = threading.Lock()
_until: float = 0.0
# (bbox, is_chosen, score|None) — score is P(speaking) in [0,1], shown on badge
_entries: List[Tuple[Tuple[int, int, int, int], bool, object]] = []


def flag(bboxes_chosen, ttl: float = 1.6) -> None:
    """Mark speaking faces for ~ttl seconds.

    Each entry is ((x1, y1, x2, y2), is_chosen) or, to show the confidence on
    screen, ((x1, y1, x2, y2), is_chosen, score) where score is P(speaking) in
    [0, 1]. Pass an empty list to badge nobody (e.g. no one was speaking).
    """
    global _until, _entries
    parsed = []
    for e in bboxes_chosen:
        bb, ch = e[0], e[1]
        sc = e[2] if len(e) > 2 else None
        parsed.append(
            (
                tuple(int(v) for v in bb),
                bool(ch),
                (float(sc) if sc is not None else None),
            )
        )
    with _lock:
        _entries = parsed
        _until = time.time() + float(ttl)


def clear() -> None:
    """Clear any flagged entries and expire the overlay immediately."""
    global _until, _entries
    with _lock:
        _entries = []
        _until = 0.0


def draw(img):
    """Draw speaking badges if not expired. Modifies img in place; returns it.

    Cheap no-op when nothing is flagged or the badge has expired, so it's safe
    to call every frame unconditionally.
    """
    now = time.time()
    with _lock:
        if now > _until or not _entries:
            return img
        entries = list(_entries)

    H, W = img.shape[:2]
    t = max(2, int(round(min(H, W) / 300.0)))
    fs = max(0.5, min(1.2, min(H, W) / 600.0))
    for (x1, y1, x2, y2), is_chosen, score in entries:
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        color = (0, 220, 0) if is_chosen else (150, 150, 150)  # BGR: green / gray
        tag = "SPEAKING" if is_chosen else "quiet"
        if score is not None:
            tag = "%s %.2f" % (tag, score)  # e.g. "SPEAKING 0.87"
        # thicker outline so it stands out over the magenta detection box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, t + 1, lineType=cv2.LINE_AA)
        # badge BELOW the box (names are drawn above it by draw_overlays)
        (tw, th), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        ty = min(H - 2, y2 + th + 6)
        cv2.rectangle(
            img,
            (x1 - 1, ty - th - bl - 2),
            (x1 + tw + 1, ty + 2),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(
            img, tag, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, color, t, cv2.LINE_AA
        )
    return img
