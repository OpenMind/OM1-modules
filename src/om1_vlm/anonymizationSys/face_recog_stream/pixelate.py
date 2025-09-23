from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def expand_clip(
    x1: int, y1: int, x2: int, y2: int, margin: float, W: int, H: int
) -> Tuple[int, int, int, int]:
    """Expand a box by margin and clip to image bounds.

    Parameters
    ----------
    x1, y1, x2, y2 : int
        Original box (top-left, bottom-right).
    margin : float
        Fractional expansion on width/height (e.g., 0.25 = +25%).
    W, H : int
        Image width/height for clipping.

    Returns
    -------
    Tuple[int, int, int, int]
        Expanded/clipped integer box (x1, y1, x2, y2).
    """
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = (x2 - x1), (y2 - y1)
    w *= 1.0 + margin
    h *= 1.0 + margin
    x1n = max(0, int(cx - w * 0.5))
    y1n = max(0, int(cy - h * 0.5))
    x2n = min(W - 1, int(cx + w * 0.5))
    y2n = min(H - 1, int(cy + h * 0.5))
    return x1n, y1n, x2n, y2n


def pixelate_roi(
    img,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    blocks_on_short: int = 8,
    noise_sigma: float = 0.0,
) -> None:
    """Apply pixelation (and optional noise) to a rectangular ROI in-place.

    Parameters
    ----------
    img : ndarray (H,W,3)
        BGR image (modified in-place).
    x1, y1, x2, y2 : int
        ROI box.
    blocks_on_short : int, default 8
        Number of blocks along the shorter side; controls pixel size.
    noise_sigma : float, default 0.0
        Std-dev of Gaussian noise added after pixelation.

    Returns
    -------
    None
        The function modifies the input image directly.
    """
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    short = max(1, min(w, h))
    scale = max(1, int(blocks_on_short)) / float(short)
    small_w = max(1, int(round(w * scale)))
    small_h = max(1, int(round(h * scale)))
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if noise_sigma and noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, size=big.shape).astype(np.float32)
        big = np.clip(big.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img[y1:y2, x1:x2] = big
