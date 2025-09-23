from __future__ import annotations

import glob
import os.path as osp
from typing import List

# --- add to utils.py (文件末尾即可) ---
import numpy as np


def pick_topk_indices(
    dets: np.ndarray, topk: int, H: int, W: int, mode: str = "score_area"
) -> np.ndarray:
    """
    Select indices of the top-K detections to run recognition on.

    By default, detections are ranked by (confidence × box area).

    Parameters
    ----------
    dets : np.ndarray
        Array of detections with shape (N, 5): [x1, y1, x2, y2, score].
    topk : int
        Maximum number of indices to return.
    H, W : int
        Image height and width (kept for future use/expansion).
    mode : str
        Ranking mode. Currently supports "score_area" (recommended).

    Returns
    -------
    np.ndarray
        Indices of the selected detections, shape (K,), dtype int32.
    """
    if dets is None or dets.shape[0] == 0 or topk <= 0:
        return np.array([], dtype=np.int32)
    x1, y1, x2, y2, sc = dets.T
    area = (x2 - x1) * (y2 - y1)
    score = sc * np.maximum(area, 1.0)
    order = np.argsort(score)[::-1]
    return order[:topk].astype(np.int32)


def infer_arc_batched(arc, crops, max_bs: int = 4):
    """
    Select indices of the top-K detections to run recognition on.

    By default, detections are ranked by (confidence × box area).

    Parameters
    ----------
    dets : np.ndarray
        Array of detections with shape (N, 5): [x1, y1, x2, y2, score].
    topk : int
        Maximum number of indices to return.
    H, W : int
        Image height and width (kept for future use/expansion).
    mode : str
        Ranking mode. Currently supports "score_area" (recommended).

    Returns
    -------
    np.ndarray
        Indices of the selected detections, shape (K,), dtype int32.
    """
    if not crops:
        return np.zeros((0, 512), dtype=np.float32)
    outs = []
    for i in range(0, len(crops), max_bs):
        outs.append(arc.infer(crops[i : i + max_bs]))
    return np.vstack(outs) if outs else np.zeros((0, 512), dtype=np.float32)


def list_images(path: str) -> List[str]:
    """Return a sorted list of image file paths.

    Parameters
    ----------
    path : str
        A directory, a single file, or a glob pattern.

    Returns
    -------
    List[str]
        Sorted image paths. Empty if none.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG")
    if osp.isdir(path):
        files: List[str] = []
        for e in exts:
            files += glob.glob(osp.join(path, e))
        return sorted(files)
    if any(ch in path for ch in "*?[]"):
        return sorted(glob.glob(path))
    return [path] if osp.exists(path) else []
