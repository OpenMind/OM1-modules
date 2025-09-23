from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def draw_overlays(
    img: np.ndarray,
    dets: Optional[np.ndarray],
    names: Optional[List[Optional[str]]] = None,
    kpss: Optional[np.ndarray] = None,
    draw_boxes: bool = True,
    draw_names: bool = True,
    show_score_fallback: bool = True,
    box_color: Tuple[int, int, int] = (255, 0, 255),
    kps_color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """
    Draw detection boxes, optional names (or scores), and keypoints on an image.

    Parameters
    ----------
    img : np.ndarray
        BGR image. Modified in-place and also returned.
    dets : np.ndarray | None
        Shape (N, 5) array of [x1, y1, x2, y2, score]. If None/empty, no-op.
    names : list[str|None] | None
        Optional labels aligned with dets. If an entry is None/"" and
        `show_score_fallback=True`, the detection score is shown instead.
    kpss : np.ndarray | None
        Optional keypoints per detection, shape (N, 5, 2).
    draw_boxes : bool
        Whether to draw rectangles for detections.
    draw_names : bool
        Whether to draw names/scores.
    show_score_fallback : bool
        When True, draw score if name is missing/empty.
    box_color : (int, int, int)
        BGR color for boxes and text.
    kps_color : (int, int, int)
        BGR color for keypoints.

    Returns
    -------
    np.ndarray
        The same `img` for chaining.
    """
    if dets is None or dets.size == 0:
        return img

    H, W = img.shape[:2]
    # Scale thickness/font with image size for readability
    t = max(1, int(round(min(H, W) / 400.0)))
    fs = max(0.5, min(1.2, min(H, W) / 600.0))
    txt_thick = max(1, t)

    for i in range(dets.shape[0]):
        x1, y1, x2, y2, sc = dets[i]
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

        # Clip to image bounds to avoid OpenCV warnings
        x1i = max(0, min(W - 1, x1i))
        y1i = max(0, min(H - 1, y1i))
        x2i = max(0, min(W - 1, x2i))
        y2i = max(0, min(H - 1, y2i))

        if draw_boxes:
            cv2.rectangle(
                img, (x1i, y1i), (x2i, y2i), box_color, t, lineType=cv2.LINE_AA
            )

        # Build tag
        tag: Optional[str] = None
        if draw_names:
            if names is not None and i < len(names) and names[i]:
                tag = str(names[i])
            elif show_score_fallback:
                tag = f"{float(sc):.2f}"

        # Put text with a filled background for legibility
        if tag:
            (tw, th), baseline = cv2.getTextSize(
                tag, cv2.FONT_HERSHEY_SIMPLEX, fs, txt_thick
            )
            tx, ty = x1i, max(0, y1i - 6)
            # Background box
            bg_tl = (tx - 1, max(0, ty - th - baseline - 2))
            bg_br = (tx + tw + 1, ty + 2)
            cv2.rectangle(img, bg_tl, bg_br, (0, 0, 0), thickness=-1)
            # Text
            cv2.putText(
                img,
                tag,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs,
                box_color,
                txt_thick,
                cv2.LINE_AA,
            )

        # Keypoints
        if kpss is not None and i < len(kpss):
            for kx, ky in kpss[i]:
                cv2.circle(
                    img,
                    (int(kx), int(ky)),
                    max(1, t + 1),
                    kps_color,
                    -1,
                    lineType=cv2.LINE_AA,
                )

    return img
