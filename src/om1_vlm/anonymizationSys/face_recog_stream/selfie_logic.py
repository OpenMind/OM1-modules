"""
Pure logic for the selfie enrollment pipeline.

Stateless. All shared state (gallery, frame_state, cfg) is owned by HttpAPI
and passed in by parameter. No GPU work happens here; embeddings are computed
elsewhere (main thread via run_job_sync) and passed in.

Place at: src/om1_vlm/anonymizationSys/face_recog_stream/selfie_logic.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 5-point landmark indices (SCRFD / RetinaFace convention).
# These are subject-perspective: subject's left eye, subject's right eye,
# nose, subject's left mouth, subject's right mouth.
EYE_A_IDX = 0
EYE_B_IDX = 1
NOSE_IDX = 2

_VALID_ID = re.compile(r"^[a-z0-9_-]+$")
# Reject names ending in _<digits> to avoid colliding with the suffix system.
# E.g. user-supplied "wendy_1" would confuse next_suffix logic.
_HAS_SUFFIX = re.compile(r"_\d+$")


# Decision objects
@dataclass
class DedupDecision:
    """Outcome of resolve_target_id()."""

    id: Optional[str] = None  # Target folder name on success
    merged: bool = False  # True if saving into existing folder
    reject: bool = False  # True if cross-name conflict (no force)
    match_label: Optional[str] = None  # Closest existing label (for telemetry)
    match_sim: float = 0.0  # Cosine to closest match


# Geometry: frontality from 5-point landmarks
def frontality(kps: Optional[np.ndarray]) -> float:
    """
    Score in [0, 1] of how frontal a face is, from its 5 keypoints.

    1.0 = eyes level AND nose centered between them.
    0.0 = full profile OR extreme head tilt.

    Robust to detector landmark ordering: we don't rely on which point is the
    "left" vs "right" eye, only on the geometric relationship between the two
    eye points and the nose. Absolute values everywhere.
    """
    if kps is None or len(kps) < 3:
        return 0.0

    eye_a = kps[EYE_A_IDX]
    eye_b = kps[EYE_B_IDX]
    nose = kps[NOSE_IDX]

    eye_dx = abs(float(eye_b[0]) - float(eye_a[0]))
    if eye_dx < 1.0:
        # Degenerate: eyes at near-identical x (extreme angle or detector
        # malfunction). Cannot meaningfully assess frontality.
        return 0.0

    # Yaw: how centered is the nose between the two eyes?
    eye_mid_x = (float(eye_a[0]) + float(eye_b[0])) / 2.0
    nose_off = (float(nose[0]) - eye_mid_x) / eye_dx
    yaw_score = max(0.0, 1.0 - 2.0 * abs(nose_off))

    # Roll: how level are the eyes?
    eye_dy = abs(float(eye_b[1]) - float(eye_a[1]))
    roll_score = max(0.0, 1.0 - eye_dy / eye_dx)

    return yaw_score * roll_score


# Scoring: who is the robot being addressed by?
def score_face(
    det: np.ndarray,
    kps: Optional[np.ndarray],
    frame_area: float,
) -> float:
    """
    Engagement score in [0, 1] (resolution-independent).

    score = (bbox_area / frame_area) * frontality

    "Engaged" = close (large in frame) AND looking at robot (frontal).
    Multiplicative — failing either component kills the score, so a huge
    but turned-away face cannot beat a smaller frontal one.

    Normalizing by frame_area decouples the threshold from camera resolution:
    a face covering 1% of the frame is the same "size" at 720p and 4K.
    """
    if frame_area <= 0:
        return 0.0
    x1, y1, x2, y2 = float(det[0]), float(det[1]), float(det[2]), float(det[3])
    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    front = frontality(kps) if kps is not None else 0.0
    return (bbox_area / frame_area) * front


# Frame-level filters
def check_ambiguity(
    scores: List[float],
    ratio_thr: float,
    score_floor: float,
) -> Tuple[bool, float]:
    """
    Detect "who is the target" ambiguity within the current frame.

    Returns (is_ambiguous, top_ratio).

    Ambiguous iff top-1 AND top-2 are BOTH above score_floor (both faces
    are "engaging") AND top2/top1 > ratio_thr (they're close in engagement).

    Faces below the floor don't count — a small background bystander can't
    create ambiguity even if their score is close to the target's.
    """
    if len(scores) < 2:
        return False, 0.0

    sorted_desc = sorted(scores, reverse=True)
    top1, top2 = sorted_desc[0], sorted_desc[1]

    if top1 < score_floor or top2 < score_floor:
        return False, 0.0

    ratio = top2 / top1 if top1 > 0 else 0.0
    return ratio > ratio_thr, ratio


def quality_check(
    crop_112: np.ndarray,
    det: np.ndarray,
    kps: Optional[np.ndarray],
    cfg: dict,
) -> Tuple[bool, str]:
    """
    Per-frame quality gate on the selected face.

    Returns (ok, reason). reason is "" on success, otherwise one of:
    too_small | low_conf | bad_pose | blurry | too_dark | too_bright
    """
    x1, y1, x2, y2 = float(det[0]), float(det[1]), float(det[2]), float(det[3])
    conf = float(det[4]) if len(det) >= 5 else 1.0
    w, h = x2 - x1, y2 - y1

    if min(w, h) < float(cfg["selfie_min_face_px"]):
        return False, "too_small"

    if conf < float(cfg["selfie_min_conf"]):
        return False, "low_conf"

    if kps is not None:
        if frontality(kps) < float(cfg["selfie_min_frontality"]):
            return False, "bad_pose"

    gray = cv2.cvtColor(crop_112, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < float(cfg["selfie_sharp_thr"]):
        return False, "blurry"

    mb = float(gray.mean())
    if mb < float(cfg["selfie_brightness_min"]):
        return False, "too_dark"
    if mb > float(cfg["selfie_brightness_max"]):
        return False, "too_bright"

    return True, ""


# Identity name handling
def validate_identity_name(name: str) -> Tuple[bool, str]:
    """
    Check identity name format. Returns (ok, reason).

    Rules:
    - non-empty, ≤32 chars
    - lowercase alphanumeric, underscore, dash only
    - must NOT end in `_<digits>` (reserved for suffix system)
    """
    if not name:
        return False, "empty"
    if len(name) > 32:
        return False, "too_long"
    if not _VALID_ID.match(name):
        return False, "invalid_chars"
    if _HAS_SUFFIX.search(name):
        return False, "ends_with_suffix"
    return True, ""


def is_same_root(label: str, root: str) -> bool:
    """True iff `label` is exactly `root` or `root_<digits>`."""
    if label == root:
        return True
    return bool(re.fullmatch(rf"{re.escape(root)}_\d+", label))


def next_suffix(name: str, existing_labels: List[str]) -> str:
    """
    Next available label in the {name, name_1, name_2, ...} family.

    - If `name` itself is free → return `name`.
    - Else return `name_<max_n + 1>` where max_n scans existing `name_<n>` labels.
    """
    existing = set(existing_labels)
    if name not in existing:
        return name

    pat = re.compile(rf"^{re.escape(name)}_(\d+)$")
    max_n = 0
    for label in existing:
        m = pat.match(label)
        if m:
            max_n = max(max_n, int(m.group(1)))

    return f"{name}_{max_n + 1}"


# Dedup
def resolve_target_id(
    embedding: np.ndarray,
    requested_name: str,
    gal_feats: Optional[np.ndarray],
    gal_labels: List[str],
    cfg: dict,
    force: bool = False,
) -> DedupDecision:
    """
    Decide which gallery folder a new face should go into.

    One full-table cosine search, then 2×2 decision matrix on
    (similarity tier, label-matches-requested-name).

    Thresholds (from cfg):
    - cross_thr (~0.60): strong cross-name match → reject (unless force)
    - merge_thr (~0.45): same-name family match → merge into existing folder

    Behavior:
    - Empty gallery                                    → use requested_name
    - sim >= cross_thr, same-root label                → merge
    - sim >= cross_thr, different name, force=False    → REJECT
    - sim >= cross_thr, different name, force=True     → new with suffix
    - merge_thr <= sim < cross_thr, same-root          → merge
    - otherwise                                        → new with suffix
    """
    cross_thr = float(cfg["selfie_cross_name_thr"])
    merge_thr = float(cfg["selfie_merge_thr"])

    if gal_feats is None or len(gal_labels) == 0:
        return DedupDecision(id=requested_name)

    # gal_feats rows and embedding are both unit-norm
    sims = gal_feats @ embedding
    j = int(np.argmax(sims))
    best_sim = float(sims[j])
    best_label = gal_labels[j]

    same_root = is_same_root(best_label, requested_name)

    # Strong match
    if best_sim >= cross_thr:
        if same_root:
            return DedupDecision(
                id=best_label,
                merged=True,
                match_label=best_label,
                match_sim=best_sim,
            )
        if force:
            return DedupDecision(
                id=next_suffix(requested_name, gal_labels),
                match_label=best_label,
                match_sim=best_sim,
            )
        return DedupDecision(
            reject=True,
            match_label=best_label,
            match_sim=best_sim,
        )

    # Soft match to same-root → merge
    if best_sim >= merge_thr and same_root:
        return DedupDecision(
            id=best_label,
            merged=True,
            match_label=best_label,
            match_sim=best_sim,
        )

    # No qualifying match → new identity (possibly with suffix)
    return DedupDecision(
        id=next_suffix(requested_name, gal_labels),
        match_label=best_label,
        match_sim=best_sim,
    )


# Multi-frame: running mean + novelty + consistency
def update_running_mean(
    current_mean: Optional[np.ndarray],
    new_v: np.ndarray,
    new_count: int,
) -> np.ndarray:
    """
    Incremental L2-normalized running mean.

    `new_count` is the count AFTER adding new_v (1, 2, 3, ...).
    Always returns a unit-norm vector.
    """
    if current_mean is None or new_count <= 1:
        out = new_v.astype(np.float32, copy=True)
    else:
        out = current_mean * (new_count - 1) + new_v

    norm = float(np.linalg.norm(out))
    if norm < 1e-12:
        return new_v.astype(np.float32, copy=True)
    return (out / norm).astype(np.float32, copy=False)


def cosine_to_mean(running_mean: np.ndarray, new_v: np.ndarray) -> float:
    """
    Cosine similarity. Both inputs assumed unit-norm.

    Used for BOTH novelty (skip frames too similar to running mean)
    AND identity consistency (reject frames too different from collected so far).
    """
    return float(np.dot(running_mean, new_v))
