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


# Decision objects
@dataclass
class DedupDecision:
    """Outcome of resolve_target_id().

    Fields
    ------
    uuid : str | None
        UUID to add samples to. ``None`` means "create a new UUID with
        ``name``" — caller invokes ``gallery.create(name).
    name : str
        Display name to use. For merges, this is the existing UUID's name
        (may differ in casing from the requested name; we use what's on disk).
        For new identities, this is the requested name verbatim.
    merged : bool
        True iff ``uuid`` is set (we're merging into an existing identity).
    reject : bool
        True iff this selfie should be rejected because the face strongly
        matches a *different* named identity (``match_name``), and
        ``force=False``. Indicates likely confusion ("are you Bob, not Wendy?").
    match_name : str | None
        Best-matching existing display name, for telemetry / user-facing
        error messages.
    match_uuid : str | None
        UUID of the best match (regardless of whether we merged with it).
    match_sim : float
        Cosine similarity to the best match.
    """

    uuid: Optional[str] = None
    name: str = ""
    merged: bool = False
    reject: bool = False
    match_name: Optional[str] = None
    match_uuid: Optional[str] = None
    match_sim: float = 0.0


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


# ---------------------------------------------------------------------------
# Recognition: 1:N identification with confidence tiering
# ---------------------------------------------------------------------------

# Recognition tier names. Exposed so callers don't have to spell strings.
TIER_CONFIDENT = "confident"
TIER_TENTATIVE = "tentative"
TIER_UNCERTAIN = "uncertain"


def recognize_from_sims(
    sims: np.ndarray,
    gal_labels: List[str],
    *,
    min_sim: float,
    strict_margin: float,
    loose_margin: float,
) -> Tuple[str, str, float, float]:
    """
    Decide identity from precomputed cosine similarities, with a 3-tier
    confidence output.

    Use this when you already have ``sims = gal_feats @ query`` (e.g. inside a
    batched recognition pass). Use ``recognize()`` for one-off queries.

    Decision tree (in order):
      1. Empty gallery                          → ("unknown", "uncertain")
      2. top1 sim < min_sim                     → ("unknown", "uncertain")
      3. top1 - top2 >= strict_margin           → (name, "confident")
      4. top1 - top2 >= loose_margin            → (name, "tentative")
      5. otherwise (ambiguous between names)    → ("unknown", "uncertain")

    Why tiered output:
    - At small gallery (N<50), top1 alone is reliable.
    - At larger gallery, many near-collisions exist; top1 alone gives false
      matches. The margin between top1 and top2 is a much better confidence
      signal than top1 itself.
    - But a strict margin gate over-filters: a person we *can* recognize will
      sometimes have margin=0.05 (clearly above zero but below strict). We
      want to surface that as "tentative" so the caller can ask for
      verification instead of giving up.

    Parameters
    ----------
    sims : ndarray (N_id,)
        Cosine similarities between the query and each gallery centroid.
        Assumed both are L2-normalized so this is just a dot product.
    gal_labels : list[str]
        Label per row of sims; ``len(gal_labels) == len(sims)``.
    min_sim : float
        Minimum top-1 cosine for any match (typical 0.40 - 0.50).
    strict_margin : float
        Required ``top1 - top2`` to call it "confident" (typical 0.08 - 0.12).
    loose_margin : float
        Required ``top1 - top2`` to call it "tentative" (typical 0.03 - 0.05).
        Must be strictly less than ``strict_margin``.

    Returns
    -------
    (name, tier, top1_sim, top2_sim) : (str, str, float, float)
        - name: gallery label or ``"unknown"``
        - tier: one of ``"confident"``, ``"tentative"``, ``"uncertain"``
        - top1_sim, top2_sim: raw scores for logging / debugging
          (top2_sim is 0.0 when gallery has only one entry)
    """
    if sims is None or len(sims) == 0:
        return ("unknown", TIER_UNCERTAIN, 0.0, 0.0)

    # Single-entry gallery: no margin to compute, fall back to threshold only.
    if len(sims) == 1:
        s1 = float(sims[0])
        if s1 < min_sim:
            return ("unknown", TIER_UNCERTAIN, s1, 0.0)
        return (gal_labels[0], TIER_CONFIDENT, s1, 0.0)

    # General case: find top-1 and top-2.
    # argpartition is O(N), faster than full sort when we only need top-K.
    # NOTE: k argument is the index position we want partitioned, not the
    # number of items. To pick top-2, k=1 is correct (positions 0 and 1 hold
    # the two smallest of -sims, which are the two largest of sims).
    # k=2 would crash with "kth out of bounds" when len(sims) == 2.
    top2_idx = np.argpartition(-sims, 1)[:2]
    if sims[top2_idx[0]] < sims[top2_idx[1]]:
        top2_idx = top2_idx[::-1]
    s1 = float(sims[top2_idx[0]])
    s2 = float(sims[top2_idx[1]])
    name = gal_labels[int(top2_idx[0])]

    if s1 < min_sim:
        return ("unknown", TIER_UNCERTAIN, s1, s2)

    margin = s1 - s2
    if margin >= strict_margin:
        return (name, TIER_CONFIDENT, s1, s2)
    if margin >= loose_margin:
        return (name, TIER_TENTATIVE, s1, s2)
    # Ambiguous: two identities tied within `loose_margin`. Better to refuse
    # than to commit to either. The temporal voting layer will see this as
    # "uncertain" and not count it toward any name.
    return ("unknown", TIER_UNCERTAIN, s1, s2)


def recognize(
    embedding: np.ndarray,
    gal_feats: Optional[np.ndarray],
    gal_labels: List[str],
    *,
    min_sim: float,
    strict_margin: float,
    loose_margin: float,
) -> Tuple[str, str]:
    """
    Convenience wrapper around :func:`recognize_from_sims` for callers that
    only need (name, tier).

    Computes ``sims = gal_feats @ embedding`` internally, then dispatches to
    ``recognize_from_sims``. Returns ``("unknown", "uncertain")`` for an
    empty gallery without touching ``gal_feats``.
    """
    if gal_feats is None or len(gal_feats) == 0 or len(gal_labels) == 0:
        return ("unknown", TIER_UNCERTAIN)
    sims = gal_feats @ embedding
    name, tier, _, _ = recognize_from_sims(
        sims,
        gal_labels,
        min_sim=min_sim,
        strict_margin=strict_margin,
        loose_margin=loose_margin,
    )
    return (name, tier)


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

    Note: in the UUID-based gallery there's no suffix collision concern, so
    names like ``"wendy_2"`` are allowed.
    """
    if not name:
        return False, "empty"
    if len(name) > 32:
        return False, "too_long"
    if not _VALID_ID.match(name):
        return False, "invalid_chars"
    return True, ""


# Dedup
def resolve_target_id(
    embedding: np.ndarray,
    requested_name: str,
    gal_feats: Optional[np.ndarray],
    gal_labels: List[str],
    gal_uuids: List[str],
    cfg: dict,
    force: bool = False,
) -> DedupDecision:
    """Decide whether a new selfie should merge with an existing UUID.

    Runs one cosine search against the in-memory centroids, then applies a
    two-tier decision:

    Thresholds (from ``cfg``)
    -------------------------
    ``selfie_cross_name_thr`` (~0.60)
        Above this, the face is "strongly recognized as someone". If that
        someone has a different name from the requested one → reject the
        selfie (suggests user confusion or impersonation). Override with
        ``force=True``.
    ``selfie_merge_thr`` (~0.45)
        Above this AND the matched identity already has the same name →
        merge (append samples to existing UUID). Below this → create a new
        UUID even if a similarly-named identity exists.

    Behavior matrix
    ---------------
    Empty gallery                                       → CREATE NEW UUID
    sim ≥ cross_thr, same-name match                    → MERGE
    sim ≥ cross_thr, different name, force=False        → REJECT
    sim ≥ cross_thr, different name, force=True         → CREATE NEW UUID
    merge_thr ≤ sim < cross_thr, same-name match        → MERGE
    otherwise                                           → CREATE NEW UUID

    Multiple UUIDs per name are explicitly allowed (two different selfie
    sessions for the same person produce two UUIDs; both will match her
    face). The merge path picks the single best-scoring UUID; future
    selfies will fold into whichever has the closest centroid.
    """
    cross_thr = float(cfg["selfie_cross_name_thr"])
    merge_thr = float(cfg["selfie_merge_thr"])
    requested_lower = requested_name.strip().lower()

    if gal_feats is None or len(gal_uuids) == 0:
        # Brand-new gallery — caller creates a fresh UUID with this name.
        return DedupDecision(uuid=None, name=requested_name)

    # gal_feats rows and embedding are both unit-norm
    sims = gal_feats @ embedding
    j = int(np.argmax(sims))
    best_sim = float(sims[j])
    best_name = gal_labels[j]
    best_uuid = gal_uuids[j]

    # Anonymous UUIDs (name="") are treated as "claimable" — the empty name
    # matches any requested name. This is the "selfie claims an anon" path:
    # when face_tracker auto-enrolled this face into UUID X with no name,
    # a follow-up selfie that matches X should fold into X and name it,
    # not reject (the old cross-name protection had a false-positive here:
    # "" ≠ "sean" was treated as a name conflict, blocking the natural
    # workflow of giving the anon a name).
    is_anon_match = (not best_name) or (not best_name.strip())
    same_name = is_anon_match or (best_name.strip().lower() == requested_lower)

    # Strong match — face is being recognized as a known identity
    if best_sim >= cross_thr:
        if same_name:
            # Merging into the matched UUID.
            # If the match is anon (no name yet), use the REQUESTED name —
            # the selfie is claiming this anon UUID. The caller will fold
            # selfie samples in AND rename the UUID. If the match already
            # has the requested name, keep the existing name (no-op rename).
            effective_name = requested_name if is_anon_match else best_name
            return DedupDecision(
                uuid=best_uuid,
                name=effective_name,
                merged=True,
                match_name=best_name,
                match_uuid=best_uuid,
                match_sim=best_sim,
            )
        if force:
            # User insists this is a different person despite the strong
            # match (twin / lookalike). Create a brand-new UUID.
            return DedupDecision(
                uuid=None,
                name=requested_name,
                match_name=best_name,
                match_uuid=best_uuid,
                match_sim=best_sim,
            )
        # Different name without force — reject and surface who we think
        # this actually is.
        return DedupDecision(
            reject=True,
            match_name=best_name,
            match_uuid=best_uuid,
            match_sim=best_sim,
        )

    # Soft match to same name → merge
    if best_sim >= merge_thr and same_name:
        # Same logic as the strong-match same_name branch: use requested
        # name when claiming an anon UUID, else preserve existing name.
        effective_name = requested_name if is_anon_match else best_name
        return DedupDecision(
            uuid=best_uuid,
            name=effective_name,
            merged=True,
            match_name=best_name,
            match_uuid=best_uuid,
            match_sim=best_sim,
        )

    # No qualifying match → create new UUID. Note: requested_name might
    # already exist as a different UUID's name; that's fine, names are
    # not unique in the UUID gallery.
    return DedupDecision(
        uuid=None,
        name=requested_name,
        match_name=best_name,
        match_uuid=best_uuid,
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
