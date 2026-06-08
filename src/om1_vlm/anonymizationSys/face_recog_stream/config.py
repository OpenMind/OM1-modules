from __future__ import annotations

from typing import Any, Dict

# ============================================================================
# 1. Face detection (SCRFD)
# ============================================================================

SCRFD_SIZE = 640
# SCRFD input resolution (square). 640 standard; 480 faster but worse on
# small faces.
# - Used in: scrfd.py TRTSCRFD(size=...)
# - cfg key: none (engine compile-time)
# - Hot-tunable: NO

DETECTION_CONFIDENCE = 0.5
# Minimum face detection confidence. Below this, the detection is dropped
# before NMS. Lower → more recall (find more faces including profiles); higher
# → more precision (only frontal/clear faces).
# - Used in: scrfd.detect(conf=...), face_tracker BoTSORT init, gallery/add_raw
# - cfg key: "conf"
# - Hot-tunable: YES via /config
# - Example: at 0.5, you reliably catch frontal faces ~3m away. At 0.3 you
#   also catch partial profiles but get more false-positive box wobbles.

DETECTION_NMS = 0.4
# NMS IoU threshold. Two boxes with IoU above this are deduped.
# Higher → more aggressive dedup (must overlap a lot to merge); lower → more
# aggressive merging (closer faces get merged into one box).
# - Used in: scrfd.nms_thresh (pushed live on /config update)
# - cfg key: "nms"
# - Hot-tunable: YES

DETECTION_MAX_NUM = 0
# Hard cap on faces returned per frame. 0 = no limit. Useful when crowds
# spike and you want to bound the work.
# - Used in: scrfd.detect(max_num=...)
# - cfg key: "max_num"
# - Hot-tunable: YES


# ============================================================================
# 2. Recognition (AdaFace + tier-aware decision)
# ============================================================================

SIM_THR = 0.55
# Cosine sim threshold below which we treat a match as 'unknown' regardless
# of margin. Top-1 floor.
# - Used in: selfie_logic.recognize_from_sims(min_sim=...),
#   face_tracker.FaceTracker(sim_thr=...)
# - cfg key: "sim_thr"
# - Hot-tunable: YES
# - Calibration: collect ~50 (same-person, different-frame) pairs and ~500
#   (different-person) pairs. Plot cosine sim distributions. Pick sim_thr at
#   the elbow of the cross-person distribution — typically 0.40-0.55 for
#   AdaFace IR101. Higher → more rejections in bad lighting; lower → more
#   false accepts.
# - Example: at 0.55, wendy in office light → match (sim=0.65). wendy in dim
#   hallway → might fall to 0.50 and become 'unknown'. Lower to 0.45 if this
#   happens too often.

STRICT_MARGIN = 0.08
# top1 - top2 margin required for the 'confident' recognition tier. When
# the top-1 candidate's similarity exceeds top-2's by at least this much, we
# trust the match without verification.
# - Used in: selfie_logic.recognize_from_sims(strict_margin=...),
#   face_tracker.FaceTracker(strict_margin=...)
# - cfg key: "strict_margin"
# - Hot-tunable: YES
# - Calibration: with a small gallery (<50 ids), this can be tight (0.06-0.08).
#   With a large gallery (>200 ids), near-collisions are more common, so
#   loosen to 0.10-0.12 to avoid false confidence.
# - Example: top1=0.62 (wendy), top2=0.51 (alice) → margin=0.11 → confident.
#            top1=0.62 (wendy), top2=0.58 (alice) → margin=0.04 → not confident.

LOOSE_MARGIN = 0.04
# top1 - top2 margin required for the 'tentative' recognition tier. Between
# strict_margin and loose_margin, we return the candidate but flag it as
# tentative — LLM should verify ('are you wendy?') instead of greeting by name.
# - Used in: same as strict_margin
# - cfg key: "loose_margin"
# - Hot-tunable: YES
# - Constraint: must be < strict_margin.
# - Example: top1=0.62, top2=0.57 → margin=0.05 → tentative.
#            top1=0.62, top2=0.59 → margin=0.03 → uncertain (don't commit).


# ============================================================================
# 3. Face tracking + voting (face_tracker.FaceTracker)
# ============================================================================

RECOG_INTERVAL_SEC = 0.2
# Seconds between recognition attempts per track. Recognition is the
# expensive operation (AdaFace TRT inference); we don't need to re-recognize
# every frame for the same track. Lower → faster identification of new
# tracks; higher → lower GPU load.
# - Used in: face_tracker.FaceTracker(recog_interval=...)
# - Hot-tunable: NO (sticky)
# - Example: at 0.2s and 15fps, every 3rd frame we run recognition for a
#   given track. A new track gets 3 votes in ~0.6s → identified.

VOTE_FRAMES = 3
# Number of recognition votes to collect before deciding identity. More
# votes = more robust against single-frame errors; fewer = faster decisions.
# - Used in: face_tracker.FaceTracker(vote_frames=...) + _finalize_identity
# - Hot-tunable: NO
# - Example: 3 votes with vote_threshold=0.5 means 2 of 3 must agree.

VOTE_THRESHOLD = 0.5
# Fraction of votes a single name needs to win. 0.5 = simple majority.
# - Used in: face_tracker.FaceTracker(vote_threshold=...) + _finalize_identity
# - Hot-tunable: NO
# - Example: vote_frames=3, vote_threshold=0.5 → 2/3 wins.
#            vote_frames=3, vote_threshold=0.7 → all 3 must agree.

MAX_RECOG_ATTEMPTS = 10
# Hard cap on recognition attempts per track before forcing a decision.
# Prevents indefinite 'identifying...' state on tracks that never get a
# clear majority.
# - Used in: face_tracker.FaceTracker(max_recog_attempts=...)
# - Hot-tunable: NO

TRACK_BUFFER = 30
# BoTSORT frames to keep a 'lost' track alive (waiting for re-detection).
# Higher = better at handling brief occlusions but increases the chance of
# track-id mixups when two people swap positions.
# - Used in: face_tracker.FaceTracker(track_buffer=...)
# - Hot-tunable: NO
# - Frame-count, scales with FPS: 30 frames @ 15fps = 2 seconds of occlusion
#   tolerance. If you change CAMERA_FPS, scale this accordingly to maintain
#   the same wall-clock duration.
# - Example: at 15fps, a face occluded for <2s gets the same track_id back
#   via BoTSORT's Kalman prediction. After 2s, occlusion timeout fires and
#   a new track_id is assigned on re-detection.

ARC_MAX_BS = 8
# AdaFace TRT max batch size. Capped by the engine's optimization profile.
# - Used in: face_tracker.FaceTracker(arc_max_bs=...)
# - Hot-tunable: NO (engine compile-time)

RE_IDENTIFY_INTERVAL_SEC = 1.0
# Seconds before retrying recognition on tracks that came back 'unknown'
# or 'tentative'. Handles the 'walking closer to camera' case — face that
# was blurry at distance becomes clear, give it another shot.
# - Used in: face_tracker.FaceTracker(re_identify_interval=...)
# - Hot-tunable: NO
# - Example: an unknown track tries again every 1 second. AutoEnroller may
#   have committed a UUID by then, in which case re-recognition succeeds.
# - Constraint check: must be < AUTO_ENROLL_STALE_TRACK_SEC (5.0) so the
#   AutoEnroller buffer for a track in the unknown→re-identify gap doesn't
#   get GC'd before observations resume. 1.0 < 5.0, safe.


# ============================================================================
# 4. Auto-enrollment (auto_enroller.AutoEnroller)
# ============================================================================

AUTO_ENROLL_MIN_SAMPLES = 3
# How many quality samples must accumulate before auto-enroll commits a
# new UUID. Lower → faster enrollment but more risk of misenrollment from
# BoTSORT track-id swaps.
# - Used in: auto_enroller.AutoEnroller(n_samples_required=...)
# - CLI: --auto-enroll-min-samples
# - Hot-tunable: NO (constructor-time)
# - Example: 3 with novelty_thr=0.97 and min_unknown_sec=1.0 means at least
#   3 distinct-enough samples within at least 1 second of observation.

AUTO_ENROLL_MAX_BUFFER = 5
# Maximum samples kept per track in the rolling buffer. When full, oldest
# is dropped. Should be ≥ n_samples_required.
# - Used in: auto_enroller.AutoEnroller(max_buffer=...)
# - CLI: --auto-enroll-max-buffer
# - Hot-tunable: NO

AUTO_ENROLL_TIGHTNESS = 0.55
# Required pairwise cosine similarity (worst pair) across the sample
# buffer before enrollment fires. Filters out tracks where BoTSORT confused
# two visitors during occlusion.
# - Used in: auto_enroller.AutoEnroller(min_pairwise_sim=...)
# - CLI: --auto-enroll-tightness
# - Hot-tunable: NO
# - Calibration: should be lower than SIM_THR (0.55) since within-person sim
#   varies with lighting/angle. 0.55 is a conservative start; tune down to
#   ~0.45 if too few auto-enrollments fire, up to 0.60 if you see polluted
#   UUIDs (samples of two people enrolled together).

AUTO_ENROLL_MIN_UNKNOWN_SEC = 1.0
# Track must have been visible/unknown for this many seconds before
# auto-enroll considers it. Filters out walk-throughs that produce a few
# quick samples but aren't really "engaged with the robot".

# Why 1.0s
# --------
# At 1.5s the robot feels sluggish — a stranger walks up, stops, and waits
# 1.5+s before being enrolled. At 1.0s the enrollment happens just as the
# robot finishes saying "hi" — feels natural.

# How this composes with other gates
# ----------------------------------
# Auto-enroll commits when ALL of these are true:
#   - sample_count ≥ n_required (3)
#   - time elapsed ≥ min_unknown_sec
#   - pairwise tightness ≥ min_pairwise_sim
#   - each sample passes novelty + consistency checks

# Other safeguards already filter walk-throughs:
#   - min_face_pixels=30 rejects distant pedestrians
#   - min_pairwise_sim=0.55 rejects samples where the face is moving/changing
#     too fast (walk-throughs)
#   - novelty_thr requires within-person variation (a single passing glance
#     rarely produces 3 distinct embeddings)

# So min_unknown_sec is one of multiple independent filters; it doesn't
# need to be conservative on its own.

# - Used in: auto_enroller.AutoEnroller(min_unknown_sec=...)
# - CLI: --auto-enroll-min-unknown-sec
# - Hot-tunable: NO
# - Calibration: at demo time, watch enrollment latency. If too many
#   walk-throughs enroll: raise to 1.5s. If robot feels slow: lower
#   to 0.7-0.8s.

AUTO_ENROLL_MIN_FACE_PX = 26
# Minimum bbox short side for an auto-enroll sample. Below this, samples
# are dropped — too small to produce reliable embeddings.

# Sized for 640×480 camera (Unitree G1 default). 30 px short side ≈ 6.25%
# of frame height, equivalent to a person ~2-3m from the camera at ~60° field of view.

# At higher camera resolutions, scale up to maintain the same physical
# distance threshold:
#   - 640×480  → 30 px
#   - 1280×720 → 50 px
#   - 1920×1080 → 75 px

# - Used in: auto_enroller.AutoEnroller(min_face_pixels=...)
# - CLI: --auto-enroll-min-face-px
# - Hot-tunable: NO
# - Note: this is slightly looser than the per-frame recognition gate would
#   be — auto-enroll wants multiple samples anyway, so a single marginal
#   one isn't fatal. The pairwise-tightness check filters out noisy
#   embeddings before commit.


AUTO_ENROLL_STALE_TRACK_SEC = 5.0
# Drop a track's buffer if no observe() in this long. Saves memory on
# tracks that walked out of frame.
# - Used in: auto_enroller.AutoEnroller(stale_track_sec=...)
# - Hot-tunable: NO
# - Constraint: must be > RE_IDENTIFY_INTERVAL_SEC (1.0) so buffers
#   survive the gap between finalize→unknown and re-identify resuming
#   observations. 5.0 > 1.0, safe.

# Dedicated auto-enroll quality gates (independent of the /selfie gates).
# A face must clear BOTH before auto-enroll will buffer/commit it, so only
# clear, frontal faces enter the gallery. These do NOT affect detection/
# tracking/recognition (the global DETECTION_CONFIDENCE stays low so the
# robot still SEES profiles/far faces) — they only gate enrollment.
AUTO_ENROLL_MIN_CONF = 0.65  # min detector confidence to auto-enroll
AUTO_ENROLL_MIN_FRONTALITY = 0.6  # min frontality (0=off .. 1=dead-on) to auto-enroll

AUTO_ENROLL_MERGE_THR = 0.50
# Cosine sim threshold for merging an auto-enroll buffer into an existing
# UUID instead of creating a new one. Solves the 'known person in bad lighting'
# problem.

# The problem
# -----------
# SIM_THR=0.55 is the per-frame floor for recognition: below it, a face is
# labeled 'uncertain' even if it's a known person. So wendy in dim hallway
# (top1_sim ≈ 0.52 per frame) gets buffered by AutoEnroller. Without this
# threshold, AutoEnroller would create a SECOND UUID for wendy every time
# she walks into different lighting.

# The fix
# -------
# Before committing a new UUID, AutoEnroller computes the MEAN of all
# buffered embeddings (more reliable than any single frame — noise averages
# out, identity signal stays). If that mean matches any existing UUID's
# centroid with sim >= AUTO_ENROLL_MERGE_THR, append the samples to that
# UUID instead.

# - Used in: auto_enroller.AutoEnroller(merge_thr=...)
# - CLI: --auto-enroll-merge-thr
# - Hot-tunable: NO
# - Calibration: should be LOWER than SIM_THR (so the merge path can rescue
#   per-frame 'uncertain' decisions) but HIGHER than random-pair sim (~0.0-0.3
#   for cross-person on AdaFace) so true strangers still get new UUIDs.
#   Setting to 0.0 disables the merge path (always creates new UUIDs).
# - Example: wendy buffered with mean sim 0.62 to her existing centroid →
#   0.62 >= 0.50 → merge. Stranger buffered with mean sim 0.18 to nearest
#   centroid → 0.18 < 0.50 → new UUID.


# ============================================================================
# 4b. Continuous sample refresh (sample_refresh.SampleRefreshManager)
# ============================================================================
#
# Once a UUID exists in the gallery, every confident per-frame recognition
# is a potential opportunity to enrich its centroid with a new feature
# (different angle, lighting, expression). Three guards keep this safe:
#
#   A. Match confidence — sim must exceed sim_thr + REFRESH_MIN_EXTRA_CONFIDENCE
#   B. Per-UUID rate limit — one attempt per REFRESH_MIN_INTERVAL_SEC
#   C. Diversity — new sample's sim to centroid in
#      [REFRESH_MIN_DIVERSITY_SIM, REFRESH_MAX_DIVERSITY_SIM]
#
# The gallery itself enforces a hard cap of MAX_SAMPLES_PER_UUID = 50 with
# FIFO eviction, so disk stays bounded even with permissive guard settings.

REFRESH_MIN_EXTRA_CONFIDENCE = 0.10
# Guard A: minimum margin over sim_thr before a confident match qualifies
# for sample refresh. Effective threshold = sim_thr + this value.
# - Default: sim_thr=0.55, so refresh threshold = 0.65
# - Higher → conservative (only enrich on rock-solid matches)
# - Lower  → aggressive (more samples folded in, more risk of misattribution)
# - Used in: SampleRefreshManager(min_extra_confidence=...)
# - Hot-tunable: NO (would need to rebuild the manager)

REFRESH_MIN_INTERVAL_SEC = 60.0
# Guard B: minimum seconds between refresh attempts per UUID. Prevents a
# single session from flooding a UUID with near-identical samples.
# - Default: 60s. A 2-3 min conversation captures 2-3 samples, enough to
#   pick up some pose/lighting variation without dominating disk I/O.
# - Combined with the 50-sample FIFO cap, the centroid stays diverse over
#   long deployments. Bumping this to 600s (10 min) is also reasonable for
#   very long-running deployments where intra-session diversity matters
#   less than cross-session diversity.
# - Used in: SampleRefreshManager(min_refresh_interval_sec=...)

REFRESH_MIN_DIVERSITY_SIM = 0.65
# Guard C (lower bound): minimum sim between new sample's embedding and
# the UUID's centroid for refresh to accept. Below this, treat as
# likely-misidentification and skip.
# - Default: 0.65 (matches sim_thr + min_extra_confidence by design)
# - Higher → only very-similar samples get in (less drift risk)
# - Used in: gallery.refresh_sample(min_diversity_sim=...)

REFRESH_MAX_DIVERSITY_SIM = 0.92
# Guard C (upper bound): maximum sim between new sample's embedding and
# centroid. Above this, the sample is essentially a duplicate and adds no
# feature diversity — skip to keep ring-buffer slots for novel samples.
# - Default: 0.92 (allows minor pose/lighting variation, rejects near-clones)
# - Lower → only sufficiently-different angles get in (more diverse centroid)
# - Used in: gallery.refresh_sample(max_diversity_sim=...)


# ============================================================================
# 4c. Gallery sample cap (gallery.UUIDGallery)
# ============================================================================

MAX_SAMPLES_PER_UUID = 50
# Hard cap on samples stored per UUID. When add_sample / refresh_sample
# would push past this, the oldest sample is evicted (FIFO):
# JPG removed, embeddings.npy row 0 dropped, centroid_sum decremented.

# Disk math: 50 samples × ~50KB per aligned 112x112 JPG = 2.5MB per UUID.
# 1000 UUIDs = 2.5GB. Effective ceiling for long-running deployments.

# - Used in: UUIDGallery(max_samples_per_uuid=...)
# - Hot-tunable: NO (would need cleanup pass to shrink existing UUIDs)

LAST_SEEN_DISK_PERSIST_INTERVAL_SEC = 60.0
# How often (in seconds) ``touch_last_seen`` flushes ``last_seen_at`` to
# disk per UUID. In-memory ``rec.metadata["last_seen_at"]`` is updated on
# every call (cheap); disk write is rate-limited.

# Without this rate limit, touch_last_seen would write metadata.json
# ~15×N times per second (N = visible faces), making it the dominant
# source of disk I/O. With 60s persist, the same load is ~1 write per
# minute per UUID — three orders of magnitude less churn.

# The on-disk last_seen_at may lag the true last sighting by up to this
# many seconds across a crash, which is fine for the ``/gallery/sweep_stale``
# use case (cleanup thresholds are typically days or months).

# - Used in: UUIDGallery(last_seen_persist_interval_sec=...)
# - Hot-tunable: NO


# ============================================================================
# 5. Selfie enrollment (http_api._do_selfie + selfie_logic)
# ============================================================================

SELFIE_WINDOW_SEC = 1.2
# Maximum time the selfie loop waits for samples. After this, returns
# whatever it's collected (or error if < SELFIE_MIN_SAMPLES).
# - cfg key: "selfie_window_sec"
# - Hot-tunable: YES
# - Sized for 15fps camera: 1.2s × 15fps = 18 frames in the window. After
#   quality gates (frontality, sharpness, brightness) filter ~40% out,
#   we still net ~10 usable frames — well above SELFIE_MIN_SAMPLES=1 and
#   enough room for SELFIE_MAX_SAMPLES=4. At 30fps you could drop to 0.8s.

SELFIE_TAP_INTERVAL_SEC = 0.07
# Sleep between selfie loop iterations to let new frames arrive. Should
# be slightly below the camera frame interval (1/fps) so we wake up just
# before each new frame is ready, without wasting CPU on busy-wait.
# - cfg key: "selfie_tap_interval_sec"
# - Hot-tunable: YES
# - For 15fps camera (frame interval = 66ms): 70ms is right.
#   For 30fps (frame interval = 33ms): 30-35ms would be the equivalent.
# - Caveat: the selfie loop ALSO uses a frame_token check to detect fresh
#   frames, so a slightly-too-short interval here doesn't double-process
#   the same frame — just wastes a tiny bit of CPU.

SELFIE_MIN_SAMPLES = 1
# Adaptive sample count: minimum number of valid frames to enroll. min=1
# lets static users enroll with just the first valid frame.
# - cfg key: "selfie_min_samples"
# - Hot-tunable: YES

SELFIE_MAX_SAMPLES = 4
# Maximum samples per selfie. Caps storage/embedding work per call.
# - cfg key: "selfie_max_samples"
# - Hot-tunable: YES

SELFIE_MIN_ENGAGEMENT = 0.0025
# Engagement score floor below which a face isn't considered a selfie
# target. score = (bbox_area / frame_area) × frontality, in [0, 1].

# Calibrated for 640×480 camera + SELFIE_MIN_FACE_PX=30
# ---------------------------------------------------------
# The minimum sensible engagement is the score of a barely-passing face:
# 30 px frontal face → score = (30×30) / (640×480) × 1.0 = 0.00293.

# 0.0025 sits just below this with a small margin to absorb detection
# box wobble (1-2 px jitter can shrink bbox area by ~10%).

# Behavior at this threshold:
#   - 30 px frontal face         → 0.00293 ✓ pass
#   - 30 px ¾-profile (f=0.85)   → 0.00249 ✗ borderline
#   - 25 px frontal face         → 0.00203 ✗ too small
#   - 40 px slight side (f=0.5)  → 0.00260 ✓ pass
#   - 70 px frontal at 1m        → 0.0159  ✓ pass (typical close selfie)

# If you change CAMERA_WIDTH/HEIGHT or SELFIE_MIN_FACE_PX, recompute:
#     min_engagement = (min_face_px)² / (W × H) × 0.85
# where the 0.85 factor leaves margin for box jitter + frontality < 1.

# - cfg key: "selfie_min_engagement"
# - Hot-tunable: YES
# - Multi-face role: engagement score is also used to PICK the best face
#   when multiple are detected (argmax). The floor here additionally
#   enables the ambiguity check (skips frames where top-1 and top-2 are
#   both above floor and close in score).

SELFIE_AMBIGUITY_RATIO = 0.80
# top2/top1 engagement ratio above which we treat the frame as 'ambiguous'
# (two people equally addressing the robot). Frame skipped.
# - cfg key: "selfie_ambiguity_ratio"
# - Hot-tunable: YES
# - Example: 0.80 means if two faces have engagement scores 0.05 and 0.045,
#   ratio = 0.9 > 0.80 → ambiguous, skip frame.

SELFIE_MIN_FACE_PX = 30
# Selfie per-frame quality gate: bbox short side in pixels.

# Sized for 640×480 camera (Unitree G1 default). 30 px short side ≈ 6.25%
# of frame height, equivalent to a person ~2-3m from the camera at ~60° field of view.

# At higher camera resolutions (e.g. 1280×720) you can raise this to keep
# the same physical distance threshold:
#   - 640×480  → 30 px
#   - 1280×720 → 50 px
#   - 1920×1080 → 75 px

# - cfg key: "selfie_min_face_px"
# - Hot-tunable: YES

SELFIE_MIN_CONF = 0.65
# Selfie per-frame quality gate: detector confidence.
# - cfg key: "selfie_min_conf"
# - Hot-tunable: YES

SELFIE_MIN_FRONTALITY = 0.4
# Selfie per-frame quality gate: frontality from 5-point landmarks, [0,1].
# - cfg key: "selfie_min_frontality"
# - Hot-tunable: YES

SELFIE_SHARP_THR = 5.0
# Selfie per-frame quality gate: Laplacian variance threshold for blur
# detection. Low value = blurry image → reject.
# - cfg key: "selfie_sharp_thr"
# - Hot-tunable: YES

SELFIE_BRIGHTNESS_MIN = 30
# Selfie per-frame quality gate: minimum grayscale mean.
# - cfg key: "selfie_brightness_min"
# - Hot-tunable: YES

SELFIE_BRIGHTNESS_MAX = 230
# Selfie per-frame quality gate: maximum grayscale mean.
# - cfg key: "selfie_brightness_max"
# - Hot-tunable: YES

SELFIE_NOVELTY_THR = 0.93
# Cosine sim threshold above which a new selfie frame is 'too similar to
# what we already have' → skip (novelty check).
# - cfg key: "selfie_novelty_thr"
# - Hot-tunable: YES
# - Example: 0.93 means new frame must have at least ~7% angular variation
#   from the running mean.

SELFIE_CONSISTENCY_THR = 0.50
# Cosine sim threshold below which a selfie frame is 'a different person
# from the one we started with' → skip (consistency check).
# - cfg key: "selfie_consistency_thr"
# - Hot-tunable: YES

SELFIE_MERGE_THR = 0.5
# Cosine sim threshold for merging a selfie into an existing same-name UUID.
# - Used in: selfie_logic.resolve_target_id
# - cfg key: "selfie_merge_thr"
# - Hot-tunable: YES
# - Example: 0.45 means if a new wendy selfie matches an existing wendy UUID
#   at sim >= 0.45, append samples instead of creating a new UUID.

SELFIE_CROSS_NAME_THR = 0.60
# Cosine sim threshold above which a cross-name match triggers rejection
# (without force=True). Prevents 'Bob accidentally enrolled as Wendy'.
# - Used in: selfie_logic.resolve_target_id
# - cfg key: "selfie_cross_name_thr"
# - Hot-tunable: YES
# - Example: 0.60 means if face matches an existing UUID at sim >= 0.60 but
#   the names differ → reject and return 'face_belongs_to'.


# ============================================================================
# 6. Pixelation / privacy blur
# ============================================================================

BLUR_ENABLED = False
# Whether to apply privacy pixelation at all.
# - cfg key: "blur"
# - Hot-tunable: YES

BLUR_MODE = "all"
# Which faces to blur. Options: "all", "known", "unknown".
# - cfg key: "blur_mode"
# - Hot-tunable: YES
# - Example: "unknown" blurs only unrecognized faces (useful when you want
#   to show only enrolled people clearly).

PIXEL_BLOCKS = 8
# Pixelation granularity: number of pixel blocks across the short side of
# the bbox. Smaller = coarser pixelation; larger = finer.
# - cfg key: "pixel_blocks"
# - Hot-tunable: YES

PIXEL_MARGIN = 0.25
# Bbox expansion (fraction of size) before pixelation. Helps hide hair/ears
# that sit outside the detection box.
# - cfg key: "pixel_margin"
# - Hot-tunable: YES

PIXEL_NOISE = 0.0
# Gaussian noise sigma added to pixelated area. 0 = off. Helps prevent
# pixel-perfect un-pixelation attacks.
# - cfg key: "pixel_noise"
# - Hot-tunable: YES


# ============================================================================
# 7. Pose detection + fall detection
# ============================================================================

POSE_ENABLED = False
# Whether to run pose detection (YOLO11s-pose). Disabled by default for
# cost. Enable with --pose.

POSE_CONF = 0.5
# YOLO pose detector confidence threshold.
# - cfg key: "pose_conf"
# - Hot-tunable: YES

POSE_MAX_NUM = 10
# Maximum persons per frame for pose detection.
# - cfg key: "pose_max_num"
# - Hot-tunable: YES

FALL_DETECTION_ENABLED = False
# Whether to run fall detection. Requires POSE_ENABLED.
# - cfg key: "fall_detection"
# - Hot-tunable: YES

# Drawing options — all hot-tunable via /config
DRAW_BOXES = False
DRAW_NAMES = False
DRAW_SKELETON = False
DRAW_POSE_BOXES = False
DRAW_FALL_STATUS = False
SHOW_FPS = False


# ============================================================================
# 8. Recognition strategy (main loop crowd handling)
# ============================================================================

RECOG_TOPK = 4
# Maximum faces per frame to send to AdaFace. Caps per-frame GPU work in
# crowds. Should be ≤ AdaFace engine's max batch size.
# - CLI: --recog-topk
# - Hot-tunable: NO

CROWD_THR = 12
# Hard crowd cap: if detected faces exceed this, skip recognition entirely
# for this frame (still draw boxes and blur). Prevents GPU starvation when
# a crowd shows up.
# - cfg key: "crowd_thr"
# - Hot-tunable: YES


# ============================================================================
# 9. Camera / RTSP / HTTP
# ============================================================================

CAMERA_DEVICE = "/dev/video0"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# Camera resolution. Unitree G1's onboard camera default is 640×480.

# All bbox-size thresholds (SELFIE_MIN_FACE_PX=30, AUTO_ENROLL_MIN_FACE_PX=30)
# are sized for this resolution. If you bump to a higher resolution camera,
# scale those thresholds proportionally:
#   640×480  → 30 px (current)
#   1280×720 → 50 px
#   1920×1080 → 75 px
# to keep the same effective "person distance" threshold.
CAMERA_FPS = 15
# Capture frame rate. The Unitree G1 onboard camera runs at 15 fps.

# All wall-clock-based parameters (recog_interval, min_unknown_sec,
# re_identify_interval, stale_track_sec) are independent of this — they
# measure seconds, not frames.

# Frame-count parameters that DO scale with fps:
#   - TRACK_BUFFER (currently 30 = 2s @ 15fps)
#   - gc_stale modulo in run.py main loop (15 = ~1s @ 15fps)

LOCAL_RTSP_URL = "rtsp://localhost:8554/top_camera"
RAW_LOCAL_RTSP_URL = "rtsp://localhost:8554/top_camera_raw"

HTTP_HOST = "127.0.0.1"
HTTP_PORT = 6793
HTTP_LOOKBACK_SEC = 10.0
# Default lookback window for /who queries (seconds).


# ============================================================================
# DEFAULTS dict — seeds the runtime cfg dict in run.py
# ============================================================================
#
# Only keys present here are hot-tunable via /config. Adding a new key here
# (plus a setter on the relevant component) makes a param /config-tunable.
# Removing an entry locks that param to its startup value.

DEFAULTS: Dict[str, Any] = {
    # Detection
    "conf": DETECTION_CONFIDENCE,
    "nms": DETECTION_NMS,
    "max_num": DETECTION_MAX_NUM,
    # Recognition tier
    "sim_thr": SIM_THR,
    "strict_margin": STRICT_MARGIN,
    "loose_margin": LOOSE_MARGIN,
    # Blur
    "blur": BLUR_ENABLED,
    "blur_mode": BLUR_MODE,
    "pixel_blocks": PIXEL_BLOCKS,
    "pixel_margin": PIXEL_MARGIN,
    "pixel_noise": PIXEL_NOISE,
    # Pose
    "pose": POSE_ENABLED,
    "pose_conf": POSE_CONF,
    "pose_max_num": POSE_MAX_NUM,
    "fall_detection": FALL_DETECTION_ENABLED,
    # Crowd
    "crowd_thr": CROWD_THR,
    # Selfie pipeline (all hot-tunable via /config)
    "selfie_window_sec": SELFIE_WINDOW_SEC,
    "selfie_tap_interval_sec": SELFIE_TAP_INTERVAL_SEC,
    "selfie_min_samples": SELFIE_MIN_SAMPLES,
    "selfie_max_samples": SELFIE_MAX_SAMPLES,
    "selfie_min_engagement": SELFIE_MIN_ENGAGEMENT,
    "selfie_ambiguity_ratio": SELFIE_AMBIGUITY_RATIO,
    "selfie_min_face_px": SELFIE_MIN_FACE_PX,
    "selfie_min_conf": SELFIE_MIN_CONF,
    "selfie_min_frontality": SELFIE_MIN_FRONTALITY,
    "selfie_sharp_thr": SELFIE_SHARP_THR,
    "selfie_brightness_min": SELFIE_BRIGHTNESS_MIN,
    "selfie_brightness_max": SELFIE_BRIGHTNESS_MAX,
    "selfie_novelty_thr": SELFIE_NOVELTY_THR,
    "selfie_consistency_thr": SELFIE_CONSISTENCY_THR,
    "selfie_merge_thr": SELFIE_MERGE_THR,
    "selfie_cross_name_thr": SELFIE_CROSS_NAME_THR,
    # Drawing
    "show_fps": SHOW_FPS,
    "draw_skeleton": DRAW_SKELETON,
    "draw_pose_boxes": DRAW_POSE_BOXES,
    "draw_fall_status": DRAW_FALL_STATUS,
}


# ============================================================================
# Hot-tunability quick reference
# ============================================================================
#
# YES — via POST /config {"set": {...}} while system is running:
#   conf, nms, max_num
#   sim_thr, strict_margin, loose_margin
#   blur, blur_mode, pixel_blocks, pixel_margin, pixel_noise
#   pose, pose_conf, pose_max_num, fall_detection
#   crowd_thr
#   selfie_* (all selfie pipeline knobs)
#   show_fps, draw_*
#
# NO — restart required (controllable via CLI flags at startup):
#   AUTO_ENROLL_* (auto_enroller constructor-time)
#   RECOG_INTERVAL_SEC, VOTE_FRAMES, VOTE_THRESHOLD, MAX_RECOG_ATTEMPTS,
#     TRACK_BUFFER, ARC_MAX_BS, RE_IDENTIFY_INTERVAL_SEC
#     (face_tracker constructor-time)
#   SCRFD_SIZE, AdaFace engine path (engine compile-time)
#   Camera / RTSP / HTTP bindings
#
# To make a constructor-time param hot-tunable:
#   1. Add the key to DEFAULTS above
#   2. Add a setter on the relevant component (like FaceTracker.set_thresholds)
#   3. Push the value from the main loop's per-frame cfg snapshot
