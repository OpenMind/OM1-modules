"""
Vision speaking detector.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from . import head_pose

log = logging.getLogger(__name__)


# landmark model I/O (don't change unless swapping models)
LM_SIZE = 192  # 2d106det input size
LM_SWAP_RB = True  # BGR -> RGB
LM_SCALE = 1.0  # pixel scale (2d106det uses raw [0,255])
LM_MEAN = 0.0  # mean subtracted
N_LANDMARKS = 106  # 2d106det point count
MOUTH_IDX = list(range(52, 72))  # mouth point block (2d106det only)

# FaceMesh (468) backend: auto-enabled when model outputs 1404
FACEMESH_N = 468  # point count
FACEMESH_SIZE = 192  # input size
FACEMESH_DIV255 = True  # normalize to [0,1]; set False if points look wrong
MP_UP_INNER, MP_LO_INNER = 13, 14  # upper/lower inner-lip centers (opening)
MP_L_CORNER, MP_R_CORNER = 61, 291  # mouth corners (width)

# signal / windowing
WINDOW_SEC = 1.6  # decision window (s); LOWER = snappier but jumpier/more false fires
BUFFER_SEC = 6.0  # history kept (s); how far back you can look; leave as-is
SLIDE_STEP_SEC = 0.5  # sliding step for long utterances; leave as-is
WINDOW_PAD_SEC = 0.4  # query-span padding; leave as-is
MIN_SAMPLES = 6  # min frames before scoring; LOWER = scores on shorter windows

# sensitivity / threshold
MAR_STD_FULL = (
    0.06  # mouth motion that maps to score 1.0; LOWER = higher scores (more sensitive)
)
MIN_CROSS_HZ = 1.0  # min open/close rhythm (crossings/s); LOWER = looser

# mouth-shape motion gates (reject head turn / profile)
TURN_WIDTH_CV = (
    0.06  # mouth-width wobble above this = turning -> suppress; RAISE = looser
)
PROFILE_WIDTH_FRAC = (
    0.90  # width < this x frontal width = profile -> suppress; LOWER = looser
)
GATE_SUPPRESS = 0.6  # how hard a tripped gate cuts the score; RAISE = looser (0.4 = dampen, not kill) ★MASTER

# head-pose gates (independent of the mouth)
FRONT_MIN = 0.45  # frontality below this = side/down -> suppress; LOWER = looser (allow more tilt)
YAW_TURN_STD = (
    8.0  # head yaw swing (deg) above this = turning -> suppress; RAISE = looser
)

# body-sway gate
SWAY_MAX = 0.25  # face-center drift (in face-widths) above this = swaying -> suppress; RAISE = looser


class VVADScorer:
    """Landmark-MAR speaking scorer (interface-compatible with the CNN one)."""

    def __init__(
        self,
        engine_path: str,
        speaking_thr: float = 0.5,
        window_sec: float = WINDOW_SEC,
        buffer_sec: float = BUFFER_SEC,
        min_frames: int = 0,
        use_cuda: bool = False,
        intra_op_threads: int = 2,
    ):
        if ort is None:
            raise RuntimeError("onnxruntime not installed")
        self.speaking_thr = float(speaking_thr)
        self.window_sec = float(window_sec)
        self.buffer_sec = max(float(buffer_sec), self.window_sec)
        # min_frames kept for signature compatibility; mapped to min samples.
        self.min_samples = int(min_frames) if min_frames > 0 else MIN_SAMPLES

        self._buf: Dict[int, deque] = {}  # track_id -> deque[(ts, mar, width)]
        self._lock = threading.Lock()
        # set VVAD_DEBUG=1 in the environment to log per-track score breakdowns
        self.debug = bool(os.environ.get("VVAD_DEBUG"))

        so = ort.SessionOptions()
        if intra_op_threads > 0:
            so.intra_op_num_threads = int(intra_op_threads)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_cuda
            else ["CPUExecutionProvider"]
        )
        self._sess = ort.InferenceSession(
            engine_path, sess_options=so, providers=providers
        )
        self._in_name = self._sess.get_inputs()[0].name
        # Auto-detect backend from the model's output size: 468*3=1404 -> FaceMesh.
        _osh = self._sess.get_outputs()[0].shape
        _onumel = 1
        for _d in _osh:
            if isinstance(_d, int) and _d > 0:
                _onumel *= _d
        self.backend = "facemesh" if _onumel == FACEMESH_N * 3 else "2d106det"
        # facemesh models come in NHWC ([1,192,192,3]) or NCHW ([1,3,192,192]).
        _ish = self._sess.get_inputs()[0].shape
        self._facemesh_nhwc = len(_ish) == 4 and _ish[-1] == 3
        log.info(
            "VVAD landmark backend = %s (output numel=%s%s)",
            self.backend,
            _onumel,
            ", NHWC" if (self.backend == "facemesh" and self._facemesh_nhwc) else "",
        )
        log.info(
            "VVAD(landmark-MAR) loaded: %s (input=%s, thr=%.2f, "
            "window=%.1fs, buffer=%.1fs, mouth_idx=%d..%d)",
            engine_path,
            self._in_name,
            self.speaking_thr,
            self.window_sec,
            self.buffer_sec,
            MOUTH_IDX[0],
            MOUTH_IDX[-1],
        )

    # landmark inference + MAR
    def _landmarks(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Run the landmark model on a face crop -> (N_LANDMARKS, 2) in a
        consistent [0,1]-ish space. Ratios below are invariant to the exact
        scaling, so we don't map back to image pixels.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 8 or w < 8:
            return None
        if getattr(self, "backend", "2d106det") == "facemesh":
            scale = (1.0 / 255.0) if FACEMESH_DIV255 else 1.0
            img = cv2.resize(crop_bgr, (FACEMESH_SIZE, FACEMESH_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) * scale
            if self._facemesh_nhwc:
                blob = img[None, ...]  # (1,192,192,3) NHWC
            else:
                blob = np.transpose(img, (2, 0, 1))[None]  # (1,3,192,192) NCHW
            out = self._sess.run(None, {self._in_name: blob})[0]
            pred = np.asarray(out, dtype=np.float32).reshape(-1, 3)
            if pred.shape[0] < FACEMESH_N:
                return None
            return pred[:FACEMESH_N, :2]  # (468,2); ratios are scale-invariant

        blob = cv2.dnn.blobFromImage(
            crop_bgr,
            LM_SCALE,
            (LM_SIZE, LM_SIZE),
            (LM_MEAN, LM_MEAN, LM_MEAN),
            swapRB=LM_SWAP_RB,
            crop=False,
        )
        out = self._sess.run(None, {self._in_name: blob})[0]
        pred = np.asarray(out, dtype=np.float32).reshape(-1, 2)
        if pred.shape[0] < N_LANDMARKS:
            return None
        # 2d106det returns coords in ~[-1,1]; map to [0,1] (scale-invariant use).
        return (pred[:N_LANDMARKS] + 1.0) * 0.5

    @staticmethod
    def _mouth_metrics(lms: np.ndarray) -> Optional[Tuple[float, float]]:
        """(MAR, mouth_width) from the mouth landmark block.

        MAR is rotation-invariant: corners are the horizontal extremes of the
        block; rotate the block so the corner line is horizontal, then take
        (vertical extent / corner distance). width = corner distance, which we
        also use to detect head turning (it shrinks on profile / while turning).
        Only MOUTH_IDX must be correct — corners are found dynamically.
        """
        try:
            if lms.shape[0] >= FACEMESH_N:
                # FaceMesh: true inner-lip opening (13-14) over corner width
                # (61-291). Rotation/scale/translation invariant (Euclidean
                # distances). Much more sensitive to small speech motion.
                up, lo = lms[MP_UP_INNER], lms[MP_LO_INNER]
                lc, rc = lms[MP_L_CORNER], lms[MP_R_CORNER]
                width = float(np.hypot(rc[0] - lc[0], rc[1] - lc[1]))
                if width < 1e-9:
                    return None
                vert = float(np.hypot(up[0] - lo[0], up[1] - lo[1]))
                return vert / width, width
            m = lms[MOUTH_IDX]
        except (IndexError, TypeError):
            return None
        if m.shape[0] < 4:
            return None
        li = int(np.argmin(m[:, 0]))  # left corner  = min-x point
        ri = int(np.argmax(m[:, 0]))  # right corner = max-x point
        pL, pR = m[li], m[ri]
        d = pR - pL
        width = float(np.hypot(d[0], d[1]))
        if width < 1e-6:
            return None
        ang = np.arctan2(d[1], d[0])  # rotate corner line to horizontal
        c, s = np.cos(-ang), np.sin(-ang)
        rot = np.empty_like(m)
        rot[:, 0] = m[:, 0] * c - m[:, 1] * s
        rot[:, 1] = m[:, 0] * s + m[:, 1] * c
        # Openness = gap between the upper-lip and lower-lip point groups
        # (mean of the top half vs bottom half of the rotated y's). Averaging
        # each lip is far less noisy than the outer top/bottom extremes, and it
        # tracks the inner opening that actually moves during speech.
        ys = np.sort(rot[:, 1])
        k = max(1, len(ys) // 2)
        height = float(ys[-k:].mean() - ys[:k].mean())
        return height / width, width

    @classmethod
    def _mar(cls, lms: np.ndarray) -> Optional[float]:
        """MAR only (kept for verify_landmarks.py)."""
        r = cls._mouth_metrics(lms)
        return None if r is None else r[0]

    # buffer
    def push(self, track_id: int, crop_bgr: np.ndarray, kps=None) -> None:
        """Frontality from the 5 keypoints FIRST (cheap). If too off-axis
        (< FRONT_MIN) the mouth is unreliable and scores quiet anyway, so skip
        the expensive landmark inference entirely and store nothing. Otherwise
        landmark the crop, compute MAR+width, and append the sample.
        """
        front, yaw = 1.0, 0.0  # defaults if we have no kps
        cx = cy = sc = 0.0  # face center + size (image px) for sway
        if kps is not None and head_pose is not None:
            try:
                ang = head_pose.head_pose_angles(kps)
                if ang is not None:
                    yaw = float(ang[0])
                    front = float(head_pose.frontality_from_angles(*ang))
            except Exception:
                pass

        if front < FRONT_MIN:  # too off-axis -> skip model, = quiet
            return

        if kps is not None:
            try:
                k = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
                if k.shape[0] >= 2:
                    cx = float(k[:, 0].mean())
                    cy = float(k[:, 1].mean())
                    sc = float(np.hypot(k[0, 0] - k[1, 0], k[0, 1] - k[1, 1]))
            except Exception:
                pass

        lms = self._landmarks(crop_bgr)  # expensive; only when frontal
        if lms is None:
            return
        mm = self._mouth_metrics(lms)
        if mm is None:
            return
        mar, width = mm
        now = time.time()
        with self._lock:
            dq = self._buf.get(track_id)
            if dq is None:
                dq = deque()
                self._buf[track_id] = dq
            dq.append(
                (now, float(mar), float(width), float(front), float(yaw), cx, cy, sc)
            )
            cutoff = now - self.buffer_sec
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def evict(self, alive_track_ids=None) -> None:
        """Drop buffered samples for tracks not in alive_track_ids (or all if None)."""
        with self._lock:
            if alive_track_ids is None:
                self._buf.clear()
                return
            alive = set(int(t) for t in alive_track_ids)
            for tid in [t for t in self._buf if t not in alive]:
                del self._buf[tid]

    def _samples_in(
        self, track_id: int, t0: float, t1: float
    ) -> List[Tuple[float, ...]]:
        with self._lock:
            dq = self._buf.get(track_id)
            if not dq:
                return []
            return [s for s in dq if t0 <= s[0] <= t1]

    def _recent(self, track_id: int) -> List[Tuple[float, ...]]:
        now = time.time()
        return self._samples_in(track_id, now - self.window_sec, now)

    def _ref_width(self, track_id: int) -> float:
        """This track's frontal mouth width = a high percentile of its buffered
        widths (robust to the odd bad-landmark frame). Used to detect profile.
        """
        with self._lock:
            dq = self._buf.get(track_id)
            if not dq:
                return 0.0
            ws = sorted(s[2] for s in dq)
        return float(ws[int(0.9 * (len(ws) - 1))]) if ws else 0.0

    # scoring
    def _score_samples(
        self,
        samples: List[Tuple[float, ...]],
        ref_width: float = 0.0,
        tid: Optional[int] = None,
    ) -> Optional[float]:
        """MAR time series -> speaking score in [0,1].

        amplitude = std of the detrended MAR (how much the mouth moves);
        rhythm    = baseline crossings per second (speech-like open/close);
        then head-motion rejection using mouth width:
          * turning (width CV high)   -> suppress
          * profile (width << ref)    -> suppress
        so a head turn no longer mimics speaking, and a still frontal talker
        keeps its score.
        """
        if len(samples) < self.min_samples:
            return None
        ts = np.array([s[0] for s in samples], dtype=np.float64)
        v = np.array([s[1] for s in samples], dtype=np.float64)
        w = np.array([s[2] for s in samples], dtype=np.float64)
        fr = np.array([s[3] if len(s) > 3 else 1.0 for s in samples], dtype=np.float64)
        yw = np.array([s[4] if len(s) > 4 else 0.0 for s in samples], dtype=np.float64)
        cx = np.array([s[5] if len(s) > 5 else 0.0 for s in samples], dtype=np.float64)
        cy = np.array([s[6] if len(s) > 6 else 0.0 for s in samples], dtype=np.float64)
        sc = np.array([s[7] if len(s) > 7 else 0.0 for s in samples], dtype=np.float64)
        dur = float(ts[-1] - ts[0])
        if dur < 1e-3:
            return None
        base = np.median(v)
        x = v - base
        amp = float(np.std(x))
        sign = np.sign(x)
        sign[sign == 0] = 1.0
        crossings = int(np.count_nonzero(np.diff(sign) != 0))
        cross_hz = crossings / dur

        mean_w = float(np.mean(w))
        width_cv = float(np.std(w) / mean_w) if mean_w > 1e-6 else 1.0
        width_ratio = mean_w / ref_width if ref_width > 1e-6 else 1.0
        front_med = float(np.median(fr))
        yaw_std = float(np.std(yw))
        med_sc = float(np.median(sc))
        sway = (
            float(np.hypot(np.std(cx), np.std(cy)) / med_sc) if med_sc > 1e-6 else 0.0
        )

        score = float(np.clip(amp / MAR_STD_FULL, 0.0, 1.0))
        why = ""
        if cross_hz < MIN_CROSS_HZ:
            score *= 0.25  # motion without speech rhythm (yawn/bite)
            why += "norhythm "
        if width_cv > TURN_WIDTH_CV:
            score *= GATE_SUPPRESS  # mouth width swinging => head turning
            why += "turning "
        if width_ratio < PROFILE_WIDTH_FRAC:
            score *= GATE_SUPPRESS  # mouth much narrower than frontal => profile
            why += "profile "
        if front_med < FRONT_MIN:
            score *= GATE_SUPPRESS  # head turned / side-on => can't judge mouth
            why += "lowfront "
        if yaw_std > YAW_TURN_STD:
            score *= GATE_SUPPRESS  # yaw swinging => head is TURNING right now
            why += "yawmove "
        if sway > SWAY_MAX:
            score *= GATE_SUPPRESS  # face sliding in frame => body swaying/walking
            why += "sway "
        if self.debug:
            log.info(
                "VVAD tid=%s amp=%.3f cross=%.1fHz wcv=%.2f wratio=%.2f "
                "front=%.2f yaw_sd=%.1f sway=%.2f -> %.2f %s",
                tid,
                amp,
                cross_hz,
                width_cv,
                width_ratio,
                front_med,
                yaw_std,
                sway,
                score,
                why or "ok",
            )
        return score

    def score_track(self, track_id: int) -> Optional[float]:
        """Most-recent WINDOW_SEC ('who is talking now' / fallback)."""
        return self._score_samples(
            self._recent(track_id), self._ref_width(track_id), track_id
        )

    def score_window(
        self, track_id: int, win_start: float, win_end: float
    ) -> Optional[float]:
        """Score an explicit [start,end] epoch span. If longer than
        WINDOW_SEC, slide a WINDOW_SEC window across it and take the MAX.
        """
        t0 = float(win_start) - WINDOW_PAD_SEC
        t1 = float(win_end) + WINDOW_PAD_SEC
        span = self._samples_in(track_id, t0, t1)
        if len(span) < self.min_samples:
            return None
        ref_w = self._ref_width(track_id)
        if (t1 - t0) <= self.window_sec:
            return self._score_samples(span, ref_w, track_id)
        best: Optional[float] = None
        start = t0
        while start <= t1 - self.window_sec + 1e-6:
            sub = [s for s in span if start <= s[0] <= start + self.window_sec]
            s = self._score_samples(sub, ref_w, track_id)
            if s is not None and (best is None or s > best):
                best = s
            start += SLIDE_STEP_SEC
        # also score the trailing window so the very end isn't missed
        tail = [s for s in span if s[0] >= t1 - self.window_sec]
        s = self._score_samples(tail, ref_w, track_id)
        if s is not None and (best is None or s > best):
            best = s
        return best

    def resolve_speaking(
        self, candidate_track_ids: List[int]
    ) -> Tuple[Optional[int], Dict[int, float]]:
        """Score each candidate track and return the top speaker (if above threshold) with all scores."""
        scores: Dict[int, float] = {}
        for tid in candidate_track_ids:
            s = self.score_track(int(tid))
            if s is not None:
                scores[int(tid)] = s
        if not scores:
            return None, scores
        best = max(scores, key=lambda tid: scores[tid])
        return (best if scores[best] >= self.speaking_thr else None), scores

    def resolve_window(
        self, candidate_track_ids: List[int], win_start: float, win_end: float
    ) -> Dict[int, float]:
        """Score each candidate track over the given [win_start, win_end] span."""
        scores: Dict[int, float] = {}
        for tid in candidate_track_ids:
            s = self.score_window(int(tid), win_start, win_end)
            if s is not None:
                scores[int(tid)] = s
        return scores
