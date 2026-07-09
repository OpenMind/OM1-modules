#!/usr/bin/env python3
"""
HTTP control plane for the realtime face pipeline (UUID-keyed gallery).

This module exposes a thin, thread-safe HTTP API that lets external tools
query presence (/who), tweak runtime configuration (/config), and manage the
face gallery (/gallery/refresh, /gallery/add_aligned, /gallery/add_raw,
/selfie, /set_name, /gallery/forget_last, /gallery/restore,
/gallery/delete, /gallery/identities, /gallery/auto_enroll_status,
/gallery/merge, /gallery/find_similar) while the main video loop keeps
running.

Identity model
--------------
Identities are referenced by UUID (32-char hex). Display names are mutable
metadata; two UUIDs may share the same name (e.g. one auto-enrolled and one
selfie-enrolled instance of the same person). All write endpoints that
target a single identity accept ``uuid`` as the primary parameter. Legacy
endpoints that took ``id``/``name`` resolve via ``gallery.find_by_name``
when the matching UUID isn't given.

Thread / GPU model
------------------
- HTTP server runs on its own thread. Handlers never call CUDA/TensorRT
  directly; instead they use ``run_job_sync(fn)`` to enqueue work onto the
  main thread's job queue so GPU/TensorRT operations are serialized.
- Shared state (config, gallery, in-memory centroids, last clean frame) is
  protected by locks supplied by the caller.
- The ``UUIDGallery`` and ``AutoEnroller`` instances assume single-thread
  access — all writes to them must go through ``run_job_sync``.
"""

from __future__ import annotations

import base64
import logging
import os.path as osp
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .vvad_speaker import VVADScorer

import cv2
import numpy as np

from . import selfie_logic as sl
from .adaface import warp_face_by_5p

logger = logging.getLogger(__name__)


# Human-readable hint for why selfie failed, based on which reason dominated.
# Maps the top-1 rejection reason to actionable user advice. Used in the
# /selfie response so the LLM (or human debugger) knows what to tell the user.
_SELFIE_HINT_MAP = {
    "no_face_detected": "No face seen in the frame. Stand in front of the camera.",
    "not_engaged": "Face is too small or not facing the camera. Walk closer and face the camera directly.",
    "ambiguous_faces": "Multiple people in frame at similar distance. Step forward so you're clearly the closest.",
    "empty_bbox": "Detection box was invalid (likely edge of frame). Center yourself in the frame.",
    "too_small": "Face is too small in the frame. Walk closer to the camera.",
    "low_conf": "Detector isn't confident this is a face. Improve lighting or get closer.",
    "bad_pose": "Face is too turned. Look at the camera directly.",
    "blurry": "Image is blurry. Hold still or improve lighting.",
    "too_dark": "Frame is too dark. Improve lighting.",
    "too_bright": "Frame is overexposed. Reduce backlight / strong light source.",
    "consistency_fail": "Frames look like different people mid-window. Hold still.",
    "novelty_fail": "All frames look identical (frozen feed?). Move slightly.",
    "quality_fail": "Generic quality check failure.",
}


def _selfie_hint(reasons: Dict[str, int]) -> str:
    """Return the actionable hint for the most-frequent rejection reason."""
    if not reasons:
        return "No frames were inspected. Camera may be stalled."
    top_reason = max(reasons.items(), key=lambda kv: kv[1])[0]
    return _SELFIE_HINT_MAP.get(top_reason, f"Unknown rejection: {top_reason}")


class HttpAPI:
    """HTTP endpoint handlers for the UUID-keyed face gallery.

    Parameters
    ----------
    who : WhoTracker
        Presence tracker; used to answer ``/who`` queries.
    scrfd : TRTSCRFD
        Detector reference; used for ``/config`` hot-tunables and to align
        ``/gallery/add_raw`` uploads.
    gallery : UUIDGallery or None
        UUID-keyed gallery. ``None`` disables all recognition endpoints.
    auto_enroller : AutoEnroller or None
        Optional auto-enrollment subsystem. If provided, ``/selfie`` will
        consult its per-track buffer to find an already-committed UUID
        for the current face before creating a new one.
    gallery_dir : str
        Filesystem root of the gallery (also the path passed to
        ``UUIDGallery``).
    gal_state : Any
        Mutable object holding in-memory centroids consumed by recognition.
        Expected attributes: ``gal_feats: np.ndarray|None``,
        ``gal_labels: list[str]``, ``gal_uuids: list[str]``.
    gal_lock : threading.Lock
        Lock guarding ``gal_state`` reads/writes.
    cfg : dict
        Live runtime configuration (thresholds, modes).
    cfg_lock : threading.Lock
        Lock guarding ``cfg``.
    frame_state : Any
        Last clean frame + detections, captured by the main loop.
        Expected attributes: ``frame_bgr``, ``dets``, ``kpss``,
        ``current_faces``, ``current_unknowns``, ``last_ts``.
    frame_lock : threading.Lock
        Lock guarding ``frame_state``.
    run_job_sync : Callable[[Callable[[], Any]], Any]
        Enqueue-and-wait helper that runs the given function on the main
        thread and returns its result.
    logger : logging.Logger, optional
        Logger to use; defaults to ``logging.getLogger("http_api")``.
    face_tracker : FaceTracker, optional
        Face tracker reference. Used to look up the currently-tracked face
        UUID at the time of ``/selfie`` so we can drain the matching
        auto-enroll buffer.
    """

    def __init__(
        self,
        *,
        who,
        scrfd,
        gallery,
        auto_enroller,
        gallery_dir: str,
        gal_state,
        gal_lock: threading.Lock,
        cfg: Dict[str, Any],
        cfg_lock: threading.Lock,
        frame_state,
        frame_lock: threading.Lock,
        run_job_sync: Callable[[Callable[[], Any]], Any],
        logger: Optional[logging.Logger] = None,
        face_tracker=None,
    ):
        self.who = who
        self.scrfd = scrfd
        self.gallery = gallery
        self.auto_enroller = auto_enroller
        self.gallery_dir = gallery_dir
        self.gal_state = gal_state
        self.gal_lock = gal_lock
        self.cfg = cfg
        self.cfg_lock = cfg_lock
        self.frame_state = frame_state
        self.frame_lock = frame_lock
        # Vision-only speaking scorer (set by run.py). None/shadow => existing
        # largest/most-engaged selection is used for enrollment.
        self.vvad: Optional["VVADScorer"] = None
        self.vvad_shadow = True
        # Speaker chosen by the most recent /speaking call (over the ASR
        # utterance window). Consumed by /selfie so enrollment binds to the
        # person who actually spoke, not the largest face. Deterministic —
        # the LLM never picks the face.
        self._speaking_latch = None  # {track_id, uuid, name, bbox, ts}
        self.speaking_latch_ttl = 5.0  # seconds a latch stays valid
        self.run_job_sync = run_job_sync
        self.log = logger or logging.getLogger("http_api")
        self.server = None
        self.face_tracker = face_tracker

        # Serialize /selfie (gallery + last_enrollment can't tolerate
        # concurrent writes from two callers)
        self.selfie_lock = threading.Lock()
        # Protect last_enrollment dict
        self.enrollment_lock = threading.Lock()
        # Serialize corrections.log writes
        self.audit_lock = threading.Lock()
        # Tracks the most recent enrollment so /gallery/forget_last and
        # /gallery/restore can act on "the one we just made". UUID-keyed.
        # Shape: {"uuid", "name", "monotonic_ts", "wall_ts", "merged",
        #         "source", "sample_count"}
        self.last_enrollment: Optional[Dict[str, Any]] = None
        # Tracks the most recently soft-deleted UUID so /gallery/restore
        # can default to it when no UUID is provided. Cleared when the
        # UUID is restored or when trash is emptied.
        self.last_soft_deleted: Optional[str] = None

    def stop(self) -> None:
        """Stop the embedded HTTP server. Idempotent; errors during shutdown
        are swallowed (typical e.g. when server is already stopped).
        """
        try:
            if self.server:
                self.server.stop()
        except Exception:
            pass

    # ==================================================================
    # Dispatch
    # ==================================================================

    def _handle(self, payload: Dict[str, Any], path: str) -> Dict[str, Any]:
        """Dispatch a POST request to the appropriate handler.

        On error returns ``{"error": <message>}``; never raises.
        """
        try:
            # Liveness / utility
            if path == "/ping":
                return {"ok": True}
            if path == "/ts":
                return {"ts": time.time()}

            # Presence
            if path == "/who":
                return self._handle_who(payload)

            # Who is speaking (over an ASR utterance window)
            if path == "/speaking":
                return self._handle_speaking(payload)

            # Config
            if path == "/config":
                return self._handle_config(payload)

            # Gallery: reading
            if path in ("/gallery/identities", "/gallery/list_identities"):
                return self._handle_gallery_identities()
            if path == "/gallery/auto_enroll_status":
                return self._handle_auto_enroll_status()

            # Gallery: writing
            if path == "/gallery/refresh":
                return self._handle_gallery_refresh()
            if path == "/gallery/add_aligned":
                return self._handle_gallery_add_aligned(payload)
            if path == "/gallery/add_raw":
                return self._handle_gallery_add_raw(payload)
            if path == "/selfie":
                return self._handle_selfie(payload)

            # Gallery: identity lifecycle
            if path == "/set_name":
                return self._handle_set_name(payload)
            if path == "/set_name_current":
                return self._handle_set_name_current(payload)
            if path == "/gallery/forget_last":
                return self._handle_gallery_forget_last(payload)
            if path == "/gallery/restore":
                return self._handle_gallery_restore(payload)
            if path == "/gallery/delete":
                return self._handle_gallery_delete(payload)
            if path == "/gallery/empty_trash":
                return self._handle_gallery_empty_trash()
            if path == "/gallery/merge":
                return self._handle_gallery_merge(payload)
            if path == "/gallery/find_similar":
                return self._handle_gallery_find_similar(payload)
            if path == "/gallery/find_similar_current":
                return self._handle_gallery_find_similar_current(payload)
            if path == "/gallery/merge_current":
                return self._handle_gallery_merge_current(payload)
            if path == "/gallery/sweep_stale":
                return self._handle_gallery_sweep_stale(payload)

            return {"error": f"unknown path {path}"}
        except Exception as e:
            self.log.exception("HTTP error on %s", path)
            return {"error": str(e)}

    # ==================================================================
    # /who
    # ==================================================================

    def _handle_who(self, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return presence snapshot plus the current frame as base64."""
        sec = (
            float(payload.get("recent_sec", self.who.lookback_sec))
            if payload
            else self.who.lookback_sec
        )
        result = self.who.snapshot(sec)

        # Read frame + faces + unknowns under one lock (consistent snapshot)
        with self.frame_lock:
            frm = (
                self.frame_state.frame_bgr.copy()
                if self.frame_state.frame_bgr is not None
                else None
            )
            faces = self.frame_state.current_faces
            unknowns = self.frame_state.current_unknowns

        if self.face_tracker is not None:
            # Decorate faces with metadata the LLM uses to choose its
            # greeting style:
            #
            #   created_ago_sec    — how long ago was this UUID first
            #                        added to the gallery. ("we've ever
            #                        met")
            #   last_seen_ago_sec  — gap (in seconds) between the
            #                        PREVIOUS sighting of this UUID and
            #                        the current track session start.
            #                        Sticky: stays constant for the
            #                        whole track session so the LLM's
            #                        "newcomer / met before" judgment
            #                        doesn't flip mid-conversation.
            #   last_seen_iso      — the previous-sighting ISO timestamp
            #                        itself (string), so the LLM can
            #                        say "I last saw you on 2026-03-05".
            #
            # Unknown / no-UUID faces get None for all three.
            now = time.time()
            decorated_faces = []
            for f in faces:
                fc = dict(f)  # shallow copy so we don't mutate frame_state
                uuid = fc.get("uuid")
                created_ago_sec: Optional[float] = None
                last_seen_ago_sec: Optional[float] = None
                last_seen_iso: Optional[str] = fc.get("session_last_seen_iso")

                if uuid and self.gallery is not None:
                    rec = self.gallery.get_record(uuid)
                    if rec is not None:
                        ts_iso = rec.metadata.get("created_at")
                        if ts_iso:
                            try:
                                ts = time.mktime(
                                    time.strptime(ts_iso[:19], "%Y-%m-%dT%H:%M:%S")
                                )
                                created_ago_sec = max(0.0, now - ts)
                            except (ValueError, TypeError):
                                pass

                # Compute last_seen_ago_sec from the sticky session value
                # (NOT from gallery's current last_seen_at, which would
                # always be ~0 for the visible face).
                if last_seen_iso:
                    try:
                        ts = time.mktime(
                            time.strptime(last_seen_iso[:19], "%Y-%m-%dT%H:%M:%S")
                        )
                        last_seen_ago_sec = max(0.0, now - ts)
                    except (ValueError, TypeError):
                        pass

                fc["created_ago_sec"] = created_ago_sec
                fc["last_seen_ago_sec"] = last_seen_ago_sec
                fc["last_seen_iso"] = last_seen_iso
                # Drop the internal field name — clients use last_seen_iso
                fc.pop("session_last_seen_iso", None)
                decorated_faces.append(fc)
            result["faces"] = decorated_faces
            if unknowns:
                result["unknown_captures"] = unknowns

        if frm is not None:
            _, buf = cv2.imencode(".jpg", frm, [cv2.IMWRITE_JPEG_QUALITY, 80])
            result["frame_b64"] = base64.b64encode(buf.tobytes()).decode("ascii")
            result["frame_hw"] = list(frm.shape[:2])
            result["frame_ts"] = time.time()
        else:
            result["frame_b64"] = None
            result["frame_hw"] = None
            result["frame_ts"] = None

        return result

    # ==================================================================
    # /config
    # ==================================================================

    def _handle_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get or set live runtime configuration.

        Payload forms:
          {"get": true}                  → return current cfg
          {"set": {"key": value, ...}}   → update keys (only existing ones)
        """
        payload = payload or {}
        if payload.get("get"):
            with self.cfg_lock:
                return {"ok": True, "cfg": dict(self.cfg)}
        if "set" in payload and isinstance(payload["set"], dict):
            updates = payload["set"]
            applied: Dict[str, Any] = {}
            with self.cfg_lock:
                for k, v in updates.items():
                    if k in self.cfg:
                        # Preserve type — coerce v through type(old)
                        try:
                            self.cfg[k] = type(self.cfg[k])(v)
                            applied[k] = self.cfg[k]
                        except (TypeError, ValueError) as e:
                            return {"error": "bad_value", "key": k, "detail": str(e)}
            # Some keys need to push into engines/trackers right away
            if "nms" in applied and self.scrfd is not None:
                self.scrfd.nms_thresh = float(applied["nms"])
            return {"ok": True, "applied": applied}
        return {"error": "missing get|set"}

    # ==================================================================
    # /gallery/identities
    # ==================================================================

    def _handle_gallery_identities(self) -> Dict[str, Any]:
        """List all identities in the gallery.

        Returns
        -------
        {
            "ok": true,
            "total": <int>,
            "identities": [
                {"uuid": "...", "name": "...", "sample_count": N,
                 "created_at": "...", "updated_at": "...", "source": "..."},
                ...
            ]
        }
        """
        if self.gallery is None:
            return {"ok": True, "total": 0, "identities": []}
        items = self.gallery.list_identities()
        return {"ok": True, "total": len(items), "identities": items}

    # ==================================================================
    # /gallery/auto_enroll_status
    # ==================================================================

    def _handle_auto_enroll_status(self) -> Dict[str, Any]:
        """Diagnostic snapshot of the auto-enroll subsystem."""
        if self.auto_enroller is None:
            return {"ok": True, "enabled": False}
        return {"ok": True, "enabled": True, **self.auto_enroller.snapshot()}

    # ==================================================================
    # /gallery/refresh
    # ==================================================================

    def _handle_gallery_refresh(self) -> Dict[str, Any]:
        """Re-scan the gallery directory and refresh in-memory centroids.

        Used after manual file-level edits (e.g. dragging in extra
        ``aligned/*.jpg`` files). Fast: just disk I/O + in-memory load,
        no GPU.
        """
        if self.gallery is None:
            return {"error": "recognition_disabled"}

        def _do():
            n_loaded = self.gallery.reload()
            feats, labels, uuids = self.gallery.get_centroids()
            with self.gal_lock:
                self.gal_state.gal_feats = feats
                self.gal_state.gal_labels = labels
                self.gal_state.gal_uuids = uuids
            return n_loaded, len(uuids)

        t0 = time.monotonic()
        n_loaded, n_active = self.run_job_sync(_do)
        return {
            "ok": True,
            "identities_loaded": int(n_loaded),
            "identities_with_samples": int(n_active),
            "took_sec": round(time.monotonic() - t0, 3),
        }

    # ==================================================================
    # /gallery/add_aligned
    # ==================================================================

    def _handle_gallery_add_aligned(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Add a pre-aligned 112×112 crop to the gallery.

        Payload
        -------
        uuid  : str (optional) — target UUID. If omitted, looks up by name;
                if no UUID matches the name, creates a new UUID.
        name  : str (required if no uuid) — display name for new UUID.
        image_path : str (one of image_path / image_b64 required)
        image_b64  : str

        Returns
        -------
        {"ok": True, "uuid": "...", "name": "...", "sample_count": N,
         "took_sec": ...}
        """
        if self.gallery is None:
            return {"error": "recognition_disabled"}
        payload = payload or {}
        uuid = str(payload.get("uuid", "")).strip()
        name = str(payload.get("name", "")).strip().lower()

        img = self._load_112(payload.get("image_path"), payload.get("image_b64"))
        if img is None:
            return {"error": "no_image"}

        if not uuid and not name:
            return {"error": "missing_uuid_or_name"}

        if name:
            ok, reason = sl.validate_identity_name(name)
            if not ok:
                return {"error": "bad_name", "detail": reason}

        def _do():
            target_uuid = uuid
            # Resolve name → UUID if no UUID given. Pick the first match.
            if not target_uuid and name:
                matches = self.gallery.find_by_name(name)
                if matches:
                    target_uuid = matches[0]
                else:
                    target_uuid = self.gallery.create(name=name, source="add_aligned")
            elif not self.gallery.get_record(target_uuid):
                return None, f"uuid not found: {target_uuid}"

            self.gallery.add_sample(target_uuid, img)
            rec = self.gallery.get_record(target_uuid)

            feats, labels, uuids = self.gallery.get_centroids()
            with self.gal_lock:
                self.gal_state.gal_feats = feats
                self.gal_state.gal_labels = labels
                self.gal_state.gal_uuids = uuids
            return rec, None

        t0 = time.monotonic()
        rec, err = self.run_job_sync(_do)
        if err:
            return {"error": err}
        return {
            "ok": True,
            "uuid": rec.uuid,
            "name": rec.name,
            "sample_count": rec.sample_count,
            "took_sec": round(time.monotonic() - t0, 3),
        }

    # ==================================================================
    # /gallery/add_raw
    # ==================================================================

    def _handle_gallery_add_raw(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Add a raw (any-size) image: detect → align → embed → add."""
        if self.gallery is None:
            return {"error": "recognition_disabled"}
        if self.scrfd is None:
            return {"error": "no_detector"}

        payload = payload or {}
        uuid = str(payload.get("uuid", "")).strip()
        name = str(payload.get("name", "")).strip().lower()

        if not uuid and not name:
            return {"error": "missing_uuid_or_name"}
        if name:
            ok, reason = sl.validate_identity_name(name)
            if not ok:
                return {"error": "bad_name", "detail": reason}

        img_path = payload.get("image_path")
        img_b64 = payload.get("image_b64")
        raw = None
        if img_path:
            raw = cv2.imread(img_path)
        elif img_b64:
            try:
                buf = base64.b64decode(img_b64)
                arr = np.frombuffer(buf, dtype=np.uint8)
                raw = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                raw = None
        if raw is None:
            return {"error": "no_image"}

        det_conf_snap = float(self.cfg.get("conf", 0.5))

        def _do():
            dets, kpss = self.scrfd.detect(raw, conf=det_conf_snap)
            if dets is None or dets.shape[0] == 0:
                return None, "no_face"
            # Pick largest face
            areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
            i = int(np.argmax(areas))
            if kpss is not None and i < len(kpss):
                crop = warp_face_by_5p(raw, kpss[i], 112)
            else:
                x1, y1, x2, y2 = dets[i, :4].astype(int)
                face = raw[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                if face.size == 0:
                    return None, "no_face_crop"
                crop = cv2.resize(face, (112, 112))

            target_uuid = uuid
            if not target_uuid and name:
                matches = self.gallery.find_by_name(name)
                target_uuid = (
                    matches[0]
                    if matches
                    else self.gallery.create(
                        name=name,
                        source="add_raw",
                    )
                )
            elif not self.gallery.get_record(target_uuid):
                return None, f"uuid not found: {target_uuid}"

            self.gallery.add_sample(target_uuid, crop)
            rec = self.gallery.get_record(target_uuid)

            feats, labels, uuids = self.gallery.get_centroids()
            with self.gal_lock:
                self.gal_state.gal_feats = feats
                self.gal_state.gal_labels = labels
                self.gal_state.gal_uuids = uuids
            return rec, None

        t0 = time.monotonic()
        rec, err = self.run_job_sync(_do)
        if err:
            return {"error": err}
        return {
            "ok": True,
            "uuid": rec.uuid,
            "name": rec.name,
            "sample_count": rec.sample_count,
            "took_sec": round(time.monotonic() - t0, 3),
        }

    # ==================================================================
    # /selfie — multi-frame enrollment from the live camera
    # ==================================================================

    def _handle_selfie(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Collect 1-N frames from the live camera and enroll the addressee.

        Payload
        -------
        name  : str (required) — display name to assign
        force : bool (optional) — bypass cross-name reject (twins/lookalikes)

        Returns (success)
        -----------------
        {ok:True, uuid, name, merged, forced, samples_saved, match_sim,
         match_name, identities}

        Errors
        ------
        bad_name | recognition_disabled | busy | no_valid_frames |
        insufficient_samples | ambiguous_subjects | face_belongs_to
        """
        if self.gallery is None:
            return {"error": "recognition_disabled"}

        payload = payload or {}
        # Accept both new "name" and legacy "id" keys for compatibility.
        requested = str(payload.get("name") or payload.get("id", "")).strip().lower()
        ok, reason = sl.validate_identity_name(requested)
        if not ok:
            return {"error": "bad_name", "detail": reason}
        force = bool(payload.get("force", False))

        if not self.selfie_lock.acquire(blocking=False):
            return {"error": "busy"}
        try:
            return self._do_selfie(requested, force)
        finally:
            self.selfie_lock.release()

    def _do_selfie(self, requested: str, force: bool) -> Dict[str, Any]:
        """Core selfie loop. Caller holds self.selfie_lock."""
        with self.cfg_lock:
            cfg_snap = dict(self.cfg)

        window_sec = float(cfg_snap["selfie_window_sec"])
        tap_interval = float(cfg_snap["selfie_tap_interval_sec"])
        min_samples = int(cfg_snap["selfie_min_samples"])
        max_samples = int(cfg_snap["selfie_max_samples"])
        score_floor = float(cfg_snap["selfie_min_engagement"])
        ambig_ratio = float(cfg_snap["selfie_ambiguity_ratio"])
        novelty_thr = float(cfg_snap["selfie_novelty_thr"])
        consist_thr = float(cfg_snap["selfie_consistency_thr"])

        deadline = time.monotonic() + window_sec
        crops: List[np.ndarray] = []
        embs: List[np.ndarray] = []
        running_mean: Optional[np.ndarray] = None

        # Rejection-reason counter for debugging "no_valid_frames" failures.
        # Counts each frame that was skipped and why. Keys mirror quality_check
        # reasons plus the higher-level pipeline skips.
        reject_reasons: Dict[str, int] = {}
        frames_seen = 0  # total frames inspected (not "skipped because stale")

        def _bump(reason: str) -> None:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

        target_uuid: Optional[str] = None
        target_name: str = requested
        merged_flag = False
        match_name: Optional[str] = None
        match_uuid: Optional[str] = None
        match_sim_seen = 0.0
        # Explicit "have we resolved which identity to enroll into yet" flag.
        # Don't conflate with target_uuid==None — for the new-UUID case
        # target_uuid is legitimately None after resolution (we'll mint a
        # UUID at commit time).
        resolved = False
        last_seen_token = -1

        # Resolve who is speaking ONCE up front. Prefers the speaker latched by
        # a recent /speaking call (the real utterance window); else resolves now.
        # None => keep the existing largest/most-engaged selection (fallback).
        speaking_bbox = self._speaking_bbox_for_selfie()

        while len(crops) < max_samples and time.monotonic() < deadline:
            # Snapshot clean frame under lock
            with self.frame_lock:
                if self.frame_state.frame_bgr is None:
                    frm, dets, kpss = None, None, None
                    ts = 0.0
                else:
                    frm = self.frame_state.frame_bgr.copy()
                    dets = (
                        None
                        if self.frame_state.dets is None
                        else self.frame_state.dets.copy()
                    )
                    kpss = (
                        None
                        if self.frame_state.kpss is None
                        else self.frame_state.kpss.copy()
                    )
                    ts = float(getattr(self.frame_state, "last_ts", 0.0))

            frame_token = ts if ts > 0 else (id(frm) if frm is not None else 0)
            if frame_token == last_seen_token:
                time.sleep(0.02)  # wait for fresh frame
                continue
            last_seen_token = frame_token
            frames_seen += 1

            if frm is None or dets is None or dets.shape[0] == 0:
                _bump("no_face_detected")
                time.sleep(tap_interval)
                continue

            # Score each detected face — pick the most engaged
            H, W = frm.shape[:2]
            scores: List[float] = []
            for d_idx in range(dets.shape[0]):
                k = kpss[d_idx] if kpss is not None and d_idx < len(kpss) else None
                scores.append(sl.score_face(dets[d_idx], k, float(H * W)))
            # Bias selection toward the speaking face (if any). The bonus makes
            # argmax pick it; when speaking_bbox is None nothing changes (the
            # existing largest/most-engaged fallback).
            if speaking_bbox is not None:
                bi, best_iou = -1, 0.0
                for d_idx in range(dets.shape[0]):
                    iou = self._iou(dets[d_idx][:4], speaking_bbox)
                    if iou > best_iou:
                        bi, best_iou = d_idx, iou
                if bi >= 0 and best_iou > 0.2:
                    scores[bi] += 1e6
            top_idx = int(np.argmax(scores))
            top_score = scores[top_idx]
            if top_score < score_floor:
                _bump("not_engaged")
                continue

            # Ambiguity: skip frame if two faces are similarly engaged
            ambiguous, _ = sl.check_ambiguity(scores, ambig_ratio, score_floor)
            if ambiguous:
                _bump("ambiguous_faces")
                continue

            det = dets[top_idx]
            kps = kpss[top_idx] if kpss is not None and top_idx < len(kpss) else None

            # Align
            if kps is not None:
                crop = warp_face_by_5p(frm, kps, 112)
            else:
                x1, y1, x2, y2 = det[:4].astype(int)
                face = frm[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                if face.size == 0:
                    _bump("empty_bbox")
                    continue
                crop = cv2.resize(face, (112, 112))

            # Per-frame quality
            ok, reason = sl.quality_check(crop, det, kps, cfg_snap)
            if not ok:
                _bump(reason or "quality_fail")
                continue

            # Embed (on main thread — TRT)
            v = self.run_job_sync(lambda c=crop: self.gallery.embed_aligned(c))

            # First valid frame → resolve target. Subsequent → consistency + novelty.
            if not resolved:
                # Pull gallery snapshot
                with self.gal_lock:
                    feats = self.gal_state.gal_feats
                    labels = list(self.gal_state.gal_labels)
                    uuids = list(getattr(self.gal_state, "gal_uuids", []))

                decision = sl.resolve_target_id(
                    v,
                    requested,
                    feats,
                    labels,
                    uuids,
                    cfg_snap,
                    force=force,
                )

                if decision.reject:
                    return {
                        "error": "face_belongs_to",
                        "name": decision.match_name,
                        "uuid": decision.match_uuid,
                        "sim": round(decision.match_sim, 3),
                    }

                target_uuid = decision.uuid  # None = create new UUID at commit time
                target_name = decision.name
                merged_flag = decision.merged
                match_name = decision.match_name
                match_uuid = decision.match_uuid
                match_sim_seen = decision.match_sim
                resolved = True
            else:
                # Consistency + novelty against running_mean. running_mean
                # is guaranteed non-None here because we set it in the
                # previous iteration before bumping `resolved`.
                assert running_mean is not None
                sim = sl.cosine_to_mean(running_mean, v)
                if sim < consist_thr:
                    _bump("consistency_fail")
                    continue  # different person mid-window
                if sim > novelty_thr:
                    _bump("novelty_fail")
                    continue  # near-duplicate frame

            crops.append(crop)
            embs.append(v)
            running_mean = sl.update_running_mean(running_mean, v, len(embs))
            time.sleep(tap_interval)

        # Finalize
        n = len(crops)
        if n == 0:
            logger.warning(
                "selfie no_valid_frames: frames_seen=%d rejections=%s",
                frames_seen,
                dict(reject_reasons),
            )
            return {
                "error": "no_valid_frames",
                "frames_seen": frames_seen,
                "rejections": dict(reject_reasons),
                "hint": _selfie_hint(reject_reasons),
            }
        if n < min_samples:
            logger.warning(
                "selfie insufficient_samples: got=%d need=%d rejections=%s",
                n,
                min_samples,
                dict(reject_reasons),
            )
            return {
                "error": "insufficient_samples",
                "got": n,
                "need": min_samples,
                "rejections": dict(reject_reasons),
                "hint": _selfie_hint(reject_reasons),
            }

        # Commit: create UUID if needed, then batch-add to the gallery.
        # After commit, proactively drop the auto-enroll buffer for the
        # track this selfie corresponds to — face_tracker will eventually
        # do the same on its next recognition pass (when it sees the new
        # centroid and the track becomes "confident"), but draining now
        # avoids a brief window where the buffer could re-commit.
        def _do_commit():
            uuid = target_uuid
            if uuid is None:
                uuid = self.gallery.create(name=target_name, source="selfie")
            else:
                # Merging into existing UUID. If we're claiming an anon
                # (target_name was overridden by resolve_target_id to the
                # requested name), also rename the UUID now. set_name is
                # O(1) so this is cheap. For non-claim merges (e.g. selfie
                # of an already-named identity), target_name == existing
                # name, so set_name is a no-op (same string).
                rec = self.gallery.get_record(uuid)
                if rec is not None and rec.name != target_name:
                    self.gallery.set_name(uuid, target_name)

            saved = self.gallery.add_samples(uuid, crops, embeddings=embs)
            n_saved = len(saved)

            # Refresh in-memory centroids so face_tracker picks up the
            # new identity on its next frame.
            feats, labels, uuids = self.gallery.get_centroids()
            with self.gal_lock:
                self.gal_state.gal_feats = feats
                self.gal_state.gal_labels = labels
                self.gal_state.gal_uuids = uuids

            # Drop the auto-enroll buffer for the currently-tracked face,
            # if any. Best-effort: if there's no auto-enroller, no face
            # tracker, or no current face, this is a no-op.
            if self.auto_enroller is not None and self.face_tracker is not None:
                track_id = self._currently_tracked_track_id()
                if track_id is not None:
                    self.auto_enroller.drop_track(track_id)

            return uuid, n_saved, len(uuids)

        t0 = time.monotonic()
        final_uuid, n_saved, n_id = self.run_job_sync(_do_commit)

        with self.enrollment_lock:
            self.last_enrollment = {
                "uuid": final_uuid,
                "name": target_name,
                "monotonic_ts": time.monotonic(),
                "wall_ts": time.time(),
                "merged": merged_flag,
                "forced": force,
                "sample_count": n_saved,
            }

        if force:
            self._audit_log(
                f"selfie forced uuid={final_uuid} name={target_name} "
                f"matched={match_name or '-'} sim={match_sim_seen:.3f}"
            )

        return {
            "ok": True,
            "uuid": final_uuid,
            "name": target_name,
            "merged": merged_flag,
            "forced": force,
            "samples_saved": n_saved,
            "match_name": match_name,
            "match_uuid": match_uuid,
            "match_sim": round(match_sim_seen, 3),
            "identities": int(n_id),
            "took_sec": round(time.monotonic() - t0, 3),
        }

    def _currently_tracked_track_id(self) -> Optional[int]:
        """Return the track_id of the largest currently-tracked face, if any.

        Used by /selfie to drop the auto-enroll buffer for the same track
        after committing. ``current_faces`` is sorted by bbox area
        descending, so faces[0] is the closest/most prominent face.
        """
        if self.face_tracker is None:
            return None
        with self.frame_lock:
            faces = list(self.frame_state.current_faces)
        if not faces:
            return None
        return faces[0].get("track_id")

    # ==================================================================
    # /set_name
    # ==================================================================

    def _handle_set_name(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Update the display name of an identity. O(1), no GPU.

        Two usage patterns:
          1. ``{"uuid": "<hex32>", "name": "wendy"}`` — rename by UUID.
          2. ``{"track_id": 5, "name": "wendy"}`` — name the person currently
             on the screen as track #5. If that track is already mapped to
             a confident UUID, we set_name on that UUID. Otherwise, if
             auto-enroll has buffered samples for that track, we force-commit
             to a brand-new UUID with the given name.

        Name collisions are intentionally allowed
        -----------------------------------------
        Two different people can share a display name (e.g. "two Wendys at
        OpenMind"). Each keeps its own UUID; recognition distinguishes them
        by face, not by name. Both will render as "wendy" on screen — that's
        a UX choice, not an identity confusion.

        To merge two UUIDs into one (same person, multiple enrollments),
        use ``/gallery/merge`` — that's an explicit operation that requires
        confirmation, not a side-effect of renaming.

        Payload
        -------
        uuid     : str (optional)
        track_id : int (optional, alternative to uuid)
        name     : str (required) — new display name
        """
        if self.gallery is None:
            return {"error": "recognition_disabled"}

        payload = payload or {}
        new_name = str(payload.get("name", "")).strip().lower()
        ok, reason = sl.validate_identity_name(new_name)
        if not ok:
            return {"error": "bad_name", "detail": reason}

        uuid = str(payload.get("uuid", "")).strip()
        track_id_raw = payload.get("track_id")

        # Resolve which UUID to update
        target_uuid: Optional[str] = None
        created_via_force = False

        if uuid:
            if self.gallery.get_record(uuid) is None:
                return {"error": "uuid_not_found", "uuid": uuid}
            target_uuid = uuid
        elif track_id_raw is not None:
            try:
                track_id = int(track_id_raw)
            except (TypeError, ValueError):
                return {"error": "bad_track_id"}
            # Look up track on the face_tracker side
            target_uuid = self._uuid_for_track(track_id)
            if target_uuid is None:
                # No confident identification yet. If auto-enroller has a
                # buffer for this track, force-commit it with the name.
                if self.auto_enroller is None:
                    return {
                        "error": "track_not_identified",
                        "detail": "no auto-enroller available",
                    }
                target_uuid = self.run_job_sync(
                    lambda: self.auto_enroller.force_commit(track_id, name=new_name)
                )
                if not target_uuid:
                    return {
                        "error": "track_not_identified",
                        "detail": "no buffered samples for track",
                    }
                created_via_force = True
        else:
            return {"error": "missing_uuid_or_track_id"}

        def _do():
            if not created_via_force:
                # If we didn't just create+name via force_commit, do the rename
                self.gallery.set_name(target_uuid, new_name)
            feats, labels, uuids = self.gallery.get_centroids()
            with self.gal_lock:
                self.gal_state.gal_feats = feats
                self.gal_state.gal_labels = labels
                self.gal_state.gal_uuids = uuids
            return self.gallery.get_record(target_uuid)

        rec = self.run_job_sync(_do)
        return {
            "ok": True,
            "uuid": target_uuid,
            "name": rec.name,
            "created": created_via_force,
            "sample_count": rec.sample_count,
        }

    # ==================================================================
    # /set_name_current — rename whoever is currently in front of camera
    # ==================================================================
    def _handle_set_name_current(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Rename the largest currently-visible face.

        Resolves the on-screen face to its UUID server-side (same
        ``_get_current_largest_face_uuid`` used by merge_current /
        find_similar_current), then renames THAT UUID. Targeting by face
        instead of by name means it works even when several identities
        share a display name ("two Wendys") -- the LLM never has to
        disambiguate and never passes a UUID.

        Unlike merge_current, this DELIBERATELY allows renaming a face
        that is already named (wendy -> yucheng is the whole point).

        It is a pure metadata rename (gallery.set_name + label refresh),
        so it does NOT run the selfie enroll pipeline and never returns
        ``face_belongs_to``.

        Payload
        -------
        name         : str (required) -- new display name
        confirmed_by : str (optional) -- audit string (e.g. "user_voice")

        Returns
        -------
        {ok, uuid, name, prev_name, sample_count, resolved_by} on success.
        Errors: recognition_disabled | bad_name | no_visible_face |
                uuid_not_found
        """
        if self.gallery is None:
            return {"error": "recognition_disabled"}

        payload = payload or {}
        new_name = str(payload.get("name", "")).strip().lower()
        ok, reason = sl.validate_identity_name(new_name)
        if not ok:
            return {"error": "bad_name", "detail": reason}

        # Resolve the target by FACE, not by name.
        face = self._get_current_largest_face_uuid()
        if face is None:
            return {"error": "no_visible_face"}
        target_uuid = face["uuid"]
        prev_name = face.get("name") or ""

        if self.gallery.get_record(target_uuid) is None:
            # In-memory frame had a UUID whose record is gone (e.g. deleted
            # underneath a running server). Surface it instead of crashing.
            return {"error": "uuid_not_found", "uuid": target_uuid}

        def _do():
            # Pure rename: update rec.name + metadata.json on disk...
            self.gallery.set_name(target_uuid, new_name)
            # ...then refresh the in-memory recognition labels so the very
            # next /who poll renders the new name. Same idiom as
            # _handle_set_name.
            feats, labels, uuids = self.gallery.get_centroids()
            with self.gal_lock:
                self.gal_state.gal_feats = feats
                self.gal_state.gal_labels = labels
                self.gal_state.gal_uuids = uuids
            return self.gallery.get_record(target_uuid)

        rec = self.run_job_sync(_do)
        return {
            "ok": True,
            "uuid": target_uuid,
            "name": rec.name if rec is not None else new_name,
            "prev_name": prev_name,
            "sample_count": rec.sample_count if rec is not None else 0,
            "resolved_by": "set_name_current",
        }

    def _uuid_for_track(self, track_id: int) -> Optional[str]:
        """Return the UUID currently assigned to a track, or None."""
        if self.face_tracker is None:
            return None
        idents = self.face_tracker.get_track_identities()
        rec = idents.get(track_id)
        if rec is None:
            return None
        return rec.get("uuid")

    # ==================================================================
    # /gallery/forget_last
    # ==================================================================

    def _handle_gallery_forget_last(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Soft-delete the most recently enrolled UUID. O(1).

        Payload (optional)
        ------------------
        uuid : str — if provided, must match last_enrollment.uuid (safety)

        With UUID gallery, forget is a single rename to ``_trash/<uuid>/`` —
        no GPU work, no rebuild. Use ``/gallery/restore`` to undo, or
        ``/gallery/empty_trash`` / ``/gallery/delete`` to commit.
        """
        if self.gallery is None:
            return {"error": "recognition_disabled"}

        payload = payload or {}
        with self.enrollment_lock:
            le = self.last_enrollment
            if le is None:
                return {"error": "no_recent_enrollment"}
            if time.monotonic() - le["monotonic_ts"] > 60.0:
                return {"error": "stale_enrollment"}
            target_uuid = le["uuid"]
            target_name = le["name"]

            if "uuid" in payload:
                if str(payload["uuid"]).strip() != target_uuid:
                    return {
                        "error": "uuid_mismatch",
                        "detail": f"last enrollment was {target_uuid}",
                    }

        def _do():
            ok = self.gallery.forget(target_uuid)
            feats, labels, uuids = self.gallery.get_centroids()
            with self.gal_lock:
                self.gal_state.gal_feats = feats
                self.gal_state.gal_labels = labels
                self.gal_state.gal_uuids = uuids
            return ok, len(uuids)

        t0 = time.monotonic()
        ok, n_id = self.run_job_sync(_do)
        with self.enrollment_lock:
            self.last_enrollment = None

        # Record for /restore convenience
        if ok:
            self.last_soft_deleted = target_uuid

        self._audit_log(f"forget_last uuid={target_uuid} name={target_name}")
        return {
            "ok": ok,
            "uuid": target_uuid,
            "name": target_name,
            "identities": int(n_id),
            "took_sec": round(time.monotonic() - t0, 3),
        }

    # ==================================================================
    # /gallery/restore
    # ==================================================================

    def _handle_gallery_restore(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Restore a UUID from ``_trash/`` back to the active gallery.

        If no UUID is given, defaults to the most recently soft-deleted UUID
        (set by ``/gallery/delete`` without ``permanent=true``, or by
        ``/gallery/forget_last``).
        """
        if self.gallery is None:
            return {"error": "recognition_disabled"}
        payload = payload or {}
        uuid = str(payload.get("uuid", "")).strip()
        if not uuid:
            # Default to last soft-deleted UUID
            uuid = self.last_soft_deleted or ""
            if not uuid:
                return {
                    "error": "missing_uuid",
                    "detail": "no recent soft-delete to restore — pass uuid explicitly",
                }

        def _do():
            ok = self.gallery.restore(uuid)
            feats, labels, uuids = self.gallery.get_centroids()
            with self.gal_lock:
                self.gal_state.gal_feats = feats
                self.gal_state.gal_labels = labels
                self.gal_state.gal_uuids = uuids
            return ok, len(uuids)

        ok, n_id = self.run_job_sync(_do)
        if not ok:
            return {"error": "not_in_trash", "uuid": uuid}
        # Clear the tracker so subsequent /restore calls without args fail loudly
        if self.last_soft_deleted == uuid:
            self.last_soft_deleted = None
        return {"ok": True, "uuid": uuid, "identities": int(n_id)}

    # ==================================================================
    # /gallery/delete
    # ==================================================================

    def _handle_gallery_delete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Delete one or more identities.

        By default this is a SOFT delete — identities are moved to ``_trash/``
        and can be brought back with ``/gallery/restore``. To permanently
        erase from disk, pass ``permanent: true`` (equivalent to forget +
        empty_trash for those UUIDs).

        Accepts UUIDs (preferred) or names (resolves to all matching UUIDs).
        Payload forms:
          {"uuid": "..."}                    # soft-delete one UUID
          {"uuids": ["...", "..."]}          # soft-delete several
          {"name": "wendy"}                  # soft-delete all UUIDs named wendy
          {"names": ["wendy", "alice"]}
          {"uuid": "...", "permanent": true} # HARD delete (no restore)
        """
        if self.gallery is None:
            return {"error": "recognition_disabled"}
        payload = payload or {}
        permanent = bool(payload.get("permanent", False))

        # Collect target UUIDs
        targets: List[str] = []
        if "uuid" in payload:
            targets.append(str(payload["uuid"]).strip())
        if "uuids" in payload and isinstance(payload["uuids"], list):
            targets.extend(str(u).strip() for u in payload["uuids"])

        # Resolve names → UUIDs
        names_in = []
        if "name" in payload:
            names_in.append(str(payload["name"]).strip().lower())
        if "names" in payload and isinstance(payload["names"], list):
            names_in.extend(str(n).strip().lower() for n in payload["names"])
        for n in names_in:
            targets.extend(self.gallery.find_by_name(n))

        # Deduplicate while preserving order
        seen = set()
        unique_targets = []
        for u in targets:
            if u and u not in seen:
                seen.add(u)
                unique_targets.append(u)

        if not unique_targets:
            return {"error": "no_targets", "detail": "supply uuid/uuids or name/names"}

        def _do():
            deleted: List[str] = []
            failed: Dict[str, str] = {}
            for u in unique_targets:
                try:
                    if permanent:
                        # Hard delete: ensure trashed (no-op if already gone),
                        # then permanently remove.
                        self.gallery.forget(u)
                        purged = self.gallery.hard_delete(u)
                        if purged:
                            deleted.append(u)
                        else:
                            failed[u] = "not_found"
                    else:
                        # Soft delete: move to _trash, restorable.
                        moved = self.gallery.forget(u)
                        if moved:
                            deleted.append(u)
                        else:
                            failed[u] = "not_found"
                except Exception as e:
                    failed[u] = str(e)

            feats, labels, uuids = self.gallery.get_centroids()
            with self.gal_lock:
                self.gal_state.gal_feats = feats
                self.gal_state.gal_labels = labels
                self.gal_state.gal_uuids = uuids
            return deleted, failed, len(uuids)

        t0 = time.monotonic()
        deleted, failed, n_id = self.run_job_sync(_do)

        # Record last soft-deleted UUID for /restore convenience.
        # For multi-delete, last in the list wins (most recent).
        # Hard delete clears it (those UUIDs are unrecoverable).
        if not permanent and deleted:
            self.last_soft_deleted = deleted[-1]
        elif permanent:
            # If any of the hard-deleted UUIDs was our restore target, clear.
            if self.last_soft_deleted in deleted:
                self.last_soft_deleted = None

        self._audit_log(
            f"delete uuids={deleted} failed={list(failed.keys())} "
            f"permanent={permanent}"
        )
        return {
            "ok": len(failed) == 0,
            "deleted": deleted,
            "failed": failed or None,
            "identities": int(n_id),
            "permanent": permanent,
            "restorable": (not permanent) and len(deleted) > 0,
            "took_sec": round(time.monotonic() - t0, 3),
        }

    # ==================================================================
    # /gallery/empty_trash
    # ==================================================================

    def _handle_gallery_empty_trash(self) -> Dict[str, Any]:
        """Permanently delete everything in ``_trash/``."""
        if self.gallery is None:
            return {"error": "recognition_disabled"}
        n = self.run_job_sync(lambda: self.gallery.empty_trash())
        # After emptying trash, nothing is restorable
        self.last_soft_deleted = None
        self._audit_log(f"empty_trash removed={n}")
        return {"ok": True, "removed": int(n)}

    # ==================================================================
    # /gallery/merge
    # ==================================================================

    def _handle_gallery_merge(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Merge ``source_uuid``'s samples into ``target_uuid``, soft-delete source.

        Use case: an anonymous auto-enrolled UUID turns out to be someone
        already in the gallery (e.g. same person, different angles or
        lighting). After confirmation (typically via LLM dialog with the
        person on screen), call this to fold the two UUIDs into one.

        This is an EXPLICIT operation — there's no automatic merge anywhere
        in the system. Face similarity alone is not reliable enough to
        merge without confirmation, since misidentification at this stage
        is hard to undo (sample embeddings get mixed).

        Payload
        -------
        source_uuid  : str (required)  — UUID whose samples will be folded in.
                                         This UUID is soft-deleted (movable to _trash).
        target_uuid  : str (required)  — UUID that absorbs source's samples.
                                         Survives with combined samples + centroid.
        confirmed_by : str (optional)  — Free-form note about how the merge
                                         was confirmed (e.g. "llm_dialog",
                                         "operator", "user_confirmed").
                                         Logged for audit; not validated.

        Returns
        -------
        ok            : bool
        uuid          : str   — target_uuid (winner)
        merged_from   : str   — source_uuid (soft-deleted)
        samples_merged: int   — number of samples added to target
        sim           : float — pre-merge centroid sim (for audit/debug)
        identities    : int   — total identities remaining

        Errors
        ------
        - missing_uuids        — source_uuid or target_uuid missing
        - same_uuid            — source == target
        - uuid_not_found       — either UUID isn't in the gallery
        """
        if self.gallery is None:
            return {"error": "recognition_disabled"}

        payload = payload or {}
        source_uuid = str(payload.get("source_uuid", "")).strip()
        target_uuid = str(payload.get("target_uuid", "")).strip()
        confirmed_by = str(payload.get("confirmed_by", "")).strip()

        if not source_uuid or not target_uuid:
            return {
                "error": "missing_uuids",
                "detail": "both source_uuid and target_uuid required",
            }
        if source_uuid == target_uuid:
            return {"error": "same_uuid"}

        src_rec = self.gallery.get_record(source_uuid)
        tgt_rec = self.gallery.get_record(target_uuid)
        if src_rec is None:
            return {"error": "uuid_not_found", "uuid": source_uuid, "role": "source"}
        if tgt_rec is None:
            return {"error": "uuid_not_found", "uuid": target_uuid, "role": "target"}

        # Pre-merge sim (informational — we do NOT gate on it; the caller
        # already confirmed they want to merge).
        pre_sim = 0.0
        sc = src_rec.centroid_normalized
        tc = tgt_rec.centroid_normalized
        if sc is not None and tc is not None:
            pre_sim = float(sc @ tc)

        def _do():
            n = self.gallery.merge_into(source_uuid, target_uuid)
            feats, labels, uuids = self.gallery.get_centroids()
            with self.gal_lock:
                self.gal_state.gal_feats = feats
                self.gal_state.gal_labels = labels
                self.gal_state.gal_uuids = uuids
            return n, len(uuids)

        t0 = time.monotonic()
        try:
            samples_merged, n_id = self.run_job_sync(_do)
        except KeyError as e:
            return {"error": "uuid_not_found", "detail": str(e)}

        # Source is now in _trash — make it restorable via /gallery/restore
        self.last_soft_deleted = source_uuid

        self._audit_log(
            f"gallery_merge source={source_uuid} target={target_uuid} "
            f"samples={samples_merged} sim={pre_sim:.3f} "
            f"confirmed_by={confirmed_by or '<unspecified>'}"
        )

        return {
            "ok": True,
            "uuid": target_uuid,
            "merged_from": source_uuid,
            "samples_merged": int(samples_merged),
            "sim": round(pre_sim, 3),
            "identities": int(n_id),
            "took_sec": round(time.monotonic() - t0, 3),
        }

    # ==================================================================
    # /gallery/find_similar
    # ==================================================================

    def _handle_gallery_find_similar(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Find the top-K most similar gallery UUIDs to a given UUID's centroid.

        Use case: LLM sees an anonymous auto-enrolled UUID and wants to know
        "do we already have someone in the gallery who looks similar?"
        Returns sorted candidates with cosine sims so the LLM can decide
        whether to prompt the user for confirmation:

          "Are you Sean? You look familiar (we have a match at 0.48)."

        Payload
        -------
        uuid     : str (required)             — UUID to compare from
        top_k    : int (optional, default 3)  — number of candidates to return
        min_sim  : float (optional, default 0.0)  — filter out matches below this

        Returns
        -------
        ok      : bool
        uuid    : str                     — the query UUID (echo)
        matches : list of {uuid, name, sim, sample_count, source}
                  Sorted by sim descending. Excludes the query UUID itself.

        Note: returns 0 matches when gallery has only the query UUID, or
        when all other UUIDs fall below min_sim.
        """
        if self.gallery is None:
            return {"error": "recognition_disabled"}

        payload = payload or {}
        uuid = str(payload.get("uuid", "")).strip()
        try:
            top_k = int(payload.get("top_k", 3))
        except (TypeError, ValueError):
            return {"error": "bad_top_k"}
        try:
            min_sim = float(payload.get("min_sim", 0.0))
        except (TypeError, ValueError):
            return {"error": "bad_min_sim"}

        if not uuid:
            return {"error": "missing_uuid"}
        if top_k < 1:
            return {"error": "bad_top_k", "detail": "top_k must be >= 1"}

        query_rec = self.gallery.get_record(uuid)
        if query_rec is None:
            return {"error": "uuid_not_found", "uuid": uuid}

        query_centroid = query_rec.centroid_normalized
        if query_centroid is None:
            return {
                "error": "no_centroid",
                "detail": "query UUID has no embeddings",
                "uuid": uuid,
            }

        # Compare against every other UUID. O(N) over gallery — fine for
        # demo-scale (~50-100 identities). For larger galleries this could
        # be vectorized via get_centroids() once, but not needed now.
        all_records = self.gallery.list_identities(include_unnamed=True)
        candidates: List[Dict[str, Any]] = []
        for rec_dict in all_records:
            cand_uuid = rec_dict["uuid"]
            if cand_uuid == uuid:
                continue
            cand_rec = self.gallery.get_record(cand_uuid)
            if cand_rec is None or cand_rec.centroid_normalized is None:
                continue
            sim = float(query_centroid @ cand_rec.centroid_normalized)
            if sim < min_sim:
                continue
            candidates.append(
                {
                    "uuid": cand_uuid,
                    "name": cand_rec.name,
                    "sim": round(sim, 3),
                    "sample_count": cand_rec.sample_count,
                    "source": cand_rec.metadata.get("source", ""),
                }
            )

        # Sort by sim descending, take top_k
        candidates.sort(key=lambda c: c["sim"], reverse=True)
        candidates = candidates[:top_k]

        return {
            "ok": True,
            "uuid": uuid,
            "matches": candidates,
        }

    # ==================================================================
    # /speaking — vision-only "who is speaking" (VVAD)
    # ==================================================================
    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, it1 = max(ax1, bx1), max(ay1, by1)
        ix2, it2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, it2 - it1)
        inter = iw * ih
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return float(inter / ua) if ua > 0 else 0.0

    def _handle_speaking(self, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve who was speaking over an utterance window (DETERMINISTIC).

        payload (all optional):
          win_start_ms, win_end_ms : epoch-ms span of the utterance (preferred,
                                     sent by the ASR side). Scored via slide+max.
          lookback_sec             : if no window given, score the most-recent
                                     N seconds instead (default 4.0).

        Rule (in code, never the LLM): among faces whose VVAD score over the
        window is >= threshold, the speaker is the LARGEST bbox. The chosen
        speaker is LATCHED so the next /selfie binds to that exact face.

        Returns {speaker: {track_id,name,uuid,score}|None, faces:[...], shadow}.
        """
        if self.vvad is None:
            return {"error": "vvad_disabled"}
        payload = payload or {}
        with self.frame_lock:
            faces = list(self.frame_state.current_faces)
        if not faces:
            return {"speaker": None, "faces": []}
        faces.sort(key=lambda f: f.get("area", 0), reverse=True)  # largest first
        tids = [int(f["track_id"]) for f in faces]

        ws, we = payload.get("win_start_ms"), payload.get("win_end_ms")
        if ws is not None and we is not None:
            scores = self.vvad.resolve_window(
                tids, float(ws) / 1000.0, float(we) / 1000.0
            )
            mode = "window"
        else:
            lookback = float(payload.get("lookback_sec", 4.0))
            now = time.time()
            scores = self.vvad.resolve_window(tids, now - lookback, now)
            mode = "lookback"

        thr = self.vvad.speaking_thr
        speaking = [f for f in faces if scores.get(int(f["track_id"]), 0.0) >= thr]
        chosen = speaking[0] if speaking else None  # largest among speakers

        # on-screen badge: chosen = green, other speakers = amber
        try:
            from . import vvad_overlay

            vvad_overlay.flag(
                [
                    (
                        tuple(f["bbox"]),
                        scores.get(int(f["track_id"]), 0.0) >= thr,
                        scores.get(int(f["track_id"]), 0.0),
                    )
                    for f in faces
                ],
                ttl=1.6,
            )
        except Exception:
            pass

        # latch the chosen speaker for the upcoming /selfie
        if chosen is not None:
            self._speaking_latch = {
                "track_id": int(chosen["track_id"]),
                "uuid": chosen.get("uuid"),
                "name": chosen.get("name") or "",
                "bbox": tuple(chosen["bbox"]),
                "score": round(scores.get(int(chosen["track_id"]), 0.0), 3),
                "ts": time.time(),
            }

        logger.info(
            "VVAD /speaking (%s): scores=%s speaking=%s chosen=%s",
            mode,
            {k: round(v, 3) for k, v in scores.items()},
            [f["track_id"] for f in speaking],
            chosen["track_id"] if chosen else None,
        )

        def _face_out(f):
            sc = scores.get(int(f["track_id"]), 0.0)
            return {
                "track_id": int(f["track_id"]),
                "name": f.get("name") or "",
                "uuid": f.get("uuid"),
                "score": round(sc, 3),
                "speaking": sc >= thr,
                "area": f.get("area", 0),
            }

        speaker = None
        if chosen is not None:
            speaker = {
                "track_id": int(chosen["track_id"]),
                "name": chosen.get("name") or "",
                "uuid": chosen.get("uuid"),
                "score": round(scores.get(int(chosen["track_id"]), 0.0), 3),
            }
        return {
            "speaker": speaker,
            "faces": [_face_out(f) for f in faces],
            "shadow": self.vvad_shadow,
        }

    def _speaking_bbox_for_selfie(self):
        """Speaking bbox to bias /selfie toward, or None (=> largest face).

        Prefers the speaker LATCHED by a recent /speaking call (bound to the
        real utterance window — the correct moment). Falls back to resolving
        'now', then to None (largest face). Shadow mode never biases enroll,
        but still draws badges.
        """
        if self.vvad is None or self.vvad_shadow:
            self._resolve_speaking_bbox()  # badges only
            return None
        latch = self._speaking_latch
        if latch is not None and (time.time() - latch["ts"]) <= self.speaking_latch_ttl:
            with self.frame_lock:
                faces = list(self.frame_state.current_faces)
            for f in faces:  # only trust it if still visible
                if int(f["track_id"]) == latch["track_id"]:
                    try:
                        from . import vvad_overlay

                        vvad_overlay.flag(
                            [(tuple(f["bbox"]), True, latch.get("score"))], ttl=1.6
                        )
                    except Exception:
                        pass
                    return tuple(f["bbox"])  # current bbox of the latched track
            # latched track no longer visible -> fall through
        return self._resolve_speaking_bbox()  # resolve 'now' (also badges)

    def _resolve_speaking_bbox(self):
        """Bbox of the chosen speaking face, or None (=> caller uses largest).

        Among visible faces speaking (score >= threshold) pick the LARGEST bbox;
        if none, return None (fallback to overall largest). Also pushes a brief
        on-screen "speaking" badge for ALL speakers (chosen = green, others =
        amber), shown for ~1.6s — even in shadow mode.
        """
        if self.vvad is None:
            return None
        with self.frame_lock:
            faces = list(self.frame_state.current_faces)
        if not faces:
            return None
        faces.sort(key=lambda f: f.get("area", 0), reverse=True)  # largest first
        _, scores = self.vvad.resolve_speaking([int(f["track_id"]) for f in faces])
        thr = self.vvad.speaking_thr
        speaking = [f for f in faces if scores.get(int(f["track_id"]), 0.0) >= thr]
        logger.info(
            "VVAD: scores=%s speaking=%s largest_tid=%s shadow=%s",
            {k: round(v, 3) for k, v in scores.items()},
            [f["track_id"] for f in speaking],
            faces[0].get("track_id"),
            self.vvad_shadow,
        )

        # visualization badge (chosen speaker = green, others = amber)
        chosen_tid = (
            max(speaking, key=lambda f: scores.get(int(f["track_id"]), 0.0))["track_id"]
            if speaking
            else None
        )
        try:
            from . import vvad_overlay

            vvad_overlay.flag(
                [
                    (
                        tuple(f["bbox"]),
                        f["track_id"] == chosen_tid,
                        scores.get(int(f["track_id"]), 0.0),
                    )
                    for f in speaking
                ],
                ttl=1.6,
            )
        except Exception:
            pass

        if self.vvad_shadow:
            return None  # shadow mode: never bias enroll
        if speaking:
            # someone speaking -> the one MOST likely speaking (highest score),
            # not merely the biggest face
            best = max(speaking, key=lambda f: scores.get(int(f["track_id"]), 0.0))
            return tuple(best["bbox"])
        return tuple(faces[0]["bbox"])  # nobody speaking -> largest face

    # ==================================================================
    # /gallery/find_similar_current — no-UUID variant for LLM ergonomics
    # ==================================================================

    def _get_current_largest_face_uuid(self) -> Optional[Dict[str, Any]]:
        """Return ``{uuid, name, tier, area, track_id}`` for the most prominent
        currently-visible face that has a UUID, or None.

        "Most prominent" = largest bbox area (closest to camera). Faces
        without a UUID (e.g. transient "unknown" still being recognized)
        are skipped — they have no identity for the LLM to act on.

        Acquires self.frame_lock for a consistent snapshot.
        """
        with self.frame_lock:
            faces = list(self.frame_state.current_faces)

        # current_faces is already sorted by area desc, but defensive
        faces.sort(key=lambda f: f.get("area", 0), reverse=True)
        for f in faces:
            uuid = f.get("uuid")
            if uuid:
                return f
        return None

    def _handle_gallery_find_similar_current(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find similar UUIDs to the currently-largest visible face.

        Wraps :meth:`_handle_gallery_find_similar` but resolves the query
        UUID server-side from the visible scene rather than requiring
        the LLM to pass one. This is the recommended entry point for
        LLM-driven "do you remember me?" interactions.

        Payload
        -------
        top_k    : int (optional, default 3)
        min_sim  : float (optional, default 0.0)

        Returns
        -------
        Either the find_similar response (with ``query_face`` describing
        which face was used) or ``{"error": "no_visible_face"}``.
        """
        face = self._get_current_largest_face_uuid()
        if face is None:
            return {"error": "no_visible_face"}

        # Delegate to the UUID-based handler. Echo back the query face
        # info so the LLM (and Go connector) can log/display it.
        sub_payload = dict(payload or {})
        sub_payload["uuid"] = face["uuid"]
        result = self._handle_gallery_find_similar(sub_payload)
        if "ok" in result and result["ok"]:
            result["query_face"] = {
                "uuid": face.get("uuid"),
                "name": face.get("name"),
                "track_id": face.get("track_id"),
                "tier": face.get("tier"),
            }
        return result

    # ==================================================================
    # /gallery/merge_current — no-UUID variant for LLM ergonomics
    # ==================================================================

    def _handle_gallery_merge_current(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Merge current visible face into the same-name UUID best matching it.

        Server-side disambiguation: the LLM gives a target NAME (e.g.
        "sean") and the server picks the source (current largest visible
        face) and target (UUID with that name closest to the source by
        centroid sim).

        Workflow this enables (the LLM doesn't see any UUIDs)::

            User: "Are you Sean?" → "Yes"
            LLM: gallery_merge_current(target_name="sean", confirmed_by="user_voice")
            Server picks source = current anon UUID + target = best "sean".

        Payload
        -------
        target_name  : str (required)  — display name the visitor confirmed
        confirmed_by : str (optional)  — audit string (e.g. "user_voice")
        min_sim      : float (optional, default 0.0) — if multiple "sean"
                       UUIDs exist, require best one's sim ≥ this to
                       avoid merging into the wrong same-name UUID

        Returns
        -------
        Same shape as /gallery/merge on success. New error codes:
        - no_visible_face          — nothing on screen with a UUID
        - source_is_named          — current face already has a name; the
                                     LLM probably called this by mistake
        - target_name_not_found    — no gallery UUID has that name
        - ambiguous_target         — same-name UUIDs exist but best sim
                                     is below min_sim (and N > 1)
        """
        if self.gallery is None:
            return {"error": "recognition_disabled"}

        payload = payload or {}
        target_name = str(payload.get("target_name", "")).strip().lower()
        confirmed_by = str(payload.get("confirmed_by", "")).strip()
        try:
            min_sim = float(payload.get("min_sim", 0.0))
        except (TypeError, ValueError):
            return {"error": "bad_min_sim"}

        if not target_name:
            return {"error": "missing_target_name"}

        # 1. Resolve source = current largest visible face with UUID
        face = self._get_current_largest_face_uuid()
        if face is None:
            return {"error": "no_visible_face"}
        source_uuid = face["uuid"]
        source_name = face.get("name") or ""

        # Defensive: if the visible face is already named, the LLM is
        # probably confused (you don't merge a named face). Allow it
        # only if the source name happens to equal target_name (no-op
        # against the same-name UUID — fall through to no-op below).
        if source_name and not source_name.startswith("anon_"):
            if source_name.lower() != target_name:
                return {
                    "error": "source_is_named",
                    "source_name": source_name,
                    "detail": "current visible face is already named differently",
                }

        # 2. Resolve target: UUIDs with name == target_name, pick best sim
        src_rec = self.gallery.get_record(source_uuid)
        if src_rec is None:
            return {"error": "uuid_not_found", "uuid": source_uuid, "role": "source"}
        src_centroid = src_rec.centroid_normalized
        if src_centroid is None:
            return {"error": "no_centroid", "uuid": source_uuid, "role": "source"}

        # Iterate active gallery for same-name candidates
        same_name_uuids: List[Tuple[str, float, int]] = []
        for rec_dict in self.gallery.list_identities(include_unnamed=False):
            cand_uuid = rec_dict["uuid"]
            if cand_uuid == source_uuid:
                continue
            if (rec_dict.get("name") or "").lower() != target_name:
                continue
            cand_rec = self.gallery.get_record(cand_uuid)
            if cand_rec is None or cand_rec.centroid_normalized is None:
                continue
            sim = float(src_centroid @ cand_rec.centroid_normalized)
            same_name_uuids.append((cand_uuid, sim, cand_rec.sample_count))

        if not same_name_uuids:
            return {"error": "target_name_not_found", "target_name": target_name}

        # Sort by sim desc; best candidate first
        same_name_uuids.sort(key=lambda t: t[1], reverse=True)
        best_uuid, best_sim, _ = same_name_uuids[0]

        # If there's more than one same-name UUID and the top sim is low,
        # don't guess — surface ambiguity to caller.
        if len(same_name_uuids) > 1 and best_sim < min_sim:
            return {
                "error": "ambiguous_target",
                "target_name": target_name,
                "candidates": [
                    {"uuid": u, "sim": round(s, 3), "samples": c}
                    for u, s, c in same_name_uuids[:5]
                ],
                "detail": (
                    f"{len(same_name_uuids)} candidates for name {target_name!r}; "
                    f"best sim {best_sim:.3f} < min_sim {min_sim:.3f}"
                ),
            }

        # 3. Delegate to the standard merge handler with resolved UUIDs.
        # Compose a richer audit string so the log shows what was picked.
        audit = (
            f"{confirmed_by or '<unspecified>'}; "
            f"resolved via merge_current target_name={target_name} "
            f"best_sim={best_sim:.3f}"
        )
        sub_payload = {
            "source_uuid": source_uuid,
            "target_uuid": best_uuid,
            "confirmed_by": audit,
        }
        result = self._handle_gallery_merge(sub_payload)
        # Annotate result with the resolution info the LLM/Go side
        # might want to surface to the user.
        if isinstance(result, dict) and result.get("ok"):
            result["target_name"] = target_name
            result["resolved_by"] = "merge_current"
        return result

    # ==================================================================
    # /gallery/sweep_stale — long-term cleanup
    # ==================================================================

    def _handle_gallery_sweep_stale(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Identify (and optionally soft-delete) UUIDs not seen recently.

        For long-running deployments where the gallery accumulates many
        identities — especially anon UUIDs from passers-by who never
        introduced themselves. Disk math: 50 samples × ~50KB = 2.5MB
        per UUID, so 1000 unused UUIDs = 2.5GB. This endpoint lets the
        operator (or a cron job) reclaim space.

        Payload
        -------
        max_age_sec   : float (required) — UUIDs with last_seen_at older
                                           than this many seconds become
                                           candidates. E.g. 30 days =
                                           2_592_000.
        dry_run       : bool (default True) — if True, list candidates
                                              without deleting. Set
                                              False to actually move
                                              them to _trash.
        protect_named : bool (default True) — if True, skip UUIDs that
                                              have been named (only sweep
                                              anon_xxx). Strong default;
                                              cleaning up a named
                                              identity by accident is
                                              hard to recover from.

        Returns
        -------
        ok           : bool
        dry_run      : bool
        candidates   : list of {uuid, name, last_seen_at, sample_count,
                                action: "deleted"|"would_delete"|"skipped_named"}
        deleted      : int — number actually soft-deleted (0 if dry_run)
        identities   : int — total active gallery size after sweep
        """
        if self.gallery is None:
            return {"error": "recognition_disabled"}

        payload = payload or {}
        try:
            max_age_sec = float(payload.get("max_age_sec", 0))
        except (TypeError, ValueError):
            return {"error": "bad_max_age_sec"}
        if max_age_sec <= 0:
            return {
                "error": "bad_max_age_sec",
                "detail": "max_age_sec must be > 0",
            }
        dry_run = bool(payload.get("dry_run", True))
        protect_named = bool(payload.get("protect_named", True))

        stale_uuids = self.gallery.list_stale(max_age_sec)
        candidates: List[Dict[str, Any]] = []
        deleted = 0

        for uuid in stale_uuids:
            rec = self.gallery.get_record(uuid)
            if rec is None:
                continue
            entry = {
                "uuid": uuid,
                "name": rec.name,
                "last_seen_at": rec.metadata.get("last_seen_at"),
                "sample_count": rec.sample_count,
            }
            # Skip named identities if protect_named is set
            if protect_named and rec.name and not rec.name.startswith("anon_"):
                entry["action"] = "skipped_named"
                candidates.append(entry)
                continue
            if dry_run:
                entry["action"] = "would_delete"
            else:
                # Actual soft-delete via gallery.forget (moves to _trash)
                ok = self.gallery.forget(uuid)
                if ok:
                    deleted += 1
                    entry["action"] = "deleted"
                else:
                    entry["action"] = "delete_failed"
            candidates.append(entry)

        # Refresh in-memory gallery snapshot used by recognition
        if not dry_run and deleted > 0:
            feats, labels, uuids = self.gallery.get_centroids()
            with self.gal_lock:
                self.gal_state.gal_feats = feats
                self.gal_state.gal_labels = labels
                self.gal_state.gal_uuids = uuids

        self._audit_log(
            f"gallery_sweep_stale max_age={max_age_sec:.0f}s "
            f"dry_run={dry_run} protect_named={protect_named} "
            f"candidates={len(candidates)} deleted={deleted}"
        )

        return {
            "ok": True,
            "dry_run": dry_run,
            "protect_named": protect_named,
            "max_age_sec": max_age_sec,
            "candidates": candidates,
            "deleted": deleted,
            "identities": len(self.gallery.list_identities(include_unnamed=True)),
        }

    # ==================================================================
    # Helpers
    # ==================================================================

    def _audit_log(self, msg: str) -> None:
        """Append a timestamped line to gallery_dir/corrections.log."""
        if not self.gallery_dir:
            return
        log_path = osp.join(self.gallery_dir, "corrections.log")
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            with self.audit_lock:
                with open(log_path, "a") as f:
                    f.write(f"{ts} {msg}\n")
        except Exception as e:
            self.log.warning("audit log failed: %s", e)

    @staticmethod
    def _load_112(
        image_path: Optional[str], image_b64: Optional[str]
    ) -> Optional[np.ndarray]:
        """Load an image from path or base64 and ensure size is 112×112."""
        img = None
        if image_path:
            img = cv2.imread(image_path)
        elif image_b64:
            try:
                raw = base64.b64decode(image_b64)
                data = np.frombuffer(raw, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception:
                img = None
        if img is None:
            return None
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        return img
