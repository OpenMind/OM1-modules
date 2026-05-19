#!/usr/bin/env python3
"""
HTTP control plane for the realtime face pipeline.

This module exposes a thin, thread-safe HTTP API that lets external tools
query presence (/who), adjust runtime configuration (/config), and manage the
face gallery (/gallery/refresh, /gallery/add_aligned, /gallery/add_raw, /selfie)
while the main video loop keeps running.

Notes
-----
- The HTTP server runs on its own thread. Handlers in this module never call
  CUDA/TensorRT directly; instead they use `run_job_sync(fn)` to enqueue work
  onto the main thread's job queue so GPU/model operations are serialized.
- Shared state (config, gallery embeddings/labels, last clean frame) is protected
  by locks supplied by the caller.
- This class does not bind a socket by itself; the caller typically registers
  `HttpAPI._handle` with a lightweight `Server` instance.
"""

from __future__ import annotations

import base64
import logging
import os
import os.path as osp
import shutil
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from . import selfie_logic as sl
from .adaface import warp_face_by_5p


class HttpAPI:
    """HTTP endpoint handlers for control and gallery operations.

    Parameters
    ----------
    who : WhoTracker
        Presence tracker; used to answer `/who` queries.
    scrfd : TRTSCRFD
        Detector reference; used for hot-applying NMS threshold (from `/config`).
    gm : GalleryManager or None
        Gallery manager for alignment/embedding. If None, recognition endpoints
        will return `{ "error": "recognition disabled" }`.
    gallery_dir : str
        Filesystem root of the gallery.
    gal_state : Any
        Mutable object holding in-memory gallery features and labels.
        Expected attributes: `gal_feats: np.ndarray|None`, `gal_labels: list[str]`.
    gal_lock : threading.Lock
        Lock guarding `gal_state` updates.
    cfg : dict[str, object]
        Live runtime configuration (blur mode, thresholds, etc.).
    cfg_lock : threading.Lock
        Lock guarding `cfg` reads/writes.
    frame_state : Any
        Holder for the last clean frame and detections/landmarks captured
        by the main loop. Expected attributes: `frame_bgr`, `dets`, `kpss`.
    frame_lock : threading.Lock
        Lock guarding `frame_state` reads/writes.
    run_job_sync : Callable[[Callable[[], Any]], Any]
        Enqueue-and-wait helper that runs the given function on the main
        thread (via the job queue) and returns its result.
    logger : logging.Logger, optional
        Logger to use; if omitted, a default `http_api` logger is created.
    """

    def __init__(
        self,
        *,
        who,
        scrfd,
        gm,
        gallery_dir: str,
        gal_state,
        gal_lock: threading.Lock,
        cfg: Dict[str, object],
        cfg_lock: threading.Lock,
        frame_state,
        frame_lock: threading.Lock,
        run_job_sync: Callable[[Callable[[], Any]], Any],
        logger: Optional[logging.Logger] = None,
        face_tracker=None,
    ):
        """Initialize the HTTP API wrapper."""
        self.who = who
        self.scrfd = scrfd
        self.gm = gm
        self.gallery_dir = gallery_dir
        self.gal_state = gal_state
        self.gal_lock = gal_lock
        self.cfg = cfg
        self.cfg_lock = cfg_lock
        self.frame_state = frame_state
        self.frame_lock = frame_lock
        self.run_job_sync = run_job_sync
        self.log = logger or logging.getLogger("http_api")
        self.server = None
        self.face_tracker = face_tracker

        # Selfie pipeline state
        self.selfie_lock = threading.Lock()  # serialize concurrent /selfie
        self.enrollment_lock = threading.Lock()  # protect last_enrollment dict
        self.audit_lock = threading.Lock()  # serialize corrections.log writes
        self.last_enrollment: Optional[Dict[str, Any]] = None

    def stop(self) -> None:
        """Stop an attached HTTP server if present."""
        try:
            if self.server:
                self.server.stop()
        except Exception:
            pass

    def _handle(self, payload: Dict[str, Any], path: str) -> Dict[str, Any]:
        """Dispatch a POST request to the appropriate handler.

        Parameters
        ----------
        payload : dict
            JSON body of the request (may be empty for some endpoints).
        path : str
            Request path, e.g. `/who`, `/config`, `/gallery/refresh`.

        Returns
        -------
        dict
            JSON-serializable response. On errors, returns `{ "error": <msg> }`.

        Endpoints
        ---------
        - `/ping`                → `{ "ok": true }`
        - `/ts`                  → `{ "ts": <unix_seconds> }`
        - `/who`                 → presence snapshot for `recent_sec`
        - `/config`              → get/set live config
        - `/gallery/refresh`     → incremental align+embed and update centroids
        - `/gallery/add_aligned` → add a 112×112 aligned crop for an identity
        - `/gallery/add_raw`     → copy a raw image and refresh gallery
        - `/selfie`              → enroll from the last clean frame (aligned only)
        """
        try:
            if path == "/ping":
                return {"ok": True}

            if path == "/ts":
                return {"ts": time.time()}

            if path == "/who":
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
                    result["faces"] = faces
                    if unknowns:
                        result["unknown_captures"] = unknowns

                if frm is not None:
                    _, buf = cv2.imencode(".jpg", frm, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    result["frame_b64"] = base64.b64encode(buf.tobytes()).decode(
                        "ascii"
                    )
                    result["frame_hw"] = list(frm.shape[:2])
                    result["frame_ts"] = time.time()
                else:
                    result["frame_b64"] = None
                    result["frame_hw"] = None
                    result["frame_ts"] = None

                return result

            if path == "/config":
                return self._handle_config(payload)

            if path == "/gallery/refresh":
                return self._handle_gallery_refresh()

            if path == "/gallery/add_aligned":
                return self._handle_gallery_add_aligned(payload)

            if path == "/gallery/add_raw":
                return self._handle_gallery_add_raw(payload)

            if path == "/selfie":
                return self._handle_selfie(payload)

            if path == "/gallery/delete":
                return self._handle_gallery_delete(payload)

            if path in ("/gallery/identities", "/gallery/list_identities"):
                return self._handle_gallery_identities()

            if path == "/gallery/move_samples":
                return self._handle_gallery_move_samples(payload)

            if path == "/gallery/forget_last":
                return self._handle_gallery_forget_last(payload)

            return {"error": f"unknown path {path}"}
        except Exception as e:
            self.log.exception("HTTP error")
            return {"error": str(e)}

    def _handle_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get or set live runtime configuration.

        Parameters
        ----------
        payload : dict
            Either `{"get": true}` to fetch config or `{"set": {...}}` to update
            a subset of known keys.

        Returns
        -------
        dict
            - For get: `{ "config": { ... } }`
            - For set: `{ "ok": true, "changed": { ... } }`
            - On misuse: `{ "error": "...", "config_keys": [...] }`

        Notes
        -----
        Updates are applied under `cfg_lock`. If the `nms` key is changed and a
        detector is available, `scrfd.nms_thresh` is updated immediately.
        """
        if payload and payload.get("get"):
            with self.cfg_lock:
                return {"config": self.cfg}

        if payload and "set" in payload and isinstance(payload["set"], dict):
            changed = {}
            with self.cfg_lock:
                for k, v in payload["set"].items():
                    if k in self.cfg:
                        self.cfg[k] = v
                        changed[k] = v
                # hot-apply SCRFD NMS if changed
                try:
                    self.scrfd.nms_thresh = float(self.cfg["nms"])  # type: ignore
                except Exception:
                    pass
            return {"ok": True, "changed": changed}

        return {
            "error": "use {'get':true} or {'set':{...}}",
            "config_keys": list(self.cfg.keys()),
        }

    def _handle_gallery_refresh(self) -> Dict[str, Any]:
        """Incrementally process new images and update in-memory centroids.

        Returns
        -------
        dict
            `{ "ok": true, "identities": <int>, "aligned_added": <int>,
               "vectors_added": <int>, "took_sec": <float> }`
            or `{ "error": "recognition disabled" }` if `gm` is not set.
        """
        if not self.gm:
            return {"error": "recognition disabled"}
        t0 = time.time()

        def _do():
            a_add, v_add = self.gm.refresh(process_raw=True)
            feats, labels = self.gm.get_identity_means()
            with self.gal_lock:
                self.gal_state.gal_feats, self.gal_state.gal_labels = feats, labels
            return a_add, v_add, len(labels)

        a_add, v_add, n_id = self.run_job_sync(_do)
        return {
            "ok": True,
            "identities": int(n_id),
            "aligned_added": int(a_add),
            "vectors_added": int(v_add),
            "took_sec": round(time.time() - t0, 3),
        }

    # -------------------- /gallery/add_aligned --------------------- #
    def _handle_gallery_add_aligned(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Add a 112×112 aligned face to the gallery and update centroids.

        Parameters
        ----------
        payload : dict
            Must include `"id"` (identity label). Provide either
            `"image_path"` or `"image_b64"`.

        Returns
        -------
        dict
            On success: `{ "ok": true, "added": <rel_path>, "identities": <int> }`.
            On failure: `{ "error": "...") }` or `{ "ok": false, "error": "..." }`.
        """
        if not self.gm:
            return {"error": "recognition disabled"}
        if not payload or "id" not in payload:
            return {"error": "missing 'id'"}
        person = str(payload["id"])
        image_path = payload.get("image_path")
        image_b64 = payload.get("image_b64")
        if not image_path and not image_b64:
            return {"error": "provide 'image_path' or 'image_b64'"}

        img112 = self._load_112(image_path, image_b64)
        if img112 is None:
            return {"ok": False, "error": "failed to load image"}
        fname_hint = os.path.basename(image_path) if image_path else None

        def _do_add():
            rel = self.gm.add_aligned_snapshot(person, img112, fname_hint=fname_hint)
            feats, labels = self.gm.get_identity_means()
            with self.gal_lock:
                self.gal_state.gal_feats, self.gal_state.gal_labels = feats, labels
            return rel, len(labels)

        rel, n_id = self.run_job_sync(_do_add)
        return {"ok": True, "added": rel, "identities": int(n_id)}

    def _handle_gallery_add_raw(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Copy a raw image into gallery/<id>/raw and run full refresh (align+embed).

        Upload pathway -> RAW only. Alignment happens once during refresh.
        Response: { ok, saved, identities }.

        Parameters
        ----------
        payload : dict
            Must include `"id"` (identity label) and `"image_path"` (absolute or
            relative path to an existing image).

        Returns
        -------
        dict
            `{ "ok": true, "saved": <abs_path>, "identities": <int> }`
            or an error object.

        Notes
        -----
        After copying, this calls `gm.refresh(process_raw=True)` via
        `run_job_sync` to align new raw images and append embeddings.
        """
        if not self.gm:
            return {"error": "recognition disabled"}
        if not payload or "id" not in payload or "image_path" not in payload:
            return {"error": "need 'id' and 'image_path'"}

        person = str(payload["id"])
        src = str(payload["image_path"])
        if not os.path.exists(src):
            return {"error": f"image_path not found: {src}"}

        raw_dir = os.path.join(self.gallery_dir, person, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        dst = os.path.join(raw_dir, os.path.basename(src))
        shutil.copy2(src, dst)

        def _do_refresh():
            _a_added, _v_added = self.gm.refresh(process_raw=True)
            feats, labels = self.gm.get_identity_means()
            with self.gal_lock:
                self.gal_state.gal_feats, self.gal_state.gal_labels = feats, labels
            return len(labels)

        n_id = self.run_job_sync(_do_refresh)
        return {"ok": True, "saved": dst, "identities": int(n_id)}

    def _handle_selfie(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Multi-frame selfie enrollment with score-based target selection,
        dedup, identity-consistency check, and adaptive sample count.

        Payload
        -------
        id    : str  (required) — requested identity name (lowercase, alnum/_/-)
        force : bool (optional) — bypass cross-name reject (twins/lookalikes)

        Returns (success)
        -----------------
        { ok:True, id, merged, forced, samples_saved, match_sim, identities }

        Errors
        ------
        bad_id | recognition_disabled | busy | no_valid_frames |
        insufficient_samples | ambiguous_subjects | face_belongs_to
        """
        if not self.gm:
            return {"error": "recognition_disabled"}

        payload = payload or {}
        requested = str(payload.get("id", "")).strip().lower()
        ok, reason = sl.validate_identity_name(requested)
        if not ok:
            return {"error": "bad_id", "detail": reason}

        force = bool(payload.get("force", False))

        # Serialize concurrent /selfie calls — gallery + last_enrollment can't
        # tolerate interleaved enrollment from two callers at once.
        if not self.selfie_lock.acquire(blocking=False):
            return {"error": "busy"}

        try:
            return self._do_selfie(requested, force)
        finally:
            self.selfie_lock.release()

    def _do_selfie(self, requested: str, force: bool) -> Dict[str, Any]:
        """Core selfie collection loop. Caller holds self.selfie_lock."""
        # Snapshot cfg once so thresholds stay consistent for this request
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
        target_id: Optional[str] = None
        merged_flag = False
        match_label: Optional[str] = None
        match_sim_seen = 0.0
        ambiguity_streak = 0
        last_seen_token = -1  # frame-change detection (see below)

        # ----- Phase 2: Collection loop -----
        while len(crops) < max_samples and time.monotonic() < deadline:
            # (A) Snapshot frame_state under lock, release fast
            with self.frame_lock:
                if self.frame_state.frame_bgr is None:
                    frm = None
                    dets = kpss = None
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

            # Prefer explicit last_ts if run.py is up-to-date. If it's 0.0
            # (older run.py without the field), fall back to id(frm) — the
            # array's memory address changes per .copy(), so consecutive
            # snapshots produce distinct tokens.
            frame_token = ts if ts > 0 else (id(frm) if frm is not None else 0)

            if frm is None or frame_token == last_seen_token:
                time.sleep(0.02)
                continue
            last_seen_token = frame_token

            # (B) No detections this frame
            if dets is None or len(dets) == 0:
                continue

            # (C) Score every face: (bbox_area / frame_area) * frontality
            h, w = frm.shape[:2]
            frame_area = float(h * w)
            scores = [
                sl.score_face(
                    dets[i],
                    kpss[i] if kpss is not None and i < len(kpss) else None,
                    frame_area,
                )
                for i in range(len(dets))
            ]

            # (D) Ambiguity — fail fast on 2 consecutive ambiguous frames
            is_amb, top_ratio = sl.check_ambiguity(scores, ambig_ratio, score_floor)
            if is_amb:
                ambiguity_streak += 1
                if ambiguity_streak >= 2:
                    self.log.info(
                        "selfie ambiguous: ratio=%.3f n_engaged=%d",
                        top_ratio,
                        sum(1 for s in scores if s >= score_floor),
                    )
                    return {
                        "error": "ambiguous_subjects",
                        "top_ratio": round(top_ratio, 3),
                        "n_engaged": sum(1 for s in scores if s >= score_floor),
                    }
                continue
            ambiguity_streak = 0  # reset on any clean frame

            # (E) Pick target & engagement floor
            target_idx = int(np.argmax(scores))
            if scores[target_idx] < score_floor:
                continue

            # (F) Need landmarks for alignment
            if kpss is None or target_idx >= len(kpss):
                continue
            target_kps = kpss[target_idx]

            # (G) Build aligned 112×112 crop + quality gate
            crop = warp_face_by_5p(frm, target_kps, 112)
            ok_q, q_reason = sl.quality_check(
                crop, dets[target_idx], target_kps, cfg_snap
            )
            if not ok_q:
                self.log.debug("selfie frame rejected: %s", q_reason)
                continue

            # (H) Embed on main thread (GPU)
            v = self.run_job_sync(lambda c=crop: self.gm.embed_aligned(c))

            # (I) First valid frame: dedup. Subsequent: consistency + novelty.
            if target_id is None:
                with self.gal_lock:
                    feats = self.gal_state.gal_feats
                    labels = list(self.gal_state.gal_labels)

                decision = sl.resolve_target_id(
                    v,
                    requested,
                    feats,
                    labels,
                    cfg_snap,
                    force=force,
                )

                if decision.reject:
                    self.log.info(
                        "selfie face_belongs_to label=%s sim=%.3f",
                        decision.match_label,
                        decision.match_sim,
                    )
                    return {
                        "error": "face_belongs_to",
                        "name": decision.match_label,
                        "sim": round(decision.match_sim, 3),
                    }

                target_id = decision.id
                merged_flag = decision.merged
                match_label = decision.match_label
                match_sim_seen = decision.match_sim
            else:
                sim = sl.cosine_to_mean(running_mean, v)
                if sim < consist_thr:
                    # Different person showed up mid-window
                    self.log.debug("selfie frame rejected: consistency=%.3f", sim)
                    continue
                if sim > novelty_thr:
                    # Near-duplicate of what we've already collected
                    self.log.debug("selfie frame rejected: not novel (sim=%.3f)", sim)
                    continue

            # (J) Accept the frame
            crops.append(crop)
            embs.append(v)
            running_mean = sl.update_running_mean(running_mean, v, len(embs))
            time.sleep(tap_interval)

        # ----- Phase 3: Finalize -----
        n = len(crops)
        if n == 0:
            return {"error": "no_valid_frames"}
        if n < min_samples:
            return {"error": "insufficient_samples", "got": n}

        ts_str = time.strftime("%Y-%m-%dT%H-%M-%S")

        # Single main-thread batch: save each crop with its pre-computed vec,
        # then recompute stats ONCE, then snapshot centroids into gal_state.
        def _do_save_batch():
            saved = []
            for k, (c, v) in enumerate(zip(crops, embs)):
                rel = self.gm.add_aligned_no_stats(
                    target_id,
                    c,
                    vec=v,
                    fname_hint=f"{ts_str}_{k:02d}.jpg",
                )
                saved.append(rel)
            self.gm.recompute_stats()
            feats, labels = self.gm.get_identity_means()
            with self.gal_lock:
                self.gal_state.gal_feats = feats
                self.gal_state.gal_labels = labels
            return saved, len(labels)

        saved_files, n_id = self.run_job_sync(_do_save_batch)

        # Record for /gallery/move_samples and /gallery/forget_last
        with self.enrollment_lock:
            self.last_enrollment = {
                "id": target_id,
                "files": saved_files,
                "monotonic_ts": time.monotonic(),
                "wall_ts": time.time(),
                "merged": merged_flag,
                "forced": force,
            }

        if force:
            self._audit_log(
                f"selfie forced id={target_id} samples={n} "
                f"matched={match_label or '-'} sim={match_sim_seen:.3f}"
            )
        else:
            self.log.info(
                "selfie ok id=%s merged=%s samples=%d",
                target_id,
                merged_flag,
                n,
            )

        return {
            "ok": True,
            "id": target_id,
            "merged": merged_flag,
            "forced": force,
            "samples_saved": n,
            "match_sim": round(match_sim_seen, 3),
            "identities": int(n_id),
        }

    def _handle_gallery_delete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Delete one or more identities and rebuild the embed store.

        Accepts:
        {"id": "alice"}
        {"ids": ["alice", "bob"]}
        {"id": ["alice", "bob"]}

        Returns (example):
        {
            "ok": true,
            "deleted": ["alice"],
            "failed": {"bob": "not found"},
            "removed_gallery": true,
            "identities": 0,
            "aligned_used": 3,
            "vectors_rebuilt": 3,
            "took_sec": 0.112
        }
        """
        if not self.gm:
            return {"error": "recognition disabled"}
        if not payload:
            return {"error": "missing 'id' or 'ids'"}

        # normalize to list[str]
        ids = []
        if "ids" in payload and isinstance(payload["ids"], list):
            ids = [str(x) for x in payload["ids"]]
        elif "id" in payload:
            if isinstance(payload["id"], list):
                ids = [str(x) for x in payload["id"]]
            else:
                ids = [str(payload["id"])]
        else:
            return {"error": "missing 'id' or 'ids'"}

        if not ids:
            return {"error": "no identities provided"}

        t0 = time.time()

        def _do():
            deleted: List[str] = []
            failed: Dict[str, str] = {}
            removed_any = False
            total_aligned = 0
            total_vectors = 0

            for person in ids:
                try:
                    removed_gallery, aligned_used, vectors_rebuilt, _n_id = (
                        self.gm.delete_identity(person)
                    )
                    if removed_gallery or aligned_used or vectors_rebuilt:
                        deleted.append(person)
                        removed_any = removed_any or bool(removed_gallery)
                        total_aligned += int(aligned_used)
                        total_vectors += int(vectors_rebuilt)
                    else:
                        # gm.delete_identity returned 0 changes: treat as not found
                        failed[person] = "not found or no changes"
                except Exception as e:
                    failed[person] = str(e)

            # always refresh means after batch
            feats, labels = self.gm.get_identity_means()
            with self.gal_lock:
                self.gal_state.gal_feats, self.gal_state.gal_labels = feats, labels

            return (
                deleted,
                failed,
                removed_any,
                total_aligned,
                total_vectors,
                len(labels),
            )

        deleted, failed, removed_any, total_aligned, total_vectors, n_id = (
            self.run_job_sync(_do)
        )
        return {
            "ok": len(failed) == 0,  # true only if all succeeded
            "deleted": deleted if len(deleted) != 1 else deleted[0],
            "failed": failed or None,
            "removed_gallery": bool(removed_any),
            "identities": int(n_id),
            "aligned_used": int(total_aligned),
            "vectors_rebuilt": int(total_vectors),
            "took_sec": round(time.time() - t0, 3),
        }

    def _handle_gallery_identities(self) -> Dict[str, Any]:
        """
        List identity folders under the gallery with lightweight counts.

        Returns
        -------
        dict
            {
            "ok": true,
            "total": <int>,                # number of identities
            "identities": [
                {"id": "Alice", "aligned": 12, "raw": 3},
                {"id": "Bob",   "aligned":  5, "raw": 0}
            ]
            }
        """

        def _count_images(dir_path: str) -> int:
            if not osp.isdir(dir_path):
                return 0
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            try:
                return sum(
                    1
                    for fn in os.listdir(dir_path)
                    if osp.splitext(fn)[1].lower() in exts
                )
            except Exception:
                return 0

        gallery_root = self.gallery_dir
        if not gallery_root or not osp.isdir(gallery_root):
            return {"ok": True, "total": 0, "identities": []}

        identities = []
        try:
            # List first-level folders as identities
            for name in sorted(os.listdir(gallery_root)):
                p = osp.join(gallery_root, name)
                if not osp.isdir(p):
                    continue
                aligned_dir = osp.join(p, "aligned")
                raw_dir = osp.join(p, "raw")
                identities.append(
                    {
                        "id": name,
                        "aligned": _count_images(aligned_dir),
                        "raw": _count_images(raw_dir),
                    }
                )
        except Exception as e:
            return {"ok": False, "error": str(e)}

        return {"ok": True, "total": len(identities), "identities": identities}

    # -------------------- /gallery/move_samples --------------------- #
    def _handle_gallery_move_samples(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move samples from one identity to another.

        Use cases:
        - Rename (typo correction): to_id doesn't exist → folder is renamed.
        - Merge into existing: to_id exists → samples merged in, from_id empty
          folder deleted.

        Payload
        -------
        from_id : str (required)
        to_id   : str (required)
        files   : List[str] (optional) — relative paths under gallery_dir.
                  If omitted, uses last_enrollment.files (60s TTL; id must match).
        """
        if not self.gm:
            return {"error": "recognition_disabled"}

        payload = payload or {}
        from_id = str(payload.get("from_id", "")).strip().lower()
        to_id = str(payload.get("to_id", "")).strip().lower()

        for label, name in (("from_id", from_id), ("to_id", to_id)):
            ok, reason = sl.validate_identity_name(name)
            if not ok:
                return {"error": "bad_id", "detail": f"{label}: {reason}"}

        if from_id == to_id:
            return {"error": "same_id"}

        payload_files = payload.get("files")
        if payload_files:
            files_to_move = [str(f) for f in payload_files]
        else:
            with self.enrollment_lock:
                le = self.last_enrollment
                if le is None or le["id"] != from_id:
                    return {
                        "error": "no_recent_enrollment",
                        "detail": "supply files=[...] or call within 60s of /selfie",
                    }
                if time.monotonic() - le["monotonic_ts"] > 60.0:
                    return {"error": "stale_enrollment"}
                files_to_move = list(le["files"])

        # Path traversal guard: every file must resolve to inside gallery_dir
        safe_files = self._filter_safe_paths(files_to_move)
        if not safe_files:
            return {"error": "no_safe_files"}

        t0 = time.monotonic()
        moved, from_removed, n_id = self.run_job_sync(
            lambda: self._do_move_samples(from_id, to_id, safe_files)
        )

        # Clear last_enrollment if we just consumed it
        with self.enrollment_lock:
            if (
                self.last_enrollment is not None
                and self.last_enrollment["id"] == from_id
            ):
                self.last_enrollment = None

        return {
            "ok": True,
            "from_id": from_id,
            "to_id": to_id,
            "moved": int(moved),
            "from_removed": bool(from_removed),
            "identities": int(n_id),
            "took_sec": round(time.monotonic() - t0, 3),
        }

    def _do_move_samples(
        self, from_id: str, to_id: str, files: List[str]
    ) -> "tuple[int, bool, int]":
        """Main-thread worker. Returns (moved_count, from_removed, n_identities)."""
        to_aligned = os.path.join(self.gallery_dir, to_id, "aligned")
        os.makedirs(to_aligned, exist_ok=True)

        moved = 0
        for rel in files:
            src = os.path.join(self.gallery_dir, rel)
            if not os.path.isfile(src):
                continue
            fname = os.path.basename(rel)
            dst = os.path.join(to_aligned, fname)
            if os.path.exists(dst):
                # Collision: append _moved<n> to disambiguate
                base, ext = os.path.splitext(fname)
                k = 1
                while True:
                    dst = os.path.join(to_aligned, f"{base}_moved{k}{ext}")
                    if not os.path.exists(dst):
                        break
                    k += 1
            shutil.move(src, dst)
            moved += 1

        from_removed = self._cleanup_empty_identity_dir(from_id)

        # Rebuild from scratch — refresh() only adds, doesn't remove stale entries
        self.gm.clear_and_rebuild()
        feats, labels = self.gm.get_identity_means()
        with self.gal_lock:
            self.gal_state.gal_feats = feats
            self.gal_state.gal_labels = labels

        self._audit_log(
            f"move {from_id}->{to_id} files={len(files)} moved={moved} "
            f"from_removed={from_removed}"
        )
        return moved, from_removed, len(labels)

    # -------------------- /gallery/forget_last --------------------- #
    def _handle_gallery_forget_last(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete the samples saved by the most recent /selfie call.
        Used when the wrong person was captured.

        Payload (optional)
        ------------------
        id : str — if provided, must match last_enrollment.id (safety check)
        """
        if not self.gm:
            return {"error": "recognition_disabled"}

        payload = payload or {}

        with self.enrollment_lock:
            le = self.last_enrollment
            if le is None:
                return {"error": "no_recent_enrollment"}
            if time.monotonic() - le["monotonic_ts"] > 60.0:
                return {"error": "stale_enrollment"}

            target_id = le["id"]
            files = list(le["files"])

            if "id" in payload:
                requested = str(payload["id"]).strip().lower()
                if requested != target_id:
                    return {
                        "error": "id_mismatch",
                        "detail": f"last enrollment was {target_id}",
                    }

        safe_files = self._filter_safe_paths(files)
        if not safe_files:
            return {"error": "no_safe_files"}

        t0 = time.monotonic()
        deleted, identity_removed, n_id = self.run_job_sync(
            lambda: self._do_forget_files(target_id, safe_files)
        )

        with self.enrollment_lock:
            self.last_enrollment = None

        return {
            "ok": True,
            "id": target_id,
            "files_deleted": int(deleted),
            "identity_removed": bool(identity_removed),
            "identities": int(n_id),
            "took_sec": round(time.monotonic() - t0, 3),
        }

    def _do_forget_files(
        self, target_id: str, files: List[str]
    ) -> "tuple[int, bool, int]":
        """Main-thread worker for /gallery/forget_last."""
        deleted = 0
        for rel in files:
            path = os.path.join(self.gallery_dir, rel)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    deleted += 1
                except OSError as e:
                    self.log.warning("failed to remove %s: %s", path, e)

        identity_removed = self._cleanup_empty_identity_dir(target_id)

        # Rebuild from scratch — refresh() leaves stale entries behind
        self.gm.clear_and_rebuild()
        feats, labels = self.gm.get_identity_means()
        with self.gal_lock:
            self.gal_state.gal_feats = feats
            self.gal_state.gal_labels = labels

        self._audit_log(
            f"forget {target_id} files={len(files)} deleted={deleted} "
            f"identity_removed={identity_removed}"
        )
        return deleted, identity_removed, len(labels)

    # -------------------- helpers --------------------- #
    def _filter_safe_paths(self, rel_paths: List[Any]) -> List[str]:
        """
        Return only the relative paths that resolve to inside gallery_dir.

        Path-traversal guard for /gallery/move_samples (where `files` may
        come from a payload). Rejects absolute paths, `../` escapes, and
        anything pointing outside the gallery root.
        """
        safe: List[str] = []
        base = os.path.abspath(self.gallery_dir) + os.sep
        for rel in rel_paths:
            if not isinstance(rel, str) or not rel:
                continue
            abs_path = os.path.abspath(os.path.join(self.gallery_dir, rel))
            if not (abs_path + os.sep).startswith(base):
                self.log.warning("rejecting path traversal attempt: %s", rel)
                continue
            safe.append(rel)
        return safe

    def _cleanup_empty_identity_dir(self, identity: str) -> bool:
        """
        Remove identity_dir and its empty subfolders. Returns True if removed.
        Conservative: leaves the folder alone if anything (e.g. raw/*.jpg)
        is still in it.
        """
        id_dir = os.path.join(self.gallery_dir, identity)
        if not os.path.isdir(id_dir):
            return False

        for sub in ("aligned", "raw"):
            sub_dir = os.path.join(id_dir, sub)
            if os.path.isdir(sub_dir) and not os.listdir(sub_dir):
                try:
                    os.rmdir(sub_dir)
                except OSError:
                    pass

        try:
            if not os.listdir(id_dir):
                os.rmdir(id_dir)
                return True
        except OSError:
            pass
        return False

    def _audit_log(self, msg: str) -> None:
        """Append a timestamped line to gallery_dir/corrections.log."""
        log_path = os.path.join(self.gallery_dir, "corrections.log")
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
        """Load an image from path or base64 and ensure size is 112×112.

        Parameters
        ----------
        image_path : str, optional
            Filesystem path to an image. If provided, takes precedence.
        image_b64 : str, optional
            Base64-encoded JPEG/PNG bytes (URL-safe or standard).

        Returns
        -------
        np.ndarray or None
            BGR image of shape `(112, 112, 3)` on success; otherwise `None`.

        Notes
        -----
        This helper performs minimal validation and resizing; callers should do
        additional checks (e.g., identity label existence) as needed.
        """
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
