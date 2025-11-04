#!/usr/bin/env python3
"""
HTTP control plane for the realtime face pipeline.

This module exposes a thin, thread-safe HTTP API that lets external tools
query presence (/who), adjust runtime configuration (/config), and manage the
face gallery (/gallery/refresh, /gallery/add_aligned, /gallery/add_raw, /selfie)
while the main video loop keeps running.

Notes
------------
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

from .arcface import warp_face_by_5p


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

    # ------------------------------ public ------------------------------ #
    def stop(self) -> None:
        """Stop an attached HTTP server if present."""
        try:
            self.server.stop()
        except Exception:
            pass

    # ----------------------------- handlers ---------------------------- #
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
                return self.who.snapshot(sec)

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

            return {"error": f"unknown path {path}"}
        except Exception as e:
            self.log.exception("HTTP error")
            return {"error": str(e)}

    # -------------------------- /config --------------------------- #
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
                    self.scrfd.nms_thresh = float(self.cfg["nms"])
                except Exception:
                    pass
            return {"ok": True, "changed": changed}

        return {
            "error": "use {'get':true} or {'set':{...}}",
            "config_keys": list(self.cfg.keys()),
        }

    # ---------------------- /gallery/refresh ----------------------- #
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

    # ---------------------- /gallery/add_raw ----------------------- #
    def _handle_gallery_add_raw(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Copy a raw image into gallery/<id>/raw and run full refresh (align+embed).
        Upload pathway -> RAW only. Alignment happens once during refresh.
        Response: { ok, saved, identities }

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

    # --------------------------- /selfie --------------------------- #
    def _handle_selfie(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save a clean snapshot from the latest frame (no overlays) into ALIGNED ONLY,
        and embed immediately (no RAW for selfie).
        Parameters
        ----------
        payload: { "id": "alice" }

        Returns
        ----------
        dict
            { ok, saved_aligned, identities }

        Notes
        -----
        Uses the cached `frame_state` (clean frame + dets/kpss) to create one
        aligned 112×112 crop without invoking extra inference in the HTTP thread.
        The embedding and centroid update are executed on the main thread via
        `run_job_sync`.
        """
        if not self.gm:
            return {"error": "recognition disabled"}
        if not payload or "id" not in payload:
            return {"error": "missing 'id'"}
        person = str(payload["id"])

        # Use last clean frame & cached dets/kpss captured in the main loop.
        with self.frame_lock:
            frm = (
                None
                if self.frame_state.frame_bgr is None
                else self.frame_state.frame_bgr.copy()
            )
            dets = (
                None if self.frame_state.dets is None else self.frame_state.dets.copy()
            )
            kpss = (
                None if self.frame_state.kpss is None else self.frame_state.kpss.copy()
            )

        if frm is None or dets is None or dets.shape[0] == 0:
            return {"error": "no recent frame/detections yet"}
        if dets.shape[0] != 1:
            return {"error": f"expect exactly 1 face, got {int(dets.shape[0])}"}

        # Build aligned crop ONCE (no new inference)
        if kpss is not None:
            crop = warp_face_by_5p(frm, kpss[0], 112)
        else:
            x1, y1, x2, y2, _ = dets[0].astype(int)
            face = frm[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            crop = cv2.resize(face if face.size > 0 else frm, (112, 112))

        ts = time.strftime("%Y-%m-%dT%H-%M-%S")

        def _do_add():
            rel = self.gm.add_aligned_snapshot(person, crop, fname_hint=f"{ts}.jpg")
            feats, labels = self.gm.get_identity_means()
            with self.gal_lock:
                self.gal_state.gal_feats, self.gal_state.gal_labels = feats, labels
            return rel, len(labels)

        rel, n_id = self.run_job_sync(_do_add)
        saved_aligned = os.path.join(
            self.gallery_dir, rel
        )  # absolute path for convenience
        return {"ok": True, "saved_aligned": saved_aligned, "identities": int(n_id)}

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
