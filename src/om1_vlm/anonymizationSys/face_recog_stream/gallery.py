"""
Gallery & embedding manager for face recognition.

This module manages a disk-backed face gallery and its embedding store.
It aligns new photos (optionally from `raw/` using SCRFD) into 112×112
`aligned/` crops, embeds them with ArcFace (TensorRT), and persists the
results under `embeds/<model_sig>/{index.json,vectors.f32,stats.json}`.
It supports incremental refresh (only process new files), adding single
aligned snapshots, clearing & rebuilding, and computing per-identity mean
vectors for fast runtime recognition.

Typical usage:
- At startup (or after HTTP actions like /gallery/refresh, /gallery/add_aligned,
  /gallery/add_raw, /selfie), refresh the gallery and fetch identity means.
- At inference time, compare a face embedding to the returned per-identity means.

Notes:
- Embeddings are float32; vectors are L2-normalized before similarity.
- Batch size respects the ArcFace TensorRT optimization profile (`arc_max_bs`).
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
import shutil
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .arcface import TRTArcFace, warp_face_by_5p
from .scrfd import TRTSCRFD
from .utils import infer_arc_batched, list_images  # helpers

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ----------------------------
# Small filesystem helpers
# ----------------------------


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%S")


# ----------------------------
# Model signature
# ----------------------------


def make_model_sig(arc: TRTArcFace, extra: Optional[str] = None) -> str:
    """Build a readable, deterministic model signature for the embedding store.

    Parameters
    ----------
    arc : TRTArcFace
        ArcFace TensorRT wrapper; used to read model name and embedding dimension.
    extra : str, optional
        Extra token to append (e.g., engine variant), by default ``None``.

    Returns
    -------
    str
        Signature string like ``"{name}-trt-l2-{dim}[-{extra}]"``.
    """

    name = getattr(arc, "name", None) or getattr(arc, "model_name", None) or "arc"
    dim = int(getattr(arc, "embedding_dim", 512))
    backend = "trt"
    norm = "l2"
    parts = [str(name), backend, norm, str(dim)]
    if extra:
        parts.append(str(extra))
    return "-".join(parts)


# ----------------------------
# Alignment utilities
# ----------------------------


def _align_largest_face_bgr(
    img_bgr: np.ndarray,
    det: TRTSCRFD,
    conf: float,
    size: int = 112,
) -> Optional[np.ndarray]:
    """
    Detect faces, choose the largest, align with 5p if available, else crop+resize.
    Returns a size×size BGR crop or None if no face.
    """
    dets, kpss = det.detect(img_bgr, conf=conf)
    if dets is None or dets.shape[0] == 0:
        return None
    areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    idx = int(np.argmax(areas))
    if kpss is not None:
        try:
            return warp_face_by_5p(img_bgr, kpss[idx], size)
        except Exception:
            pass
    x1, y1, x2, y2, _ = dets[idx].astype(int)
    face = img_bgr[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    if face.size == 0:
        return None
    return cv2.resize(face, (size, size))


# ----------------------------
# Gallery Manager
# ----------------------------


class GalleryManager:
    """
    Manages a face gallery with aligned crops and an embedding store. It supports incremental refresh from
    ``raw/`` and ``aligned/``, batched embedding respecting the ArcFace TensorRT max batch size, and per-identity statistics (counts and mean vectors)

    Layout:
      gallery/
        Alice/ aligned/*.jpg   (112x112)  [source of truth]
               raw/*.jpg       (optional; will be aligned -> aligned/]
        Bob/   aligned/*.jpg
      embeds/
        <model_sig>/
          index.json    { "meta": {...}, "items": { "Alice/aligned/foo.jpg": {"row":0,"label":"Alice"}, ... } }
          vectors.f32   float32 raw, concatenated (N * dim)
          stats.json    { "labels": {"Alice": {"count": X, "mean": [...]}}, "count": N, "dim": dim }

    Typical usage:
      gm = GalleryManager(gallery_dir, embeds_dir, arc, scrfd)
      gm.refresh()                    # incremental: align new raw/, embed new aligned/
      feats, labels = gm.get_identity_means()  # per-identity means (for recognition)

    Attributes
    ----------
    gallery_dir : str
        Absolute path to the gallery root (one subfolder per identity).
    embeds_root : str
        Absolute path to the embeddings root folder.
    arc : TRTArcFace
        ArcFace TensorRT wrapper used for feature extraction.
    scrfd : TRTSCRFD or None
        SCRFD detector used to align new images from ``raw/``.
    det_conf : float
        Detection confidence threshold for alignment.
    aligned_size : int
        Aligned face crop size (pixels).
    model_sig : str
        Model signature used to segregate embed stores.
    arc_max_bs : int
        Maximum ArcFace batch size (matches TRT optimization profile).
    store_dir : str
        Directory under ``embeds_root`` for this model signature.
    index_path : str
        JSON index mapping relative image paths to row indices and labels.
    vectors_path : str
        Binary file storing concatenated float32 embeddings ``(N * dim)``.
    stats_path : str
        JSON file with per-identity counts and mean vectors.
    """

    def __init__(
        self,
        gallery_dir: str,
        embeds_dir: str,
        *,
        arc: TRTArcFace,
        scrfd: Optional[TRTSCRFD] = None,
        det_conf: float = 0.5,
        aligned_size: int = 112,
        model_sig: Optional[str] = None,
        arc_max_bs: int = 4,  # IMPORTANT: respect TensorRT optimization profile (e.g., 1..4)
    ) -> None:
        """
        Initialize the gallery manager and load/create the embed store.
        """
        self.gallery_dir = osp.abspath(gallery_dir)
        self.embeds_root = osp.abspath(embeds_dir)
        self.arc = arc
        self.scrfd = scrfd
        self.det_conf = float(det_conf)
        self.aligned_size = int(aligned_size)
        self.model_sig = model_sig or make_model_sig(arc)
        self.arc_max_bs = int(arc_max_bs)

        # Paths under embeds/
        self.store_dir = osp.join(self.embeds_root, self.model_sig)
        self.index_path = osp.join(self.store_dir, "index.json")
        self.vectors_path = osp.join(self.store_dir, "vectors.f32")
        self.stats_path = osp.join(self.store_dir, "stats.json")

        _ensure_dir(self.store_dir)
        self._index = self._load_index()
        self._dim = int(getattr(self.arc, "embedding_dim", 512))

    # ---------- public API ----------

    def refresh(self, process_raw: bool = True) -> Tuple[int, int]:
        """
        Incrementally update the embed store.

        - Optionally process new images in `raw/`: detect→align→write into `aligned/`.
        - Embed any new 112×112 crops under `aligned/` and append to vectors.

        Parameters
        ----------
        process_raw : bool, optional
            Whether to detect→align images from ``raw/`` into ``aligned/``, by default ``True``.

        Returns:
          (num_aligned_added, num_vectors_appended)
        """
        if process_raw and self.scrfd is None:
            log.warning(
                "refresh(): 'process_raw=True' but SCRFD was not provided; skipping raw/"
            )

        if process_raw and self.scrfd is not None:
            self._harvest_raw_to_aligned()

        added = self._embed_new_aligned()
        self._recompute_stats()
        return added

    def clear_and_rebuild(self) -> Tuple[int, int]:
        """Delete current vectors and rebuild embeddings from ``aligned/`` only.

        Returns
        -------
        tuple[int, int]
            ``(num_aligned_added, num_vectors_appended)`` after rebuild.
        """
        if osp.exists(self.vectors_path):
            try:
                os.remove(self.vectors_path)
            except Exception:
                pass
        self._index = {"meta": self._index.get("meta", {}), "items": {}}
        self._save_index()

        added = self._embed_new_aligned()
        self._recompute_stats()
        return added

    def add_aligned_snapshot(
        self, label: str, img_112_bgr: np.ndarray, fname_hint: Optional[str] = None
    ) -> str:
        """
        Add a new 112×112 face (already aligned) under gallery/<label>/aligned,
        then embed and append to the store. Returns the relative aligned path.

        Parameters
        ----------
        label : str
            Identity name (subfolder under gallery).
        img_112_bgr : np.ndarray
            Aligned BGR crop of shape ``(112,112,3)`` (will be resized if needed).
        fname_hint : str, optional
            Preferred filename (``.jpg`` added if missing), by default ``None``.

        Returns
        -------
        str
            Relative path of the saved aligned image (e.g., ``"Alice/aligned/xxx.jpg"``).
        """
        if img_112_bgr is None or img_112_bgr.size == 0:
            raise ValueError("empty snapshot")

        if img_112_bgr.shape[:2] != (self.aligned_size, self.aligned_size):
            img_112_bgr = cv2.resize(
                img_112_bgr, (self.aligned_size, self.aligned_size)
            )

        # Save
        label_dir = osp.join(self.gallery_dir, label, "aligned")
        _ensure_dir(label_dir)
        base = (fname_hint or f"{_now_iso()}.jpg").strip()
        if not base.lower().endswith(".jpg"):
            base += ".jpg"
        rel = osp.join(label, "aligned", base).replace("\\", "/")
        absf = osp.join(self.gallery_dir, rel)
        # unique filename if exists
        i = 0
        while osp.exists(absf):
            i += 1
            stem, ext = osp.splitext(base)
            rel = osp.join(label, "aligned", f"{stem}_{i}{ext}").replace("\\", "/")
            absf = osp.join(self.gallery_dir, rel)

        cv2.imwrite(absf, img_112_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Embed & append
        vec = self._embed_single(img_112_bgr)
        row = self._append_vectors(vec[None, :])
        self._index["items"][rel] = {"row": row, "label": label}
        self._save_index()
        self._recompute_stats()
        return rel

    def delete_identity(self, label: str) -> tuple[bool, int, int, int]:
        """
        Delete one identity from the gallery and rebuild the embedding store.

        Steps
        -----
        - Remove the folder `gallery/<label>/` (both raw/ and aligned/).
        - Clear & rebuild the embed store from remaining `aligned/` photos.
        - Recompute stats, so runtime recognition stops matching this label.

        Parameters
        -------
        label: person name: wendy

        Returns
        -------
        (removed_gallery, aligned_used, vectors_rebuilt, identities_left)
        """
        p = osp.join(self.gallery_dir, label)
        removed_gallery = False
        if osp.isdir(p):
            shutil.rmtree(p)
            removed_gallery = True

        aligned_used, vectors_rebuilt = self.clear_and_rebuild()

        feats, labels = self.get_identity_means()
        return removed_gallery, aligned_used, vectors_rebuilt, len(labels)

    def get_all_vectors_and_labels(self) -> Tuple[np.ndarray, List[str]]:
        """
        Load the whole vectors file into RAM and return (vectors, labels_per_row).
        Returns
        -------
        tuple[np.ndarray, list[str]]
            ``(vectors, labels_per_row)`` where ``vectors`` is ``(N, dim)`` float32.
        """
        if not osp.exists(self.vectors_path):
            return np.zeros((0, self._dim), np.float32), []
        arr = np.fromfile(self.vectors_path, dtype=np.float32)
        if arr.size % self._dim != 0:
            raise RuntimeError("vectors.f32 is corrupted or wrong dim")
        arr = arr.reshape((-1, self._dim))
        labels = self._row_labels(len(arr))
        return arr, labels

    def get_identity_means(self) -> Tuple[np.ndarray, List[str]]:
        """
        Return per-identity mean vectors (N_id × dim) and label list.
        Mirrors your old build_gallery_embeddings() contract.
        Returns
        -------
        tuple[np.ndarray, list[str]]
            ``(means, labels)`` where ``means`` is ``(N_id, dim)`` float32 and
            each row is L2-normalized.
        """
        stats = self._load_stats()
        labels: List[str] = []
        means: List[np.ndarray] = []
        if stats and "labels" in stats:
            for lab, rec in stats["labels"].items():
                if rec.get("count", 0) > 0 and "mean" in rec:
                    labels.append(lab)
                    means.append(np.asarray(rec["mean"], dtype=np.float32))
        if not means:
            return np.zeros((0, self._dim), np.float32), []
        return np.stack(means, axis=0).astype(np.float32), labels

    def legacy_build_gallery_embeddings(
        self, det_conf: Optional[float] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Backward-compatible shim for older call sites:
            feats, labels = build_gallery_embeddings(gallery, scrfd, arc, det_conf)
        Under the hood, we refresh and return per-identity means.
        Parameters
        ----------
        det_conf : float, optional
            Detection confidence to use during refresh, by default ``None`` (keep current).

        Returns
        -------
        tuple[np.ndarray, list[str]]
            ``(means, labels)`` as in :meth:`get_identity_means`.
        """
        if det_conf is not None:
            self.det_conf = float(det_conf)
        self.refresh(process_raw=True)
        return self.get_identity_means()

    # ---------- internals ----------

    def _load_index(self) -> Dict:
        """Load or initialize the JSON index file.

        Returns
        -------
        dict
            Index structure with ``meta`` and ``items`` keys. ``items`` maps
            relative image paths to ``{"row": int, "label": str}``.
        """
        if osp.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    idx = json.load(f)
                if "items" not in idx:
                    idx = {"meta": {}, "items": {}}
            except Exception:
                idx = {"meta": {}, "items": {}}
        else:
            idx = {"meta": {}, "items": {}}

        idx["meta"].update(
            {
                "model_sig": self.model_sig,
                "dim": int(getattr(self.arc, "embedding_dim", 512)),
                "created": idx["meta"].get("created") or _now_iso(),
                "updated": _now_iso(),
            }
        )
        self._save_json(self.index_path, idx)
        return idx

    def _save_index(self) -> None:
        """Update the index ``updated`` timestamp and persist to disk."""
        self._index["meta"]["updated"] = _now_iso()
        self._save_json(self.index_path, self._index)

    def _save_json(self, path: str, obj: Dict) -> None:
        """Write a JSON object to a file (ensuring parent directory exists)."""
        _ensure_dir(osp.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _load_stats(self) -> Optional[Dict]:
        """Load per-identity statistics from ``stats.json`` if present."""
        if not osp.exists(self.stats_path):
            return None
        try:
            with open(self.stats_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _recompute_stats(self) -> None:
        """
        Build per-identity means and counts; write to stats.json.
        """
        if not osp.exists(self.vectors_path):
            stats = {
                "count": 0,
                "dim": self._dim,
                "labels": {},
                "updated": _now_iso(),
            }
            self._save_json(self.stats_path, stats)
            return

        vecs, row_labels = self.get_all_vectors_and_labels()
        per: Dict[str, Dict[str, object]] = {}
        uniq = sorted(set(row_labels))
        for lab in uniq:
            idxs = [i for i, L in enumerate(row_labels) if L == lab]
            if not idxs:
                continue
            m = vecs[idxs].mean(axis=0)
            m /= np.linalg.norm(m) + 1e-9
            per[lab] = {"count": int(len(idxs)), "mean": m.tolist()}
        stats = {
            "count": int(vecs.shape[0]),
            "dim": int(self._dim),
            "labels": per,
            "updated": _now_iso(),
        }
        self._save_json(self.stats_path, stats)

    def _row_labels(self, n_rows: int) -> List[str]:
        """
        Build a dense row->label list from index.json.
        """
        lab = [""] * n_rows
        for _rel, rec in self._index["items"].items():
            r = int(rec["row"])
            if 0 <= r < n_rows:
                lab[r] = rec["label"]
        return lab

    def _harvest_raw_to_aligned(self) -> None:
        """
        For every gallery/<id>/raw/*.jpg, if there isn't a corresponding file
        under gallery/<id>/aligned/, detect & align and save it.
        """
        ids = self._iter_identities(self.gallery_dir)
        if not ids:
            return
        for label, (aligned_dir, raw_dir) in ids:
            if not osp.isdir(raw_dir):
                continue
            for p in list_images(raw_dir):
                base = osp.splitext(osp.basename(p))[0]
                tgt_dir = aligned_dir
                _ensure_dir(tgt_dir)
                tgt = osp.join(tgt_dir, f"{base}.jpg")
                if osp.exists(tgt):
                    continue  # already aligned once
                img = cv2.imread(p)
                if img is None or img.size == 0:
                    continue
                crop = _align_largest_face_bgr(
                    img, self.scrfd, self.det_conf, self.aligned_size
                )  # type: ignore[arg-type]
                if crop is None:
                    continue
                cv2.imwrite(tgt, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def _embed_new_aligned(self) -> Tuple[int, int]:
        """
        Scan aligned/ for images not in index; embed and append to vectors.
        Returns (num_aligned_found, num_vectors_appended)
        """
        ids = self._iter_identities(self.gallery_dir)
        if not ids:
            return 0, 0

        new_paths: List[Tuple[str, str]] = []  # (label, rel_path)
        for label, (aligned_dir, _raw_dir) in ids:
            if not osp.isdir(aligned_dir):
                continue
            for ap in list_images(aligned_dir):
                rel = osp.relpath(ap, self.gallery_dir).replace("\\", "/")
                if rel in self._index["items"]:
                    continue
                new_paths.append((label, rel))

        if not new_paths:
            return 0, 0

        imgs: List[np.ndarray] = []
        rels: List[str] = []
        labels: List[str] = []
        for label, rel in new_paths:
            absf = osp.join(self.gallery_dir, rel)
            img = cv2.imread(absf)
            if img is None:
                continue
            if img.shape[:2] != (self.aligned_size, self.aligned_size):
                img = cv2.resize(img, (self.aligned_size, self.aligned_size))
            imgs.append(img)
            rels.append(rel)
            labels.append(label)

        if not imgs:
            return 0, 0

        # >>> respect TensorRT optimization profile (e.g., max batch = 4)
        feats = self._embed_batch(imgs)
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
        feats = (feats / norms).astype(np.float32)

        start_row = self._append_vectors(feats)
        for i, rel in enumerate(rels):
            self._index["items"][rel] = {"row": int(start_row + i), "label": labels[i]}
        self._save_index()

        log.info(
            "gallery: appended %d vectors (total now: %d)",
            feats.shape[0],
            start_row + feats.shape[0],
        )
        return len(rels), feats.shape[0]

    def _embed_single(self, img_112_bgr: np.ndarray) -> np.ndarray:
        """Embed a single aligned 112×112 BGR crop and L2-normalize the vector.

        Parameters
        ----------
        img_112_bgr : np.ndarray
            Aligned crop of shape ``(112,112,3)`` (will be resized if needed).

        Returns
        -------
        np.ndarray
            Normalized embedding vector of shape ``(dim,)`` with dtype float32.
        """
        vec = self.arc.infer([img_112_bgr])
        if vec.ndim == 2 and vec.shape[0] == 1:
            vec = vec[0]
        vec = vec.astype(np.float32, copy=False)
        n = np.linalg.norm(vec) + 1e-9
        return (vec / n).astype(np.float32, copy=False)

    def _embed_batch(self, imgs_112_bgr: List[np.ndarray]) -> np.ndarray:
        """Embed a list of aligned crops, chunked to respect ArcFace max batch.

        Parameters
        ----------
        imgs_112_bgr : list[np.ndarray]
            List of aligned BGR crops, each ``(112,112,3)``.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(K, dim)`` containing raw (unnormalized) vectors.
        """
        if not imgs_112_bgr:
            return np.empty((0, self._dim), np.float32)
        # Use your shared helper (already used at runtime)
        vecs = infer_arc_batched(self.arc, imgs_112_bgr, max_bs=self.arc_max_bs)
        return np.asarray(vecs, dtype=np.float32, copy=False)

    def _append_vectors(self, vecs: np.ndarray) -> int:
        """Append embeddings to ``vectors.f32`` and return the starting row index.

        Parameters
        ----------
        vecs : np.ndarray
            Float32 array of shape ``(K, dim)`` to append.

        Returns
        -------
        int
            Starting row index where the first appended vector was written.

        Raises
        ------
        AssertionError
            If ``vecs`` does not have shape ``(K, dim)``.
        """
        assert (
            vecs.ndim == 2 and vecs.shape[1] == self._dim
        ), f"expected (K,{self._dim}) got {vecs.shape}"
        prev_rows = 0
        if osp.exists(self.vectors_path):
            sz = osp.getsize(self.vectors_path)
            prev_rows = sz // (4 * self._dim)
        with open(self.vectors_path, "ab") as f:
            vecs.astype(np.float32, copy=False).tofile(f)
        return int(prev_rows)

    @staticmethod
    def _iter_identities(gallery_root: str) -> List[Tuple[str, Tuple[str, str]]]:
        """Enumerate identity folders and their aligned/raw subdirectories.

        Parameters
        ----------
        gallery_root : str
            Gallery root path.

        Returns
        -------
        list[tuple[str, tuple[str, str]]]
            Pairs of ``(label, (aligned_dir, raw_dir))`` for each identity folder.
        """
        out: List[Tuple[str, Tuple[str, str]]] = []
        if not osp.isdir(gallery_root):
            return out
        for name in sorted(os.listdir(gallery_root)):
            p = osp.join(gallery_root, name)
            if not osp.isdir(p):
                continue
            aligned_dir = osp.join(p, "aligned")
            raw_dir = osp.join(p, "raw")
            out.append((name, (aligned_dir, raw_dir)))
        return out


# ----------------------------------------
# Back-compat free function
# ----------------------------------------


def build_gallery_embeddings(
    gallery_root: str, scrfd: TRTSCRFD, arc: TRTArcFace, det_conf: float
) -> Tuple[np.ndarray, List[str]]:
    """
    One-shot builder that returns per-identity means (old API).
    It builds/refreshes an embed store under embeds/<model_sig>/ next to gallery/.
    Parameters
    ----------
    gallery_root : str
        Path to the gallery root.
    scrfd : TRTSCRFD
        SCRFD detector used for alignment.
    arc : TRTArcFace
        ArcFace TensorRT wrapper used for embedding.
    det_conf : float
        Detection confidence threshold for alignment.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        ``(means, labels)`` where ``means`` is ``(N_id, dim)`` float32 and L2-normalized.
    """
    embeds_root = osp.join(osp.dirname(osp.abspath(gallery_root)), "embeds")
    gm = GalleryManager(
        gallery_dir=gallery_root,
        embeds_dir=embeds_root,
        arc=arc,
        scrfd=scrfd,
        det_conf=det_conf,
        arc_max_bs=4,  # safe default for your engine profile
    )
    gm.refresh(process_raw=True)
    return gm.get_identity_means()
