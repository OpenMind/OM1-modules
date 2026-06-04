"""
UUID-keyed face gallery with per-identity self-contained folders.

Layout
------
gallery/
  <uuid>/
    metadata.json     # {uuid, name, created_at, updated_at, sample_count,
                      #  source, notes?}
    aligned/          # 112x112 BGR JPGs
      sample_001.jpg
      sample_002.jpg
      ...
    embeddings.npy    # (k, 512) float32, L2-normalized, one row per sample
    centroid.npy      # (512,) float32, L2-normalized incremental mean
  _legacy/            # one-shot migration moves old <name>/ folders here
  _trash/             # forget()'d UUIDs (soft delete, restorable)

Why UUIDs not names
-------------------
- Renames are O(1): just update metadata.name (no folder rename, no GPU work,
  no embedding store rewrite).
- Multiple "profiles" per name are natural: two distinct UUIDs can both be
  named "wendy" (e.g. one auto-enrolled, one selfie). Either can be deleted
  independently.
- Auto-enrolled identities start with name="" (anonymous) and get named
  later when the LLM asks. The UUID never changes — only the metadata.
- Privacy: filenames don't leak identities; the UUID is opaque on disk.
  metadata.json is the only place names live, and can be encrypted later.

Why per-folder embeddings (vs one global file)
----------------------------------------------
- forget(uuid) is rmtree on one folder — O(1) filesystem op, no GPU.
  The old design rebuilt the entire embedding store on every delete, which
  was ~25 seconds at 1000 identities × 5 samples.
- add_sample updates one centroid incrementally — O(1), no scan of others.
- get_centroids() scans folders at load and caches; in-memory after that.

Threading model
---------------
All gallery write operations should be called from a single thread (the
main loop, via run_job_sync). This module does NOT take internal locks —
the caller is responsible for serialization. Reads (get_centroids,
list_identities) are safe to call from any thread but should ideally also
be serialized with writes.
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
import re
import shutil
import time
import uuid as uuidlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .adaface import TRTFaceRecognition

log = logging.getLogger(__name__)

# UUIDs are written as plain hex strings (no dashes) to keep filenames short.
# Regex matches both lowercase hex strings and standard 8-4-4-4-12 format.
_UUID_RE = re.compile(
    r"^[a-f0-9]{32}$|^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"
)

# Reserved directory names — never treated as UUIDs.
_RESERVED_DIRS = frozenset({"_legacy", "_trash", "embeds"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _is_uuid_dir(name: str) -> bool:
    return bool(_UUID_RE.match(name))


def _atomic_write_json(path: str, obj: dict) -> None:
    """Write JSON via temp-file + rename so partial writes can't corrupt.

    Crashes between the write and rename leave the previous file intact.
    """
    _ensure_dir(osp.dirname(path))
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_npy(path: str, arr: np.ndarray) -> None:
    """Write .npy via temp + rename. Same crash-safety as JSON above."""
    _ensure_dir(osp.dirname(path))
    tmp = f"{path}.tmp.{os.getpid()}"
    np.save(tmp, arr.astype(np.float32, copy=False))
    # np.save appends .npy automatically if extension missing — handle that.
    if not osp.exists(tmp) and osp.exists(tmp + ".npy"):
        tmp = tmp + ".npy"
    os.replace(tmp, path)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize, defensive against zero vectors."""
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32, copy=False)


def _new_uuid(existing: set) -> str:
    """Generate a fresh UUID4 (hex form, 32 chars).

    UUID4 collision is astronomically unlikely (~10^-15 at 100M UUIDs), but
    we retry on the off chance to be safe — also catches the case where a
    user manually copied a folder.
    """
    for _ in range(10):
        u = uuidlib.uuid4().hex
        if u not in existing:
            return u
    raise RuntimeError("UUID generation collision (10x). Bug or corrupt state.")


# ---------------------------------------------------------------------------
# Per-UUID record (in-memory representation)
# ---------------------------------------------------------------------------


@dataclass
class IdentityRecord:
    """In-memory snapshot of one UUID folder.

    Fields are populated from metadata.json + centroid.npy + the aligned/ dir.
    Mutations on this object are NOT written back automatically — call
    ``UUIDGallery.save_metadata(record)`` after mutating ``record.metadata``.

    Centroid representation
    -----------------------
    ``centroid_sum`` is the UN-normalized sum of all sample embeddings — i.e.
    ``sum(embeddings)``, not ``mean(embeddings)`` and not L2-normalized. This
    is what we persist in ``centroid.npy``.

    Why store the sum (not the L2-normalized mean):
    Adding a new sample is just ``sum_new = sum_old + emb``. If we stored the
    L2-normalized mean, we'd lose the magnitude information needed to weight
    the existing mean correctly when folding in a new vector. Storing the
    sum keeps the add operation O(1) and mathematically equivalent to a
    full recompute.

    The L2-normalized form (what face recognition actually uses) is computed
    on the fly in :meth:`UUIDGallery.get_centroids` — that's just N vector
    normalizations, trivial even at thousands of identities.
    """

    uuid: str
    name: str
    sample_count: int
    centroid_sum: Optional[np.ndarray] = None  # un-normalized sum, (512,)
    metadata: dict = field(default_factory=dict)

    @property
    def is_named(self) -> bool:
        """True if this identity has a non-empty display name (i.e. not anon)."""
        return bool(self.name and self.name.strip())

    @property
    def centroid_normalized(self) -> Optional[np.ndarray]:
        """Lazy L2-normalized centroid for similarity comparisons."""
        if self.centroid_sum is None:
            return None
        return _l2_normalize(self.centroid_sum)


# ---------------------------------------------------------------------------
# UUIDGallery
# ---------------------------------------------------------------------------


class UUIDGallery:
    """Disk-backed face gallery keyed by UUID.

    Parameters
    ----------
    gallery_dir : str
        Root directory containing the UUID folders.
    arc : TRTFaceRecognition
        AdaFace TRT engine for embedding extraction. Used by ``embed_aligned``
        and any operation that adds a sample without a pre-computed vector.
    aligned_size : int, default 112
        Face crop dimension. Almost always 112 for AdaFace/ArcFace.
    """

    def __init__(
        self,
        gallery_dir: str,
        *,
        arc: TRTFaceRecognition,
        aligned_size: int = 112,
    ) -> None:
        self.gallery_dir = osp.abspath(gallery_dir)
        self.arc = arc
        self.aligned_size = int(aligned_size)
        self._dim = int(getattr(arc, "embedding_dim", 512))

        _ensure_dir(self.gallery_dir)
        _ensure_dir(osp.join(self.gallery_dir, "_trash"))

        # In-memory cache: uuid → IdentityRecord. Populated on demand by
        # _load() and refreshed by reload().
        self._records: Dict[str, IdentityRecord] = {}
        self.reload()

    # ==================================================================
    # Loading
    # ==================================================================

    def reload(self) -> int:
        """Scan ``gallery_dir`` and rebuild the in-memory record cache.

        Returns the number of UUIDs loaded. Call this after external
        modifications to ``gallery_dir`` (e.g. file-level rsync).
        """
        self._records.clear()
        if not osp.isdir(self.gallery_dir):
            return 0
        for entry in os.listdir(self.gallery_dir):
            if entry in _RESERVED_DIRS:
                continue
            path = osp.join(self.gallery_dir, entry)
            if not osp.isdir(path):
                continue
            if not _is_uuid_dir(entry):
                # Legacy folder; needs migration via migrate_legacy().
                log.warning(
                    "Skipping non-UUID directory '%s' (run migrate_legacy)",
                    entry,
                )
                continue
            try:
                rec = self._load(entry)
                if rec is not None:
                    self._records[entry] = rec
            except Exception as e:
                log.warning("Failed to load UUID %s: %s", entry, e)
        log.info("Gallery loaded: %d UUIDs", len(self._records))
        return len(self._records)

    def _load(self, uuid: str) -> Optional[IdentityRecord]:
        """Load one UUID's record from disk. Returns None on corrupted entry."""
        uuid_dir = osp.join(self.gallery_dir, uuid)
        meta_path = osp.join(uuid_dir, "metadata.json")
        centroid_path = osp.join(uuid_dir, "centroid.npy")

        if not osp.exists(meta_path):
            log.warning("UUID %s missing metadata.json", uuid)
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            log.warning("UUID %s metadata.json invalid: %s", uuid, e)
            return None

        centroid_sum: Optional[np.ndarray] = None
        if osp.exists(centroid_path):
            try:
                centroid_sum = np.load(centroid_path).astype(np.float32)
                if centroid_sum.shape != (self._dim,):
                    log.warning(
                        "UUID %s centroid shape %s, expected (%d,)",
                        uuid,
                        centroid_sum.shape,
                        self._dim,
                    )
                    centroid_sum = None
            except Exception as e:
                log.warning("UUID %s centroid.npy invalid: %s", uuid, e)

        sample_count = int(meta.get("sample_count", 0))
        # Cross-check sample_count against actual files (cheap sanity check)
        aligned_dir = osp.join(uuid_dir, "aligned")
        if osp.isdir(aligned_dir):
            actual = len([f for f in os.listdir(aligned_dir) if f.endswith(".jpg")])
            if actual != sample_count:
                log.debug(
                    "UUID %s: metadata sample_count=%d, actual files=%d (will fix)",
                    uuid,
                    sample_count,
                    actual,
                )
                sample_count = actual
                meta["sample_count"] = actual

        return IdentityRecord(
            uuid=uuid,
            name=str(meta.get("name", "")),
            sample_count=sample_count,
            centroid_sum=centroid_sum,
            metadata=meta,
        )

    # ==================================================================
    # Read APIs (called from recognition + UI)
    # ==================================================================

    def get_centroids(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """Build the (feats, names, uuids) triple consumed by face_tracker.

        Only includes UUIDs with at least one sample. Names are the
        human-readable labels from metadata.json — they may be empty for
        auto-enrolled (yet-to-be-named) identities.

        Returns
        -------
        feats : (N, 512) float32
            L2-normalized centroids, one row per included UUID.
        names : list[str]
            Display names, parallel to ``feats``.
        uuids : list[str]
            UUIDs, parallel to ``feats``. Use this for stable identity refs
            (e.g. /set_name, /forget).
        """
        feats_list: List[np.ndarray] = []
        names: List[str] = []
        uuids: List[str] = []
        for u, rec in self._records.items():
            if rec.centroid_sum is None or rec.sample_count == 0:
                continue
            # Normalize the running sum for cosine-similarity use. Storing
            # the un-normalized sum on disk lets add_sample stay O(1); the
            # per-frame normalization here is N vector norms (trivial).
            feats_list.append(_l2_normalize(rec.centroid_sum))
            names.append(rec.name)
            uuids.append(u)
        if not feats_list:
            return np.zeros((0, self._dim), dtype=np.float32), [], []
        return np.stack(feats_list, axis=0).astype(np.float32), names, uuids

    def get_record(self, uuid: str) -> Optional[IdentityRecord]:
        """Look up a single UUID's record."""
        return self._records.get(uuid)

    def get_metadata(self, uuid: str) -> Optional[dict]:
        """Return metadata.json contents for ``uuid``, or None if unknown."""
        rec = self._records.get(uuid)
        return dict(rec.metadata) if rec is not None else None

    def list_identities(self, include_unnamed: bool = True) -> List[dict]:
        """Public listing of all identities (one dict per UUID).

        Used by /gallery/identities. Returns lightweight summary dicts —
        not full centroids.
        """
        out: List[dict] = []
        for u, rec in sorted(self._records.items()):
            if not include_unnamed and not rec.is_named:
                continue
            out.append(
                {
                    "uuid": u,
                    "name": rec.name,
                    "sample_count": rec.sample_count,
                    "created_at": rec.metadata.get("created_at"),
                    "updated_at": rec.metadata.get("updated_at"),
                    "source": rec.metadata.get("source"),
                }
            )
        return out

    def find_by_name(self, name: str) -> List[str]:
        """All UUIDs whose name matches ``name`` (case-insensitive).

        Multiple UUIDs may share the same name (e.g. enrolled twice). The
        caller decides what to do — typically, the first or the one with
        the most samples.
        """
        target = name.strip().lower()
        return [
            u for u, rec in self._records.items() if rec.name.strip().lower() == target
        ]

    def num_identities(self) -> int:
        """Number of UUIDs with at least one sample."""
        return sum(1 for r in self._records.values() if r.sample_count > 0)

    # ==================================================================
    # Write APIs — create / append / rename / delete
    # ==================================================================

    def create(self, name: str = "", source: str = "selfie", **extra) -> str:
        """Create an empty UUID folder. Returns the new UUID.

        The folder is created on disk immediately; ``add_sample`` is required
        before the UUID will show up in ``get_centroids``.

        Parameters
        ----------
        name : str
            Display name. Use ``""`` for auto-enrolled / anonymous identities;
            ``set_name`` later when the name is known.
        source : str
            Provenance tag (``"selfie"``, ``"auto_enroll"``, ``"migration"``,
            ``"add_raw"``, ...). Helps with debugging and audit.
        extra : dict
            Extra metadata key/values to merge into metadata.json (e.g.
            ``notes="enrolled at GTC"``).
        """
        u = _new_uuid(set(self._records.keys()))
        uuid_dir = osp.join(self.gallery_dir, u)
        _ensure_dir(uuid_dir)
        _ensure_dir(osp.join(uuid_dir, "aligned"))

        meta = {
            "uuid": u,
            "name": name,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "sample_count": 0,
            "source": source,
            **extra,
        }
        _atomic_write_json(osp.join(uuid_dir, "metadata.json"), meta)

        self._records[u] = IdentityRecord(
            uuid=u,
            name=name,
            sample_count=0,
            centroid_sum=None,
            metadata=meta,
        )
        log.info("Created UUID %s (name=%r, source=%s)", u, name, source)
        return u

    def add_sample(
        self,
        uuid: str,
        aligned_112_bgr: np.ndarray,
        embedding: Optional[np.ndarray] = None,
        *,
        fname_hint: Optional[str] = None,
    ) -> str:
        """Append one sample to an existing UUID.

        Steps
        -----
        1. Compute embedding if not provided (GPU work).
        2. Write JPG to ``aligned/``.
        3. Append the embedding to ``embeddings.npy`` (full rewrite, since
           numpy lacks streaming append for ``.npy`` format — but it's still
           cheap, embeddings are ~2KB per row).
        4. Update centroid incrementally: ``new = norm((old*N + emb))``.
        5. Update metadata.sample_count and updated_at.

        Returns the filename of the saved sample (e.g. ``"sample_004.jpg"``).
        """
        rec = self._records.get(uuid)
        if rec is None:
            raise KeyError(f"UUID not found: {uuid}")

        # Normalize/validate the crop
        if aligned_112_bgr is None or aligned_112_bgr.size == 0:
            raise ValueError("empty aligned crop")
        crop = aligned_112_bgr
        if crop.shape[:2] != (self.aligned_size, self.aligned_size):
            crop = cv2.resize(crop, (self.aligned_size, self.aligned_size))

        # Compute or validate embedding
        if embedding is None:
            embedding = self.embed_aligned(crop)
        else:
            embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if embedding.shape[0] != self._dim:
                # Caller gave a malformed vector — recompute defensively
                log.warning(
                    "add_sample: embedding shape %s, expected (%d,), re-embedding",
                    embedding.shape,
                    self._dim,
                )
                embedding = self.embed_aligned(crop)
            else:
                # Ensure L2-normalized (caller might have skipped)
                embedding = _l2_normalize(embedding)

        uuid_dir = osp.join(self.gallery_dir, uuid)
        aligned_dir = osp.join(uuid_dir, "aligned")
        _ensure_dir(aligned_dir)

        # Pick a filename: ``sample_NNN.jpg`` keeps things sortable.
        if fname_hint:
            fname = fname_hint if fname_hint.endswith(".jpg") else f"{fname_hint}.jpg"
        else:
            fname = f"sample_{rec.sample_count + 1:04d}.jpg"

        # Collision-safe filename. Compute stem ONCE from the original; the
        # suffix counter keeps the chain `snap.jpg`, `snap_1.jpg`,
        # `snap_2.jpg` instead of compounding as `snap_1_2.jpg`.
        stem = fname.removesuffix(".jpg")
        full_path = osp.join(aligned_dir, fname)
        i = 0
        while osp.exists(full_path):
            i += 1
            fname = f"{stem}_{i}.jpg"
            full_path = osp.join(aligned_dir, fname)

        # Write image (NB: this is not atomic — cv2.imwrite goes direct. A
        # crash mid-write could leave a partial JPG. We accept this risk
        # because (a) the embedding+centroid update happens after, so a
        # half-JPG just orphans bytes, doesn't corrupt the gallery; (b)
        # atomic JPG write would require a temp file dance for every sample.)
        ok = cv2.imwrite(full_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise IOError(f"failed to write {full_path}")

        # Append embedding to embeddings.npy
        embeddings_path = osp.join(uuid_dir, "embeddings.npy")
        if osp.exists(embeddings_path):
            try:
                prev = np.load(embeddings_path).astype(np.float32)
            except Exception:
                log.warning("UUID %s embeddings.npy corrupt; resetting", uuid)
                prev = np.zeros((0, self._dim), dtype=np.float32)
        else:
            prev = np.zeros((0, self._dim), dtype=np.float32)
        new_arr = np.vstack([prev, embedding[None, :]]).astype(np.float32)
        _atomic_write_npy(embeddings_path, new_arr)

        N_prev = prev.shape[0]
        # Incremental centroid sum: new_sum = old_sum + embedding.
        # We store the un-normalized sum so this stays exact under O(1).
        # The L2-normalized form used for similarity is computed on the fly
        # in get_centroids().
        if rec.centroid_sum is not None and N_prev > 0:
            new_sum = rec.centroid_sum + embedding
        else:
            new_sum = embedding.copy()
        _atomic_write_npy(osp.join(uuid_dir, "centroid.npy"), new_sum)

        # Update metadata
        rec.metadata["sample_count"] = N_prev + 1
        rec.metadata["updated_at"] = _now_iso()
        _atomic_write_json(osp.join(uuid_dir, "metadata.json"), rec.metadata)

        # Update in-memory record
        rec.sample_count = N_prev + 1
        rec.centroid_sum = new_sum

        log.debug("add_sample: uuid=%s file=%s count=%d", uuid, fname, rec.sample_count)
        return fname

    def add_samples(
        self,
        uuid: str,
        aligned_list: List[np.ndarray],
        embeddings: Optional[List[np.ndarray]] = None,
    ) -> List[str]:
        """Batch-add multiple samples. More efficient than N add_sample calls
        because the centroid + metadata are written once at the end.

        ``embeddings`` may contain ``None`` entries for samples that need
        re-embedding; pass a complete list with all entries valid for the
        fast path.
        """
        rec = self._records.get(uuid)
        if rec is None:
            raise KeyError(f"UUID not found: {uuid}")
        if not aligned_list:
            return []

        # Normalize embeddings input
        if embeddings is None:
            embeddings = [None] * len(aligned_list)
        if len(embeddings) != len(aligned_list):
            raise ValueError("embeddings/aligned_list length mismatch")

        uuid_dir = osp.join(self.gallery_dir, uuid)
        aligned_dir = osp.join(uuid_dir, "aligned")
        _ensure_dir(aligned_dir)

        # Embed any missing vectors in a single batch
        to_embed_idx = [i for i, e in enumerate(embeddings) if e is None]
        if to_embed_idx:
            crops_to_embed = [aligned_list[i] for i in to_embed_idx]
            vecs = self.arc.infer(crops_to_embed)
            for k, i in enumerate(to_embed_idx):
                embeddings[i] = _l2_normalize(vecs[k])

        # Validate / normalize provided embeddings
        clean_embs: List[np.ndarray] = []
        for e in embeddings:
            e = np.asarray(e, dtype=np.float32).reshape(-1)
            if e.shape[0] != self._dim:
                raise ValueError(f"embedding shape {e.shape}, expected ({self._dim},)")
            clean_embs.append(_l2_normalize(e))

        # Write all crops
        saved: List[str] = []
        for k, (crop, emb) in enumerate(zip(aligned_list, clean_embs)):
            if crop.shape[:2] != (self.aligned_size, self.aligned_size):
                crop = cv2.resize(crop, (self.aligned_size, self.aligned_size))
            fname = f"sample_{rec.sample_count + k + 1:04d}.jpg"
            full_path = osp.join(aligned_dir, fname)
            i = 0
            while osp.exists(full_path):
                i += 1
                fname = f"sample_{rec.sample_count + k + 1:04d}_{i}.jpg"
                full_path = osp.join(aligned_dir, fname)
            ok = cv2.imwrite(full_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise IOError(f"failed to write {full_path}")
            saved.append(fname)

        # Embeddings: append all in one rewrite
        embeddings_path = osp.join(uuid_dir, "embeddings.npy")
        if osp.exists(embeddings_path):
            try:
                prev = np.load(embeddings_path).astype(np.float32)
            except Exception:
                prev = np.zeros((0, self._dim), dtype=np.float32)
        else:
            prev = np.zeros((0, self._dim), dtype=np.float32)
        new_arr = np.vstack([prev] + [e[None, :] for e in clean_embs]).astype(
            np.float32
        )
        _atomic_write_npy(embeddings_path, new_arr)

        # Centroid sum: just add all new embeddings to the running sum.
        # See IdentityRecord.centroid_sum for why we store un-normalized.
        if rec.centroid_sum is not None and prev.shape[0] > 0:
            new_sum = rec.centroid_sum.copy()
        else:
            new_sum = np.zeros(self._dim, dtype=np.float32)
        for e in clean_embs:
            new_sum = new_sum + e
        _atomic_write_npy(osp.join(uuid_dir, "centroid.npy"), new_sum)

        # Metadata
        N = prev.shape[0] + len(clean_embs)
        rec.metadata["sample_count"] = N
        rec.metadata["updated_at"] = _now_iso()
        _atomic_write_json(osp.join(uuid_dir, "metadata.json"), rec.metadata)

        rec.sample_count = N
        rec.centroid_sum = new_sum

        log.info("add_samples: uuid=%s +%d (total=%d)", uuid, len(saved), N)
        return saved

    def set_name(self, uuid: str, name: str) -> None:
        """Update the display name for a UUID. O(1), no GPU work.

        Used by ``/set_name`` when the LLM learns who an auto-enrolled
        identity actually is. The UUID never changes, so existing references
        from face_tracker tracks remain valid.
        """
        rec = self._records.get(uuid)
        if rec is None:
            raise KeyError(f"UUID not found: {uuid}")

        rec.name = name
        rec.metadata["name"] = name
        rec.metadata["updated_at"] = _now_iso()
        _atomic_write_json(
            osp.join(self.gallery_dir, uuid, "metadata.json"),
            rec.metadata,
        )
        log.info("set_name: uuid=%s name=%r", uuid, name)

    def merge_into(self, source_uuid: str, target_uuid: str) -> int:
        """Merge all samples from ``source_uuid`` into ``target_uuid``, then
        delete the source. Returns the number of samples merged.

        Use case: an anonymous auto-enrolled UUID turns out to be someone we
        already know. Rather than carry two UUIDs for the same person (one
        with front-face samples, one with side-face), fold them into one.
        Recognition then matches the union of poses/angles.

        Mechanics:
          1. Read source's ``aligned/*.jpg`` crops.
          2. Re-embed via AdaFace (cost: a few ms per crop).
          3. ``add_samples(target, crops, embeddings)`` — updates centroid +
             metadata atomically.
          4. ``forget(source)`` — moves source to ``_trash/`` so the merge
             is undoable for ~one delete cycle.

        Raises
        ------
        KeyError
            If either UUID isn't in the gallery.
        """
        src_rec = self._records.get(source_uuid)
        tgt_rec = self._records.get(target_uuid)
        if src_rec is None:
            raise KeyError(f"source UUID not found: {source_uuid}")
        if tgt_rec is None:
            raise KeyError(f"target UUID not found: {target_uuid}")
        if source_uuid == target_uuid:
            return 0

        # Read source's aligned crops from disk
        aligned_dir = osp.join(self.gallery_dir, source_uuid, "aligned")
        if not osp.isdir(aligned_dir):
            log.warning("merge_into: source %s has no aligned/ dir", source_uuid)
            self.forget(source_uuid)
            return 0

        files = sorted(f for f in os.listdir(aligned_dir) if f.endswith(".jpg"))
        if not files:
            log.warning("merge_into: source %s has no samples", source_uuid)
            self.forget(source_uuid)
            return 0

        crops: List[np.ndarray] = []
        for f in files:
            path = osp.join(aligned_dir, f)
            img = cv2.imread(path)
            if img is None:
                log.warning("merge_into: skipping unreadable %s", path)
                continue
            if img.shape[:2] != (self.aligned_size, self.aligned_size):
                img = cv2.resize(img, (self.aligned_size, self.aligned_size))
            crops.append(img)

        if not crops:
            log.warning("merge_into: no readable crops in source %s", source_uuid)
            self.forget(source_uuid)
            return 0

        # add_samples re-embeds (when embeddings=None) and updates target centroid
        saved = self.add_samples(target_uuid, crops, embeddings=None)

        # Soft-delete source (still in _trash if user wants to undo)
        self.forget(source_uuid)

        log.info(
            "merge_into: %d samples from %s → %s",
            len(saved),
            source_uuid,
            target_uuid,
        )
        return len(saved)

    def forget(self, uuid: str) -> bool:
        """Soft-delete: move UUID folder to ``_trash/``. O(1) filesystem op.

        Returns True if removed, False if UUID didn't exist. Use ``restore``
        to undo, or ``hard_delete`` / ``empty_trash`` to commit permanently.
        """
        if uuid not in self._records:
            return False
        src = osp.join(self.gallery_dir, uuid)
        if not osp.isdir(src):
            self._records.pop(uuid, None)
            return False
        dst_root = osp.join(self.gallery_dir, "_trash")
        _ensure_dir(dst_root)
        # If a trash entry with the same UUID already exists (shouldn't, but
        # safe), suffix it with a timestamp.
        dst = osp.join(dst_root, uuid)
        if osp.exists(dst):
            dst = osp.join(dst_root, f"{uuid}_{int(time.time())}")
        shutil.move(src, dst)
        self._records.pop(uuid, None)
        log.info("forget: uuid=%s → _trash", uuid)
        return True

    def restore(self, uuid: str) -> bool:
        """Move a UUID back from ``_trash/`` to active gallery.

        Returns True on success, False if not found in trash.
        """
        src = osp.join(self.gallery_dir, "_trash", uuid)
        if not osp.isdir(src):
            return False
        dst = osp.join(self.gallery_dir, uuid)
        if osp.exists(dst):
            log.warning("restore: uuid=%s already active; skipping", uuid)
            return False
        shutil.move(src, dst)
        rec = self._load(uuid)
        if rec is not None:
            self._records[uuid] = rec
        log.info("restore: uuid=%s ← _trash", uuid)
        return True

    def hard_delete(self, uuid: str) -> bool:
        """Permanently delete a UUID from ``_trash/``. Returns True if removed.

        Use ``forget`` first to move into trash, ``hard_delete`` to commit.
        For "delete immediately" workflows, call both in sequence.
        """
        target = osp.join(self.gallery_dir, "_trash", uuid)
        if osp.isdir(target):
            shutil.rmtree(target)
            log.info("hard_delete: uuid=%s removed from _trash", uuid)
            return True
        # If not in trash, check active (shouldn't happen unless caller skipped forget)
        active = osp.join(self.gallery_dir, uuid)
        if osp.isdir(active):
            shutil.rmtree(active)
            self._records.pop(uuid, None)
            log.info("hard_delete: uuid=%s removed from active (skipped trash)", uuid)
            return True
        return False

    def empty_trash(self) -> int:
        """Delete everything in ``_trash/``. Returns number of UUIDs removed."""
        trash = osp.join(self.gallery_dir, "_trash")
        if not osp.isdir(trash):
            return 0
        count = 0
        for entry in os.listdir(trash):
            p = osp.join(trash, entry)
            if osp.isdir(p):
                shutil.rmtree(p)
                count += 1
        log.info("empty_trash: removed %d entries", count)
        return count

    # ==================================================================
    # Embedding (passthrough to AdaFace)
    # ==================================================================

    def embed_aligned(self, img_112_bgr: np.ndarray) -> np.ndarray:
        """Compute L2-normalized embedding for one aligned 112×112 BGR crop.

        Exposed publicly so HttpAPI can pre-compute embeddings during selfie
        (for dedup) without writing to the gallery yet.
        """
        vec = self.arc.infer([img_112_bgr])
        if vec.ndim == 2 and vec.shape[0] == 1:
            vec = vec[0]
        return _l2_normalize(vec.astype(np.float32, copy=False))

    # ==================================================================
    # Migration from legacy gallery layout
    # ==================================================================

    def migrate_legacy(self, legacy_root: Optional[str] = None) -> int:
        """One-shot migration from the old name-keyed layout.

        Old layout:
            <legacy_root>/<name>/aligned/*.jpg  (112x112 BGR)
            <legacy_root>/<name>/raw/*.jpg      (optional, not migrated)

        For each old identity:
          - Create a new UUID via :meth:`create` with the same name.
          - Re-embed the aligned crops (one-time GPU cost) and add them via
            :meth:`add_samples`.
          - Move the original folder under ``_legacy/<name>/`` for safety.

        Parameters
        ----------
        legacy_root : str, optional
            Directory containing the old format. Defaults to
            ``self.gallery_dir`` — useful when migration is in-place (old
            non-UUID folders sit alongside new UUID folders during migration).

        Returns
        -------
        int
            Number of identities migrated.
        """
        if legacy_root is None:
            legacy_root = self.gallery_dir
        if not osp.isdir(legacy_root):
            return 0

        legacy_archive = osp.join(self.gallery_dir, "_legacy")
        _ensure_dir(legacy_archive)

        migrated = 0
        for entry in os.listdir(legacy_root):
            if entry in _RESERVED_DIRS or _is_uuid_dir(entry):
                continue
            src = osp.join(legacy_root, entry)
            if not osp.isdir(src):
                continue
            aligned_dir = osp.join(src, "aligned")
            if not osp.isdir(aligned_dir):
                log.info("migrate: skipping %s (no aligned/ dir)", entry)
                continue

            crops: List[np.ndarray] = []
            for fn in sorted(os.listdir(aligned_dir)):
                if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img = cv2.imread(osp.join(aligned_dir, fn))
                if img is None or img.size == 0:
                    continue
                if img.shape[:2] != (self.aligned_size, self.aligned_size):
                    img = cv2.resize(img, (self.aligned_size, self.aligned_size))
                crops.append(img)

            if not crops:
                log.info("migrate: skipping %s (no usable crops)", entry)
                continue

            new_uuid = self.create(
                name=entry,
                source="migration",
                migrated_from=entry,
                migrated_at=_now_iso(),
            )
            # add_samples handles batched embedding + centroid in one pass
            self.add_samples(new_uuid, crops, embeddings=None)

            # Archive the original folder
            dst = osp.join(legacy_archive, entry)
            if osp.exists(dst):
                dst = osp.join(legacy_archive, f"{entry}_{int(time.time())}")
            shutil.move(src, dst)

            migrated += 1
            log.info("migrate: %s → %s (samples=%d)", entry, new_uuid, len(crops))

        if migrated:
            log.info("Migration complete: %d identities", migrated)
        return migrated

    def has_legacy_data(self) -> bool:
        """Return True if ``gallery_dir`` contains old name-keyed folders."""
        if not osp.isdir(self.gallery_dir):
            return False
        for entry in os.listdir(self.gallery_dir):
            if entry in _RESERVED_DIRS or _is_uuid_dir(entry):
                continue
            p = osp.join(self.gallery_dir, entry)
            if osp.isdir(p) and osp.isdir(osp.join(p, "aligned")):
                return True
        return False
