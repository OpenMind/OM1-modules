from __future__ import annotations

import os
import os.path as osp
from typing import List, Tuple

import cv2
import numpy as np

from .arcface import TRTArcFace, warp_face_by_5p
from .scrfd import TRTSCRFD
from .utils import list_images


def iter_identities(gallery_root: str) -> List[Tuple[str, List[str]]]:
    """Enumerate identity folders and their images.

    Parameters
    ----------
    gallery_root : str
        Root containing one subfolder per identity.

    Returns
    -------
    list[tuple[str, list[str]]]
        (identity_name, image_paths) pairs, excluding empty folders.
    """
    ids: List[Tuple[str, List[str]]] = []
    if not osp.isdir(gallery_root):
        return ids
    for name in sorted(os.listdir(gallery_root)):
        d = osp.join(gallery_root, name)
        if osp.isdir(d):
            imgs = list_images(d)
            if imgs:
                ids.append((name, imgs))
    return ids


def build_gallery_embeddings(
    gallery_root: str, scrfd: TRTSCRFD, arc: TRTArcFace, det_conf: float
):
    """Build per-identity mean embeddings from a folder gallery.

    Parameters
    ----------
    gallery_root : str
        Root path with one folder per identity.
    scrfd : TRTSCRFD
        Face detector instance.
    arc : TRTArcFace
        ArcFace embedder instance.
    det_conf : float
        Detection confidence threshold.

    Returns
    -------
    tuple
        (features[N,512], labels[N]) where N is number of identities.
    """
    feats: List[np.ndarray] = []
    labels: List[str] = []
    identities = iter_identities(gallery_root)
    if not identities:
        raise RuntimeError(f"No identities under: {gallery_root}")

    for label, paths in identities:
        emb_list = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            dets, kpss = scrfd.detect(img, conf=det_conf)
            if dets is None or dets.shape[0] == 0:
                continue
            idx = int(np.argmax((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])))
            if kpss is not None:
                crop = warp_face_by_5p(img, kpss[idx], 112)
            else:
                x1, y1, x2, y2, _ = dets[idx].astype(int)
                face = img[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                if face.size == 0:
                    continue
                crop = cv2.resize(face, (112, 112))
            feat = arc.infer([crop])
            if feat.shape[0] == 1:
                emb_list.append(feat[0])
        if emb_list:
            m = np.mean(np.stack(emb_list, axis=0), axis=0)
            m /= np.linalg.norm(m) + 1e-9
            feats.append(m.astype(np.float32))
            labels.append(label)

    if not feats:
        raise RuntimeError("No valid gallery embeddings.")

    return np.stack(feats, axis=0).astype(np.float32), labels
