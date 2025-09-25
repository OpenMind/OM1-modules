from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

from .trt_base import TRTModule


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode SCRFD distances to boxes (xyxy).

    Parameters
    ----------
    points : ndarray (K,2)
        Anchor centers.
    distance : ndarray (K,4)
        L, T, R, B distances per anchor.

    Returns
    -------
    ndarray (K,4)
        Boxes in xyxy order.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode SCRFD keypoint distances to absolute coords.

    Parameters
    ----------
    points : ndarray (K,2)
        Anchor centers.
    distance : ndarray (K,10)
        Offsets for 5 points (x,y)*5.

    Returns
    -------
    ndarray (K,10)
        Absolute coordinates (x1,y1,x2,y2,...,x5,y5).
    """
    preds: List[np.ndarray] = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def nms(dets: np.ndarray, iou_thr: float) -> List[int]:
    """Perform greedy NMS on detections.

    Parameters
    ----------
    dets : ndarray (N,5)
        Boxes with scores.
    iou_thr : float
        IoU threshold for suppression.

    Returns
    -------
    list[int]
        Kept indices in descending score order.
    """
    if dets.size == 0:
        return []
    x1, y1, x2, y2, sc = dets.T
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = sc.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return keep


class TRTSCRFD(TRTModule):
    """TensorRT SCRFD detector with optional keypoints output.

    Parameters
    ----------
    engine_path : str
        Path to SCRFD TensorRT engine.
    input_name : str | None
        Explicit input tensor name (TRT < 10 may use binding names).
    size : int
        Square model size (e.g., 640).
    """

    def __init__(
        self, engine_path: str, input_name: Optional[str] = None, size: int = 640
    ):
        super().__init__(engine_path)
        self.size = int(size)
        self.nms_thresh = 0.4
        self.conf_thresh = 0.5
        if self.v10_api:
            self.in_name = (
                input_name
                if (input_name and input_name in self.input_names)
                else self.input_names[0]
            )
            self.out_names = {n: n for n in self.output_names}
        else:
            self.in_name = (
                input_name
                if input_name
                else next(
                    n
                    for n, i in self.bindings_map.items()
                    if self.engine.binding_is_input(i)
                )
            )
            self.in_idx = self.bindings_map[self.in_name]
            self.out_names = {
                n: self.bindings_map[n]
                for n in self.bindings_map
                if not self.engine.binding_is_input(self.bindings_map[n])
            }
        # Discover strides by output tensor names
        self.strides = [
            s
            for s in (8, 16, 32, 64, 128)
            if any(f"score_{s}" in n for n in self.out_names)
        ] or [8, 16, 32]
        self.num_anchors = 2

    @staticmethod
    def _preprocess_letterbox(img_bgr: np.ndarray, size: int):
        """Resize+pad to square, normalize to SCRFD input (NCHW float32).

        Parameters
        ----------
        img_bgr : ndarray
            Source BGR image.
        size : int
            Target square size.

        Returns
        -------
        tuple
            (input_tensor[N,3,H,W], scale_factor)
        """
        H, W = img_bgr.shape[:2]
        model_w = model_h = size
        im_ratio = H / float(W)
        model_ratio = model_h / float(model_w)
        if im_ratio > model_ratio:
            new_h = model_h
            new_w = int(round(new_h / im_ratio))
        else:
            new_w = model_w
            new_h = int(round(new_w * im_ratio))
        scale = float(new_h) / H
        resized = cv2.resize(img_bgr, (new_w, new_h))
        det_img = np.zeros((model_h, model_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w] = resized
        rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        chw = np.transpose(rgb, (2, 0, 1))[None, ...].copy()
        return chw, scale

    def detect(
        self, img_bgr: np.ndarray, conf: float = 0.5, max_num: int = 0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run SCRFD detection on a single frame.

        Parameters
        ----------
        img_bgr : ndarray
            Source image in BGR.
        conf : float, default 0.5
            Confidence threshold before NMS.
        max_num : int, default 0
            Keep at most this many detections (0 = all).

        Returns
        -------
        tuple
            (dets[N,5], kpss[N,5,2] or None), both float32.
        """
        self.conf_thresh = float(conf)
        size = self.size
        inp, scale = self._preprocess_letterbox(img_bgr, size)
        H0, W0 = img_bgr.shape[:2]

        def K_for_stride(s: int) -> int:
            h = size // s
            w = size // s
            return h * w * self.num_anchors

        host_scores, host_bboxes, host_kps = {}, {}, {}
        for s in self.strides:
            K = K_for_stride(s)
            host_scores[s] = np.empty((1, K, 1), dtype=np.float32)
            host_bboxes[s] = np.empty((1, K, 4), dtype=np.float32)
            host_kps[s] = np.empty((1, K, 10), dtype=np.float32)

        d_in = self._malloc_bytes(inp.nbytes)

        def get_name(tag: str, s: int) -> Optional[str]:
            cand = f"{tag}_{s}"
            for n in self.out_names:
                if n.endswith(cand) or cand in n:
                    return n
            return None

        # Set up I/O
        if self.v10_api:
            self.context.set_input_shape(self.in_name, tuple(inp.shape))
            self.context.set_tensor_address(self.in_name, int(d_in))
            d_scores, d_bboxes, d_kps = {}, {}, {}
            for s in self.strides:
                ns = get_name("score", s)
                nb = get_name("bbox", s)
                nk = get_name("kps", s)
                if ns:
                    d_scores[s] = self._malloc_bytes(host_scores[s].nbytes)
                    self.context.set_tensor_address(ns, int(d_scores[s]))
                if nb:
                    d_bboxes[s] = self._malloc_bytes(host_bboxes[s].nbytes)
                    self.context.set_tensor_address(nb, int(d_bboxes[s]))
                if nk:
                    d_kps[s] = self._malloc_bytes(host_kps[s].nbytes)
                    self.context.set_tensor_address(nk, int(d_kps[s]))
            # Bind any remaining outputs (strict in TRT10)
            if not hasattr(self, "_scratch_out"):
                self._scratch_out = {}
            for name in self.output_names:
                if self.context.get_tensor_address(name):
                    continue
                shape = tuple(self.context.get_tensor_shape(name))
                if -1 in shape:
                    shape = tuple(self.context.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
                if nbytes == 0:
                    continue
                buf = self._malloc_bytes(nbytes)
                self.context.set_tensor_address(name, int(buf))
                self._scratch_out[name] = buf
        else:
            self.context.set_binding_shape(self.in_idx, tuple(inp.shape))
            bindings = [None] * self.engine.num_bindings
            bindings[self.in_idx] = int(d_in)
            d_scores, d_bboxes, d_kps = {}, {}, {}
            for s in self.strides:
                ns = get_name("score", s)
                nb = get_name("bbox", s)
                nk = get_name("kps", s)
                if ns:
                    idx = self.out_names[ns]
                    d_scores[s] = self._malloc_bytes(host_scores[s].nbytes)
                    bindings[idx] = int(d_scores[s])
                if nb:
                    idx = self.out_names[nb]
                    d_bboxes[s] = self._malloc_bytes(host_bboxes[s].nbytes)
                    bindings[idx] = int(d_bboxes[s])
                if nk:
                    idx = self.out_names[nk]
                    d_kps[s] = self._malloc_bytes(host_kps[s].nbytes)
                    bindings[idx] = int(d_kps[s])

        # Run
        cuda.memcpy_htod_async(d_in, inp, self.stream)
        ok = (
            self.context.execute_async_v3(self.stream.handle)
            if self.v10_api
            else self.context.execute_async_v2(
                bindings=bindings, stream_handle=self.stream.handle
            )
        )
        if not ok:
            raise RuntimeError("SCRFD execute_async failed")

        # Copy outputs
        for s in self.strides:
            if s in d_scores:
                cuda.memcpy_dtoh_async(host_scores[s], d_scores.get(s), self.stream)
            if s in d_bboxes:
                cuda.memcpy_dtoh_async(host_bboxes[s], d_bboxes.get(s), self.stream)
            if s in d_kps:
                cuda.memcpy_dtoh_async(host_kps[s], d_kps.get(s), self.stream)
        self.stream.synchronize()

        # Decode
        scores_list, bboxes_list, kpss_list = [], [], []
        for s in self.strides:
            scores = host_scores[s].reshape(-1)
            bboxes = host_bboxes[s].reshape(-1, 4) * float(s)
            kps = host_kps[s].reshape(-1, 10) * float(s)
            h = size // s
            w = size // s
            centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            centers = (centers * s).reshape(-1, 2)
            if self.num_anchors > 1:
                centers = np.repeat(centers, self.num_anchors, axis=0)
            pos = np.where(scores >= self.conf_thresh)[0]
            if pos.size == 0:
                continue
            boxes_xyxy = distance2bbox(centers, bboxes)
            scores_list.append(scores[pos, None])
            bboxes_list.append(boxes_xyxy[pos])
            kpss_list.append(distance2kps(centers, kps)[pos].reshape(-1, 5, 2))

        if not scores_list:
            return np.zeros((0, 5), dtype=np.float32), None

        boxes = np.vstack(bboxes_list) / float(scale)
        scores = np.vstack(scores_list).ravel()
        kpss = (np.vstack(kpss_list) / float(scale)) if kpss_list else None

        order = scores.argsort()[::-1]
        pre_det = np.hstack([boxes, scores[:, None]])[order]
        keep = nms(pre_det, self.nms_thresh)
        det = pre_det[keep]
        if kpss is not None:
            kpss = kpss[order][keep]

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            Hc, Wc = H0 // 2, W0 // 2
            offs = np.vstack(
                [(det[:, 0] + det[:, 2]) * 0.5 - Wc, (det[:, 1] + det[:, 3]) * 0.5 - Hc]
            )
            val = area - np.sum(offs * offs, axis=0) * 2.0
            idx = np.argsort(val)[::-1][:max_num]
            det = det[idx]
            if kpss is not None:
                kpss = kpss[idx]

        return det.astype(np.float32), (
            kpss.astype(np.float32) if kpss is not None else None
        )
