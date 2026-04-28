"""TensorRT YOLO11 Pose detector — uses torch.cuda for GPU memory."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .trt_base import TRTModule

log = logging.getLogger(__name__)

COCO_KP_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Apply non-maximum suppression on bounding boxes."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return keep


class TRTYOLOPose(TRTModule):
    """TensorRT YOLO11-Pose detector."""

    def __init__(
        self,
        engine_path: str,
        input_name: Optional[str] = None,
        size: int = 640,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.45,
    ):
        super().__init__(engine_path)
        self.size = int(size)
        self.conf_thresh = float(conf_thresh)
        self.nms_thresh = float(nms_thresh)
        self.num_keypoints = 17

        if self.v10_api:
            self.in_name = input_name or self.input_names[0]
            self.out_name = self.output_names[0]
        else:
            self.in_name = input_name or next(
                n
                for n, i in self.bindings_map.items()
                if self.engine.binding_is_input(i)
            )
            self.in_idx = self.bindings_map[self.in_name]
            self.out_idx = next(
                i
                for n, i in self.bindings_map.items()
                if not self.engine.binding_is_input(i)
            )

    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        H, W = img_bgr.shape[:2]
        scale = min(self.size / H, self.size / W)
        new_w, new_h = int(W * scale), int(H * scale)
        pad_w = (self.size - new_w) // 2
        pad_h = (self.size - new_h) // 2

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(chw), scale, pad_w, pad_h

    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad_w: int,
        pad_h: int,
        orig_h: int,
        orig_w: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if output.ndim == 3:
            output = output[0]
        if output.shape[0] == 56:
            output = output.T

        if output.shape[1] < 56:
            log.warning(f"Unexpected output shape: {output.shape}")
            return np.zeros((0, 5), np.float32), np.zeros((0, 17, 3), np.float32)

        cx, cy, w, h = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        conf = output[:, 4]
        kp_raw = output[:, 5:56].reshape(-1, 17, 3)

        mask = conf >= self.conf_thresh
        if not np.any(mask):
            return np.zeros((0, 5), np.float32), np.zeros((0, 17, 3), np.float32)

        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        conf = conf[mask]
        kp_raw = kp_raw[mask]

        x1 = np.clip((cx - w / 2 - pad_w) / scale, 0, orig_w)
        y1 = np.clip((cy - h / 2 - pad_h) / scale, 0, orig_h)
        x2 = np.clip((cx + w / 2 - pad_w) / scale, 0, orig_w)
        y2 = np.clip((cy + h / 2 - pad_h) / scale, 0, orig_h)

        kps = kp_raw.copy()
        kps[:, :, 0] = np.clip((kps[:, :, 0] - pad_w) / scale, 0, orig_w)
        kps[:, :, 1] = np.clip((kps[:, :, 1] - pad_h) / scale, 0, orig_h)

        dets = np.stack([x1, y1, x2, y2, conf], axis=1).astype(np.float32)
        keep = nms_boxes(dets[:, :4], dets[:, 4], self.nms_thresh)
        if not keep:
            return np.zeros((0, 5), np.float32), np.zeros((0, 17, 3), np.float32)

        return dets[keep], kps[keep].astype(np.float32)

    def detect(
        self,
        img_bgr: np.ndarray,
        conf: Optional[float] = None,
        max_num: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect poses and return bounding boxes with keypoints."""
        if conf is not None:
            self.conf_thresh = float(conf)

        H, W = img_bgr.shape[:2]
        inp, scale, pad_w, pad_h = self._preprocess(img_bgr)

        d_in = self._to_gpu(inp)

        if self.v10_api:
            self.context.set_input_shape(self.in_name, tuple(inp.shape))
            self.context.set_tensor_address(self.in_name, d_in.data_ptr())

            out_shape = tuple(self.context.get_tensor_shape(self.out_name))
            d_out = self._empty_gpu(out_shape)
            self.context.set_tensor_address(self.out_name, d_out.data_ptr())

            ok = self.context.execute_async_v3(self.stream_handle)
        else:
            self.context.set_binding_shape(self.in_idx, tuple(inp.shape))
            out_shape = tuple(self.context.get_binding_shape(self.out_idx))
            d_out = self._empty_gpu(out_shape)

            bindings = [None] * self.engine.num_bindings
            bindings[self.in_idx] = d_in.data_ptr()
            bindings[self.out_idx] = d_out.data_ptr()

            ok = self.context.execute_async_v2(
                bindings=bindings, stream_handle=self.stream_handle
            )

        if not ok:
            log.error("YOLO pose execute_async failed")
            return np.zeros((0, 5), np.float32), np.zeros((0, 17, 3), np.float32)

        self._sync()
        out_host = d_out.cpu().numpy()

        dets, kps = self._postprocess(out_host, scale, pad_w, pad_h, H, W)

        if max_num > 0 and len(dets) > max_num:
            areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
            idx = np.argsort(areas)[::-1][:max_num]
            dets, kps = dets[idx], kps[idx]

        return dets, kps
