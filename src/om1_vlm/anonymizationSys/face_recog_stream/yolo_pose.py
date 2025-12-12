# face_recog_stream/yolo_pose.py
"""
TensorRT YOLO11 Pose detector for human keypoint estimation.

This module wraps a YOLO11-pose TensorRT engine and provides batched inference
for detecting humans and their 17 COCO keypoints. Designed to integrate with
the existing face anonymization pipeline.

Output format per detection:
- bbox: [x1, y1, x2, y2] in image coordinates
- conf: detection confidence
- keypoints: (17, 3) array of [x, y, visibility] for each COCO keypoint

COCO Keypoint indices:
    0: nose,         1: left_eye,      2: right_eye,     3: left_ear,
    4: right_ear,    5: left_shoulder, 6: right_shoulder,7: left_elbow,
    8: right_elbow,  9: left_wrist,   10: right_wrist,  11: left_hip,
   12: right_hip,   13: left_knee,    14: right_knee,   15: left_ankle,
   16: right_ankle
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pycuda.driver as cuda

from .trt_base import TRTModule

log = logging.getLogger(__name__)

# COCO keypoint names for reference
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

# Skeleton connections for drawing (pairs of keypoint indices)
SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # head
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # arms
    (5, 11),
    (6, 12),
    (11, 12),  # torso
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # legs
]

# Keypoint colors (BGR) - grouped by body part
KP_COLORS = {
    "head": (255, 128, 0),  # orange
    "shoulder": (0, 255, 0),  # green
    "elbow": (0, 255, 255),  # yellow
    "wrist": (255, 255, 0),  # cyan
    "hip": (255, 0, 255),  # magenta
    "knee": (128, 0, 255),  # purple
    "ankle": (0, 128, 255),  # orange-red
}


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Greedy NMS on boxes with scores.

    Parameters
    ----------
    boxes : ndarray (N, 4)
        Boxes in xyxy format.
    scores : ndarray (N,)
        Confidence scores.
    iou_thr : float
        IoU threshold for suppression.

    Returns
    -------
    list[int]
        Indices of kept detections.
    """
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
    """TensorRT YOLO11-Pose detector.

    Attributes
    ----------
    size : int
        Input image size (square).
    conf_thresh : float
        Detection confidence threshold.
    nms_thresh : float
        NMS IoU threshold.
    num_keypoints : int
        Number of keypoints (17 for COCO).
    """

    def __init__(
        self,
        engine_path: str,
        input_name: Optional[str] = None,
        size: int = 640,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.45,
    ):
        """Initialize YOLO pose detector.

        Parameters
        ----------
        engine_path : str
            Path to TensorRT engine file.
        input_name : str, optional
            Input tensor name (auto-detected if None).
        size : int
            Model input size (default 640).
        conf_thresh : float
            Detection confidence threshold.
        nms_thresh : float
            NMS IoU threshold.
        """
        super().__init__(engine_path)
        self.size = int(size)
        self.conf_thresh = float(conf_thresh)
        self.nms_thresh = float(nms_thresh)
        self.num_keypoints = 17

        # Setup I/O names
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
        """Preprocess image with letterbox padding.

        Parameters
        ----------
        img_bgr : ndarray
            Input BGR image.

        Returns
        -------
        tuple
            (input_tensor, scale, pad_w, pad_h)
        """
        H, W = img_bgr.shape[:2]
        scale = min(self.size / H, self.size / W)
        new_w, new_h = int(W * scale), int(H * scale)
        pad_w = (self.size - new_w) // 2
        pad_h = (self.size - new_h) // 2

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        # BGR -> RGB, normalize, NHWC -> NCHW
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
        """Decode YOLO pose output to detections and keypoints.

        Parameters
        ----------
        output : ndarray
            Raw model output, shape varies by YOLO version.
        scale : float
            Preprocessing scale factor.
        pad_w, pad_h : int
            Letterbox padding offsets.
        orig_h, orig_w : int
            Original image dimensions.

        Returns
        -------
        tuple
            (dets[N,5], keypoints[N,17,3]) where dets is [x1,y1,x2,y2,conf]
            and keypoints is [x, y, visibility] per point.
        """
        # YOLO11 pose output: [1, 56, num_anchors] -> transpose to [num_anchors, 56]
        # 56 = 4 (bbox xywh) + 1 (conf) + 17*3 (keypoints)
        if output.ndim == 3:
            output = output[0]  # remove batch dim
        if output.shape[0] == 56:
            output = output.T  # [56, N] -> [N, 56]

        if output.shape[1] < 56:
            log.warning(f"Unexpected output shape: {output.shape}, expected (N, 56)")
            return np.zeros((0, 5), np.float32), np.zeros((0, 17, 3), np.float32)

        # Extract components
        cx, cy, w, h = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        conf = output[:, 4]
        kp_raw = output[:, 5:56].reshape(-1, 17, 3)  # [N, 17, 3]

        # Filter by confidence
        mask = conf >= self.conf_thresh
        if not np.any(mask):
            return np.zeros((0, 5), np.float32), np.zeros((0, 17, 3), np.float32)

        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        conf = conf[mask]
        kp_raw = kp_raw[mask]

        # Convert xywh -> xyxy
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Remove letterbox padding and rescale to original image
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        # Clip to image bounds
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # Rescale keypoints
        kps = kp_raw.copy()
        kps[:, :, 0] = (kps[:, :, 0] - pad_w) / scale  # x
        kps[:, :, 1] = (kps[:, :, 1] - pad_h) / scale  # y
        # visibility stays as is (index 2)

        # Clip keypoints to image bounds
        kps[:, :, 0] = np.clip(kps[:, :, 0], 0, orig_w)
        kps[:, :, 1] = np.clip(kps[:, :, 1], 0, orig_h)

        # Stack detections
        dets = np.stack([x1, y1, x2, y2, conf], axis=1).astype(np.float32)

        # NMS
        keep = nms_boxes(dets[:, :4], dets[:, 4], self.nms_thresh)
        if not keep:
            return np.zeros((0, 5), np.float32), np.zeros((0, 17, 3), np.float32)

        dets = dets[keep]
        kps = kps[keep].astype(np.float32)

        return dets, kps

    def detect(
        self,
        img_bgr: np.ndarray,
        conf: Optional[float] = None,
        max_num: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run pose detection on a single frame.

        Parameters
        ----------
        img_bgr : ndarray
            Input BGR image.
        conf : float, optional
            Override confidence threshold.
        max_num : int
            Max detections to return (0 = all).

        Returns
        -------
        tuple
            (dets[N,5], keypoints[N,17,3]) where dets is [x1,y1,x2,y2,conf].
        """
        if conf is not None:
            self.conf_thresh = float(conf)

        H, W = img_bgr.shape[:2]
        inp, scale, pad_w, pad_h = self._preprocess(img_bgr)

        # Allocate device memory
        d_in = self._malloc_bytes(inp.nbytes)
        cuda.memcpy_htod_async(d_in, inp, self.stream)

        if self.v10_api:
            self.context.set_input_shape(self.in_name, tuple(inp.shape))
            self.context.set_tensor_address(self.in_name, int(d_in))

            out_shape = tuple(self.context.get_tensor_shape(self.out_name))
            out_size = int(np.prod(out_shape)) * 4
            d_out = self._malloc_bytes(out_size)
            self.context.set_tensor_address(self.out_name, int(d_out))

            ok = self.context.execute_async_v3(self.stream.handle)
        else:
            self.context.set_binding_shape(self.in_idx, tuple(inp.shape))
            out_shape = tuple(self.context.get_binding_shape(self.out_idx))
            out_size = int(np.prod(out_shape)) * 4
            d_out = self._malloc_bytes(out_size)

            bindings = [None] * self.engine.num_bindings
            bindings[self.in_idx] = int(d_in)
            bindings[self.out_idx] = int(d_out)

            ok = self.context.execute_async_v2(
                bindings=bindings, stream_handle=self.stream.handle
            )

        if not ok:
            log.error("YOLO pose execute_async failed")
            return np.zeros((0, 5), np.float32), np.zeros((0, 17, 3), np.float32)

        out_host = np.empty(out_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(out_host, d_out, self.stream)
        self.stream.synchronize()

        dets, kps = self._postprocess(out_host, scale, pad_w, pad_h, H, W)

        if max_num > 0 and len(dets) > max_num:
            # Keep largest by area
            areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
            idx = np.argsort(areas)[::-1][:max_num]
            dets = dets[idx]
            kps = kps[idx]

        return dets, kps
