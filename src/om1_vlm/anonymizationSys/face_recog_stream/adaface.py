"""
TensorRT face embedding extractor (512-D, L2-normalized).

Supports both ArcFace and AdaFace engines — they share identical I/O:
  - Input:  (N, 3, 112, 112) BGR->RGB, normalized by (x-127.5)/128.0
  - Output: (N, 512) L2-normalized float32 embeddings

Recommended engine: AdaFace IR-101 WebFace12M (cvlface_adaface_ir101_webface12m)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

from .trt_base import TRTModule

log = logging.getLogger(__name__)

# Standard 5-point reference landmarks for 112x112 alignment
ARC_112_LMK = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def warp_face_by_5p(
    img_bgr: np.ndarray, kps_5x2: np.ndarray, out_size: int = 112
) -> np.ndarray:
    """Align a face crop to 112x112 using 5-point landmarks."""
    dst = kps_5x2.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(dst, ARC_112_LMK, method=cv2.LMEDS)
    if M is None:
        return cv2.resize(img_bgr, (out_size, out_size))
    return cv2.warpAffine(img_bgr, M, (out_size, out_size), flags=cv2.INTER_LINEAR)


class TRTFaceRecognition(TRTModule):
    """TensorRT face embedding extractor (512-D, L2-normalized).

    Works with any 112x112 -> 512-d face recognition engine:
    ArcFace, AdaFace, CosFace, etc.
    """

    def __init__(self, engine_path: str, input_name: Optional[str] = None):
        super().__init__(engine_path)
        if self.v10_api:
            self.in_name = input_name if input_name else self.input_names[0]
            self.out_name = self.output_names[0]
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
            self.out_idx = next(
                i
                for n, i in self.bindings_map.items()
                if not self.engine.binding_is_input(i)
            )

    @staticmethod
    def _preprocess_batch(imgs_bgr: List[np.ndarray]) -> np.ndarray:
        """Convert BGR faces to NCHW float32 normalized for recognition."""
        arrs = []
        for img in imgs_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            img = (img - 127.5) / 128.0
            arrs.append(np.transpose(img, (2, 0, 1)))
        return np.ascontiguousarray(np.stack(arrs, axis=0))

    def infer(self, faces_bgr: List[np.ndarray]) -> np.ndarray:
        """Run inference and return L2-normalized 512-d embeddings.

        Parameters
        ----------
        faces_bgr : list[ndarray]
            List of BGR face crops at 112x112.

        Returns
        -------
        ndarray (N, 512)
            L2-normalized float32 embeddings.
        """
        if not faces_bgr:
            return np.zeros((0, 512), dtype=np.float32)

        inp = self._preprocess_batch(faces_bgr)
        N, C, H, W = inp.shape

        d_in = self._to_gpu(inp)

        if self.v10_api:
            self.context.set_input_shape(self.in_name, (N, C, H, W))
            self.context.set_tensor_address(self.in_name, d_in.data_ptr())

            out_shape = tuple(self.context.get_tensor_shape(self.out_name))
            # Replace dynamic dims with actual batch size
            out_shape = tuple(N if d == -1 else d for d in out_shape)
            d_out = self._empty_gpu(out_shape)
            self.context.set_tensor_address(self.out_name, d_out.data_ptr())

            _extra_bufs = []  # keep references alive until execute completes
            for name in self.output_names:
                if self.context.get_tensor_address(name):
                    continue
                extra_shape = tuple(self.context.get_tensor_shape(name))
                # Replace dynamic dims (-1) with actual batch size
                extra_shape = tuple(N if d == -1 else d for d in extra_shape)
                if 0 in extra_shape:
                    continue
                extra_buf = self._empty_gpu(extra_shape)
                self.context.set_tensor_address(name, extra_buf.data_ptr())
                _extra_bufs.append(extra_buf)

            ok = self.context.execute_async_v3(self.stream_handle)
            if not ok:
                log.error("Face recognition execute_async_v3 failed")
                return np.zeros((N, 512), dtype=np.float32)
        else:
            self.context.set_binding_shape(self.in_idx, (N, C, H, W))
            out_shape = tuple(self.context.get_binding_shape(self.out_idx))
            d_out = self._empty_gpu(out_shape)

            bindings = [None] * self.engine.num_bindings
            bindings[self.in_idx] = d_in.data_ptr()
            bindings[self.out_idx] = d_out.data_ptr()

            ok = self.context.execute_async_v2(
                bindings=bindings, stream_handle=self.stream_handle
            )
            if not ok:
                log.error("Face recognition execute_async_v2 failed")
                return np.zeros((N, 512), dtype=np.float32)

        self._sync()
        out = d_out.cpu().numpy()
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out.astype(np.float32)
