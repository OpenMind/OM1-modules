from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import pycuda.driver as cuda

from .trt_base import TRTModule

# Standard ArcFace 5-point reference landmarks for 112×112 alignment
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
    """Align a face crop to 112×112 using 5-point landmarks.

    Parameters
    ----------
    img_bgr : ndarray
        Source image in BGR layout.
    kps_5x2 : ndarray (5,2)
        Landmark points in image coordinates.
    out_size : int, default 112
        Output square size in pixels.

    Returns
    -------
    ndarray (out_size, out_size, 3)
        Aligned face crop (BGR). Falls back to resized box if alignment fails.
    """
    src = ARC_112_LMK
    dst = kps_5x2.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(dst, src, method=cv2.LMEDS)
    if M is None:
        return cv2.resize(img_bgr, (out_size, out_size))
    return cv2.warpAffine(img_bgr, M, (out_size, out_size), flags=cv2.INTER_LINEAR)


class TRTArcFace(TRTModule):
    """TensorRT ArcFace embedding extractor (512-D, L2-normalized).

    Methods
    -------
    infer(faces_bgr): Compute embeddings for a batch of 112×112 BGR faces.
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
        """Convert BGR faces to NCHW float32 in ArcFace range.

        Parameters
        ----------
        imgs_bgr : list[ndarray]
            List of faces (BGR, any dtype/size). Will be converted to 112×112.

        Returns
        -------
        ndarray (N,3,112,112)
            Preprocessed batch ready for ArcFace.
        """
        arrs = []
        for img in imgs_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            img = (img - 127.5) / 128.0
            arrs.append(np.transpose(img, (2, 0, 1)))
        return np.ascontiguousarray(np.stack(arrs, axis=0))

    def infer(self, faces_bgr: List[np.ndarray]) -> np.ndarray:
        """Run ArcFace inference and return L2-normalized embeddings.

        Parameters
        ----------
        faces_bgr : list[ndarray]
            List of BGR face crops at 112×112.

        Returns
        -------
        ndarray (N,512)
            L2-normalized float32 embeddings.
        """
        if not faces_bgr:
            return np.zeros((0, 512), dtype=np.float32)
        inp = self._preprocess_batch(faces_bgr)
        N, C, H, W = inp.shape
        d_in = self._malloc_bytes(inp.nbytes)
        cuda.memcpy_htod_async(d_in, inp, self.stream)
        if self.v10_api:
            self.context.set_input_shape(self.in_name, (N, C, H, W))
            self.context.set_tensor_address(self.in_name, int(d_in))
            out_shape = tuple(self.context.get_tensor_shape(self.out_name))
            if -1 in out_shape:
                out_shape = tuple(self.context.get_tensor_shape(self.out_name))
            d_out = self._malloc_bytes(int(np.prod(out_shape) * 4))
            self.context.set_tensor_address(self.out_name, int(d_out))
            ok = self.context.execute_async_v3(self.stream.handle)
            if not ok:
                raise RuntimeError("ArcFace execute_async_v3 failed")
            out_host = np.empty(out_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(out_host, d_out, self.stream)
            self.stream.synchronize()
        else:
            self.context.set_binding_shape(self.in_idx, (N, C, H, W))
            out_shape = tuple(self.context.get_binding_shape(self.out_idx))
            d_out = self._malloc_bytes(int(np.prod(out_shape) * 4))
            bindings = [None] * self.engine.num_bindings
            bindings[self.in_idx] = int(d_in)
            bindings[self.out_idx] = int(d_out)
            ok = self.context.execute_async_v2(
                bindings=bindings, stream_handle=self.stream.handle
            )
            if not ok:
                raise RuntimeError("ArcFace execute_async_v2 failed")
            out_host = np.empty(out_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(out_host, d_out, self.stream)
            self.stream.synchronize()
        out_host /= np.linalg.norm(out_host, axis=1, keepdims=True) + 1e-9
        return out_host.astype(np.float32)
