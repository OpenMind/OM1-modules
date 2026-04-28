# face_recog_stream/trt_base.py
"""Lightweight TensorRT wrapper using torch.cuda for GPU memory management.

Replaces pycuda dependency with PyTorch — no extra install needed.
"""

from __future__ import annotations

import os.path as osp

import numpy as np
import tensorrt as trt
import torch

__all__ = ["TRTModule", "GpuBuffer"]


class GpuBuffer:
    """GPU memory buffer backed by a torch tensor.

    Supports ``int(buf)`` to get raw device pointer for TRT,
    and ``buf.htod(np_arr)`` / ``buf.dtoh(np_arr)`` for copies.
    """

    __slots__ = ("_tensor",)

    def __init__(self, nbytes: int):
        self._tensor = torch.empty(nbytes, dtype=torch.uint8, device="cuda")

    def __int__(self) -> int:
        """Return raw device pointer for TensorRT."""
        return self._tensor.data_ptr()

    @property
    def ptr(self) -> int:
        """Return raw device pointer."""
        return self._tensor.data_ptr()

    def htod(self, host_np: np.ndarray, stream: torch.cuda.Stream) -> None:
        """Copy host numpy array into this GPU buffer (async)."""
        src = torch.from_numpy(host_np).contiguous()
        with torch.cuda.stream(stream):
            self._tensor[: src.nbytes].copy_(
                src.view(torch.uint8).reshape(-1), non_blocking=True
            )

    def dtoh(self, host_np: np.ndarray, stream: torch.cuda.Stream) -> None:
        """Copy this GPU buffer into host numpy array (async)."""
        nbytes = host_np.nbytes
        with torch.cuda.stream(stream):
            tmp = self._tensor[:nbytes].cpu()
        np.copyto(
            host_np,
            np.frombuffer(tmp.numpy().tobytes(), dtype=host_np.dtype).reshape(
                host_np.shape
            ),
        )


class TRTModule:
    """Lightweight TensorRT wrapper supporting TRT v8 (bindings) and v10 (named I/O).

    Uses torch.cuda for GPU memory management — no pycuda dependency.
    """

    def __init__(self, engine_path: str):
        assert osp.exists(engine_path), f"engine not found: {engine_path}"
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # torch CUDA stream — .cuda_stream gives raw cudaStream_t handle
        self._torch_stream = torch.cuda.Stream()
        self.stream = self._torch_stream  # for subclass access

        self.v10_api = hasattr(self.engine, "num_io_tensors")
        if self.v10_api:
            self.io_names = [
                self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)
            ]
            self.input_names = [
                n
                for n in self.io_names
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
            ]
            self.output_names = [
                n
                for n in self.io_names
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
            ]
        else:
            self.bindings_map = {
                self.engine.get_binding_name(i): i
                for i in range(self.engine.num_bindings)
            }

    @property
    def stream_handle(self) -> int:
        """Raw CUDA stream handle for TRT execute_async calls."""
        return self._torch_stream.cuda_stream

    def _malloc_bytes(self, nbytes: int) -> GpuBuffer:
        """Allocate GPU buffer. Returns GpuBuffer (int(buf) gives device ptr)."""
        return GpuBuffer(int(nbytes))

    def _to_gpu(self, host_np: np.ndarray) -> torch.Tensor:
        """Copy a numpy array to GPU, return torch.Tensor (keeps memory alive).

        Use ``tensor.data_ptr()`` to get the raw pointer for TRT.
        """
        with torch.cuda.stream(self._torch_stream):
            return torch.from_numpy(host_np).cuda(non_blocking=True)

    def _empty_gpu(self, shape, dtype=torch.float32) -> torch.Tensor:
        """Allocate an empty GPU tensor for TRT output."""
        return torch.empty(shape, dtype=dtype, device="cuda")

    def _sync(self) -> None:
        """Synchronize the CUDA stream."""
        self._torch_stream.synchronize()
