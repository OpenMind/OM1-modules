# face_recog_stream/trt_base.py
from __future__ import annotations

import os.path as osp

import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

__all__ = ["TRTModule"]


class TRTModule:
    """Lightweight TensorRT wrapper supporting TRT v8 (bindings) and v10 (named I/O)."""

    def __init__(self, engine_path: str):
        assert osp.exists(engine_path), f"engine not found: {engine_path}"
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
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

    @staticmethod
    def _malloc_bytes(nbytes: int):
        """CUDA device malloc in bytes."""
        return cuda.mem_alloc(int(nbytes))
