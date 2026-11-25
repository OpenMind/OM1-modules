import importlib
import itertools
import sys
import weakref
from unittest.mock import MagicMock

import numpy as np
import pytest

ADAFACE = "om1_vlm.anonymizationSys.face_recog_stream.adaface"
TRT_BASE = "om1_vlm.anonymizationSys.face_recog_stream.trt_base"


class FakeGpuTensor:
    """Stand-in for a CUDA tensor: unique device pointer, host copy on .cpu()."""

    _next_ptr = itertools.count(0x1000, 0x1000)

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ptr = next(self._next_ptr)

    def data_ptr(self):
        return self.ptr

    def cpu(self):
        return self

    def numpy(self):
        return np.ones(self.shape, dtype=np.float32)


class FakeContext:
    """Minimal TensorRT 10 execution context for a (N,3,112,112) -> (N,512), (N,1) engine.

    Like the real context, tensor addresses persist across calls. On execute it
    checks that every output points at a buffer that is still alive and large
    enough for the current batch.
    """

    def __init__(self, buffers):
        self.buffers = buffers
        self.addresses = {}
        self.batch = 0

    def set_input_shape(self, name, shape):
        self.batch = shape[0]

    def get_tensor_shape(self, name):
        return {"embedding": (self.batch, 512), "norm": (self.batch, 1)}[name]

    def set_tensor_address(self, name, ptr):
        self.addresses[name] = ptr

    def get_tensor_address(self, name):
        return self.addresses.get(name, 0)

    def execute_async_v3(self, stream_handle):
        for name in ("embedding", "norm"):
            ref = self.buffers.get(self.addresses.get(name))
            buf = ref() if ref is not None else None
            assert buf is not None, f"output {name!r} is bound to a freed buffer"
            assert buf.shape[0] >= self.batch, f"output {name!r} buffer too small"
        return True


@pytest.fixture
def adaface(monkeypatch):
    # adaface/trt_base import tensorrt and torch at module level; neither is
    # needed here since all GPU work is faked.
    monkeypatch.setitem(sys.modules, "tensorrt", MagicMock())
    monkeypatch.setitem(sys.modules, "torch", MagicMock())
    for name in (ADAFACE, TRT_BASE):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield importlib.import_module(ADAFACE)
    for name in (ADAFACE, TRT_BASE):
        sys.modules.pop(name, None)


def make_recognizer(adaface):
    buffers = {}

    def alloc(shape):
        t = FakeGpuTensor(shape)
        buffers[t.ptr] = weakref.ref(t)
        return t

    rec = object.__new__(adaface.TRTFaceRecognition)
    rec.v10_api = True
    rec.in_name = "input"
    rec.out_name = "embedding"
    rec.output_names = ["embedding", "norm"]
    rec.context = FakeContext(buffers)
    rec._torch_stream = MagicMock()
    rec._to_gpu = lambda arr: alloc(arr.shape)
    rec._empty_gpu = lambda shape, dtype=None: alloc(shape)
    rec._sync = lambda: None
    return rec


def faces(n):
    return [np.zeros((112, 112, 3), dtype=np.uint8) for _ in range(n)]


def test_infer_returns_normalized_embeddings(adaface):
    rec = make_recognizer(adaface)
    out = rec.infer(faces(3))
    assert out.shape == (3, 512)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)


def test_infer_rebinds_extra_outputs_on_every_call(adaface):
    rec = make_recognizer(adaface)
    # Varying batch sizes, as FaceTracker does when the number of faces changes.
    for n in (1, 4, 2, 8, 1):
        out = rec.infer(faces(n))
        assert out.shape == (n, 512)
