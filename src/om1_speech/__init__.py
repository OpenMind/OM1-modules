from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .riva import ASRProcessor, AudioDeviceInput, AudioStreamInput, TTSProcessor
    from .audio import (
        AudioInputStream,
        AudioOutputLiveStream,
        AudioOutputStream,
        AudioRTSPInputStream,
    )


def __getattr__(name: str):
    # Keep package import lightweight: some audio helpers depend on optional deps
    # (e.g., zenoh). Only import submodules when explicitly requested.
    if name in {
        "ASRProcessor",
        "TTSProcessor",
        "AudioDeviceInput",
        "AudioStreamInput",
    }:
        mod = importlib.import_module(".riva", __name__)
        return getattr(mod, name)

    if name in {
        "AudioInputStream",
        "AudioRTSPInputStream",
        "AudioOutputStream",
        "AudioOutputLiveStream",
    }:
        mod = importlib.import_module(".audio", __name__)
        return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ASRProcessor",
    "TTSProcessor",
    "AudioDeviceInput",
    "AudioStreamInput",
    "AudioInputStream",
    "AudioRTSPInputStream",
    "AudioOutputStream",
    "AudioOutputLiveStream",
]
