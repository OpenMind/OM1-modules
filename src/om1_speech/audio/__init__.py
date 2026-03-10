from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audio_input_stream import AudioInputStream
    from .audio_output_live_stream import AudioOutputLiveStream
    from .audio_output_stream import AudioOutputStream
    from .audio_rtsp_input_stream import AudioRTSPInputStream


def __getattr__(name: str):
    # Lazily import audio helpers because some depend on optional packages (zenoh, etc.).
    mapping = {
        "AudioInputStream": ".audio_input_stream",
        "AudioOutputLiveStream": ".audio_output_live_stream",
        "AudioOutputStream": ".audio_output_stream",
        "AudioRTSPInputStream": ".audio_rtsp_input_stream",
    }
    if name in mapping:
        mod = importlib.import_module(mapping[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AudioInputStream",
    "AudioRTSPInputStream",
    "AudioOutputStream",
    "AudioOutputLiveStream",
]
