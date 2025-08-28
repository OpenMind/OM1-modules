"""
VILA Vision Language Model integration for om1.

This module provides integration with the VILA (Vision Language) model
for real-time video analysis and description generation.
"""

from .args import VILAArgParser
from .video_stream_input import VideoStreamInput
from .vila_processor import VILAProcessor

# __all__ = ["VILAProcessor", "VILAArgParser", "VideoStreamInput"]

# om1_vlm/__init__.py
__all__ = ["ConnectionProcessor"]

def __getattr__(name):
    if name == "ConnectionProcessor":
        from .processor import ConnectionProcessor
        return ConnectionProcessor
    raise AttributeError(name)
