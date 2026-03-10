from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from . import http, logging
from .logging import LoggingConfig, get_logging_config, setup_logging
from .singleton import singleton

if TYPE_CHECKING:
    from . import ws as ws


def __getattr__(name: str):
    if name == "ws":
        # Lazy import so environments without the optional `websockets` dependency
        # can still import `om1_utils` (e.g., when only HTTP/logging are used).
        return importlib.import_module(".ws", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ws",
    "http",
    "logging",
    "singleton",
    "setup_logging",
    "get_logging_config",
    "LoggingConfig",
]
