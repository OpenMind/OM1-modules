from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

from .mouth_features import MouthFeature


@dataclass
class DetectorConfig:
    """Configuration for :class:`SpeakingDetector`."""

    fps: float = 30.0
    window_sec: float = 1.0

    band_low_hz: float = 2.0
    band_high_hz: float = 8.0

    score_on: float = 0.35
    score_off: float = 0.22

    min_on_frames: int = 3
    min_off_frames: int = 5

    max_invalid_frames: int = 5


@dataclass
class DetectorState:
    """State of a :class:`SpeakingDetector`."""

    speaking: bool = False
    score: float = 0.0
    _buf: Deque[float] = field(default_factory=deque)
    _pending: bool = False
    _pending_count: int = 0
    _invalid_count: int = 0


class SpeakingDetector:
    """Streaming visual VAD. Feed it one :class:`MouthFeature` per frame."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.cfg = config or DetectorConfig()
        self.state = DetectorState()
        self._window_n = max(8, int(round(self.cfg.window_sec * self.cfg.fps)))
        self.state._buf = deque(maxlen=self._window_n)

    def reset(self) -> None:
        """Reset the detector state to its initial values."""
        self.state = DetectorState()
        self.state._buf = deque(maxlen=self._window_n)

    def update(self, feature: Optional[MouthFeature]) -> DetectorState:
        """Advance one frame. Pass ``None`` or an invalid feature when no
        trustworthy face was found this frame.
        """
        st = self.state
        cfg = self.cfg

        if feature is None or not feature.valid:
            st._invalid_count += 1
            if st._invalid_count >= cfg.max_invalid_frames:
                st._buf.clear()
                self._commit(False)

            st.score = 0.0
            return st

        st._invalid_count = 0
        st._buf.append(feature.aperture)
        st.score = self._speaking_score()
        self._apply_hysteresis(st.score)

        return st

    def _speaking_score(self) -> float:
        buf = self.state._buf
        # Need enough samples for a meaningful spectrum.
        if len(buf) < self._window_n // 2:
            return 0.0

        x = np.asarray(buf, dtype=np.float64)
        x = x - x.mean()

        std = float(x.std())
        if std < 1e-6:
            return 0.0

        freqs = np.fft.rfftfreq(len(x), d=1.0 / self.cfg.fps)
        power = np.abs(np.fft.rfft(x)) ** 2

        total = float(power.sum())
        if total <= 0:
            return 0.0

        band = (freqs >= self.cfg.band_low_hz) & (freqs <= self.cfg.band_high_hz)
        band_ratio = float(power[band].sum()) / total  # fraction in speech band

        amp_gate = float(np.tanh(std / 0.03))

        return band_ratio * amp_gate

    def _apply_hysteresis(self, score: float) -> None:
        st = self.state
        cfg = self.cfg
        target = st.speaking
        if not st.speaking and score >= cfg.score_on:
            target = True
        elif st.speaking and score <= cfg.score_off:
            target = False

        if target == st.speaking:
            st._pending_count = 0
            return

        if target == st._pending:
            st._pending_count += 1
        else:
            st._pending = target
            st._pending_count = 1

        need = cfg.min_on_frames if target else cfg.min_off_frames
        if st._pending_count >= need:
            self._commit(target)

    def _commit(self, speaking: bool) -> None:
        st = self.state
        st.speaking = speaking
        st._pending = speaking
        st._pending_count = 0
