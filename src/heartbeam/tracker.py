"""Heart rate over a sliding window of frames."""

from __future__ import annotations

from collections import deque
from statistics import median

import numpy as np

from heartbeam.rppg import Estimate, bandpass, estimate_bpm, pos, resample_uniform


class PulseTracker:
    """Keeps the last `window_s` seconds of skin colours and estimates the heart rate.
    The returned bpm is the median of the last `smoothing` good readings."""

    def __init__(self, window_s: float = 10.0, min_s: float = 5.0, fs: float = 30.0, smoothing: int = 5):
        if min_s > window_s:
            raise ValueError("min_s cannot be longer than window_s")
        self.window_s = window_s
        self.min_s = min_s
        self.fs = fs
        self._times: deque[float] = deque()
        self._colours: deque[np.ndarray] = deque()
        self._recent: deque[Estimate] = deque(maxlen=smoothing)
        self.wave: np.ndarray | None = None

    def push(self, t: float, rgb: np.ndarray) -> None:
        if self._times and t <= self._times[-1]:
            return
        self._times.append(t)
        self._colours.append(np.asarray(rgb, dtype=np.float64))
        while self._times and t - self._times[0] > self.window_s:
            self._times.popleft()
            self._colours.popleft()

    def reset(self) -> None:
        self._times.clear()
        self._colours.clear()
        self._recent.clear()
        self.wave = None

    @property
    def seconds(self) -> float:
        return self._times[-1] - self._times[0] if len(self._times) > 1 else 0.0

    def pulse(self) -> np.ndarray | None:
        if self.seconds < self.min_s:
            return None
        _, rgb = resample_uniform(np.array(self._times), np.array(self._colours), self.fs)
        return bandpass(pos(rgb, self.fs), self.fs)

    def estimate(self) -> Estimate | None:
        wave = self.wave = self.pulse()
        if wave is None:
            return None
        raw = estimate_bpm(wave, self.fs)
        if raw.confidence != "low":
            self._recent.append(raw)
        if not self._recent:
            return raw
        return Estimate(
            bpm=median(e.bpm for e in self._recent),
            snr_db=raw.snr_db,
        )
