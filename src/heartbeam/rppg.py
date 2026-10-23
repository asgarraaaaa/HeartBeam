"""rPPG: estimate heart rate from small colour changes of the skin.

Pipeline: mean skin RGB -> POS -> band-pass -> peak of the spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

MIN_BPM = 42.0
MAX_BPM = 240.0

# POS projection (Wang et al. 2017). Both rows are orthogonal to [1, 1, 1],
# so changes in brightness that hit all channels equally cancel out.
_POS_PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])


@dataclass(frozen=True)
class Estimate:
    bpm: float
    snr_db: float

    @property
    def confidence(self) -> str:
        if self.snr_db >= 3.0:
            return "high"
        if self.snr_db >= 0.0:
            return "medium"
        return "low"


def pos(rgb: np.ndarray, fs: float, window_s: float = 1.6) -> np.ndarray:
    """POS pulse extraction. rgb has shape (n, 3), one mean colour per frame."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape (n, 3), got {rgb.shape}")
    n = len(rgb)
    win = max(2, int(round(window_s * fs)))
    if n < win:
        raise ValueError(f"need at least {win} samples, got {n}")

    pulse = np.zeros(n)
    for start in range(n - win + 1):
        block = rgb[start : start + win]
        mean = block.mean(axis=0)
        if np.any(mean <= 0):
            continue
        s = _POS_PROJECTION @ (block / mean).T
        std_2 = s[1].std()
        alpha = s[0].std() / std_2 if std_2 > 1e-12 else 0.0
        p = s[0] + alpha * s[1]
        pulse[start : start + win] += p - p.mean()
    return pulse


def bandpass(x: np.ndarray, fs: float, order: int = 3) -> np.ndarray:
    nyquist = fs / 2.0
    low = MIN_BPM / 60.0 / nyquist
    high = min(MAX_BPM / 60.0, 0.95 * nyquist) / nyquist
    sos = sps.butter(order, [low, high], btype="band", output="sos")
    return sps.sosfiltfilt(sos, x)


def estimate_bpm(pulse: np.ndarray, fs: float) -> Estimate:
    # SNR as in de Haan & Jeanne 2013: power around the peak and its first
    # harmonic is signal, the rest of the band is noise.
    pulse = np.asarray(pulse, dtype=np.float64)
    # zero padding, gives ~0.1 bpm resolution at 30 fps
    nfft = max(1 << 14, 1 << int(np.ceil(np.log2(len(pulse)))))
    freqs, psd = sps.periodogram(pulse, fs, window="hann", nfft=nfft, detrend="linear")

    band = (freqs >= MIN_BPM / 60.0) & (freqs <= MAX_BPM / 60.0)
    if not band.any() or psd[band].sum() <= 0:
        return Estimate(bpm=float("nan"), snr_db=float("-inf"))

    band_idx = np.flatnonzero(band)
    peak = band_idx[np.argmax(psd[band])]
    f0 = _refine_peak(freqs, psd, peak)

    near_peak = np.abs(freqs - f0) <= 0.1
    near_harmonic = np.abs(freqs - 2 * f0) <= 0.2
    signal_mask = band & (near_peak | near_harmonic)
    noise_mask = band & ~signal_mask
    signal_power = psd[signal_mask].sum()
    noise_power = psd[noise_mask].sum()
    snr_db = 10.0 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")

    return Estimate(bpm=float(f0 * 60.0), snr_db=float(snr_db))


def heart_rate(rgb: np.ndarray, fs: float) -> Estimate:
    return estimate_bpm(bandpass(pos(rgb, fs), fs), fs)


def resample_uniform(t: np.ndarray, values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    # webcams don't give frames at a constant rate, the FFT needs one
    t = np.asarray(t, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    resampled = np.column_stack([np.interp(grid, t, values[:, c]) for c in range(values.shape[1])])
    return grid, resampled


def _refine_peak(freqs: np.ndarray, psd: np.ndarray, i: int) -> float:
    # parabolic interpolation around the peak
    if i <= 0 or i >= len(psd) - 1:
        return float(freqs[i])
    a, b, c = np.log(psd[i - 1 : i + 2] + 1e-30)
    denom = a - 2 * b + c
    offset = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return float(freqs[i] + offset * (freqs[1] - freqs[0]))
