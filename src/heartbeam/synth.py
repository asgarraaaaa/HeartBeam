"""Fake skin signals and videos with a known heart rate, used by the tests.

Colour model from de Haan & Jeanne 2013: skin tone * light intensity, plus a
small pulse component (mostly in the green channel).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SKIN_RGB = np.array([200.0, 140.0, 115.0])
# pulse strength in R, G, B
PULSE_DIRECTION = np.array([0.33, 0.77, 0.53])


def skin_trace(
    bpm: float,
    seconds: float = 20.0,
    fs: float = 30.0,
    pulse_amplitude: float = 0.003,
    flicker: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Mean skin RGB over time, shape (n, 3). flicker is a slow random change of
    the light (same on all channels), noise is in 8-bit units."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    # small variation of the rate + a harmonic, so it's not a perfect sine
    phase = 2 * np.pi * bpm / 60.0 * t + 0.15 * np.sin(2 * np.pi * 0.1 * t)
    wave = np.sin(phase) + 0.3 * np.sin(2 * phase + 0.8)

    light = 1.0 + flicker * _smooth_random(len(t), fs, rng)
    rgb = light[:, None] * SKIN_RGB * (1.0 + pulse_amplitude * wave[:, None] * PULSE_DIRECTION)
    return rgb + noise * rng.standard_normal(rgb.shape)


def write_video(
    path: str | Path,
    bpm: float,
    seconds: float = 15.0,
    fps: float = 30.0,
    size: tuple[int, int] = (320, 240),
    seed: int = 0,
) -> tuple[int, int, int, int]:
    """Writes a video with a coloured rectangle that pulses at `bpm`.
    Returns the rectangle (x, y, w, h) to use as --roi, since the face detector
    obviously won't find a face in it."""
    rng = np.random.default_rng(seed)
    w, h = size
    face = (w // 4, h // 6, w // 2, (2 * h) // 3)
    colours = skin_trace(bpm, seconds, fps, pulse_amplitude=0.01, flicker=0.02, seed=seed)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open {path} for writing")
    try:
        x, y, fw, fh = face
        for rgb in colours:
            frame = np.full((h, w, 3), 40.0)
            frame[y : y + fh, x : x + fw] = rgb[::-1]
            # noise works as dithering, otherwise 8-bit rounding kills the pulse
            frame += 4.0 * rng.standard_normal(frame.shape)
            writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    finally:
        writer.release()
    return face


def _smooth_random(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    kernel = np.hanning(max(3, int(2 * fs)))
    drift = np.convolve(rng.standard_normal(n + len(kernel)), kernel, mode="same")[:n]
    return drift / (np.abs(drift).max() or 1.0)
