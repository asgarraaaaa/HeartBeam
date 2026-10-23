"""Face detection and skin regions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def parse(cls, text: str) -> Box:
        parts = [int(p) for p in text.split(",")]
        if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
            raise ValueError(f"expected x,y,w,h with positive size, got {text!r}")
        return cls(*parts)

    def sub(self, fx: float, fy: float, fw: float, fh: float) -> Box:
        return Box(
            x=int(self.x + fx * self.w),
            y=int(self.y + fy * self.h),
            w=max(1, int(fw * self.w)),
            h=max(1, int(fh * self.h)),
        )


def skin_regions(face: Box) -> list[Box]:
    # forehead and cheeks, eyes and mouth move too much
    return [
        face.sub(0.30, 0.08, 0.40, 0.17),  # forehead
        face.sub(0.15, 0.50, 0.20, 0.20),  # left cheek
        face.sub(0.65, 0.50, 0.20, 0.20),  # right cheek
    ]


def mean_skin_rgb(frame_bgr: np.ndarray, regions: list[Box]) -> np.ndarray | None:
    """Mean RGB of the skin pixels in the regions (rough YCrCb skin mask)."""
    h, w = frame_bgr.shape[:2]
    patches = []
    for r in regions:
        x0, y0 = max(0, r.x), max(0, r.y)
        x1, y1 = min(w, r.x + r.w), min(h, r.y + r.h)
        if x1 > x0 and y1 > y0:
            patches.append(frame_bgr[y0:y1, x0:x1].reshape(-1, 3))
    if not patches:
        return None

    pixels = np.concatenate(patches)
    ycrcb = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    skin = (ycrcb[:, 1] >= 133) & (ycrcb[:, 1] <= 173) & (ycrcb[:, 2] >= 77) & (ycrcb[:, 2] <= 127)
    if skin.sum() >= 0.2 * len(pixels):
        pixels = pixels[skin]
    return pixels.mean(axis=0)[::-1].astype(np.float64)


class FaceFinder:
    """Largest face in the frame. Runs the detector every few frames on a smaller
    image and smooths the box, otherwise the jitter shows up as colour noise."""

    def __init__(self, detect_every: int = 4, smoothing: float = 0.7, scale: float = 0.5):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"could not load face model from {path}")
        self.detect_every = detect_every
        self.smoothing = smoothing
        self.scale = scale
        self._frame = 0
        self._box: np.ndarray | None = None
        self._misses = 0

    def update(self, frame_bgr: np.ndarray) -> Box | None:
        if self._frame % self.detect_every == 0:
            found = self._detect(frame_bgr)
            if found is None:
                self._misses += 1
                if self._misses >= 3:
                    self._box = None
            else:
                self._misses = 0
                if self._box is None:
                    self._box = found
                else:
                    self._box = self.smoothing * self._box + (1 - self.smoothing) * found
        self._frame += 1
        if self._box is None:
            return None
        return Box(*(int(round(v)) for v in self._box))

    def _detect(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        small = cv2.resize(frame_bgr, None, fx=self.scale, fy=self.scale)
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        largest = max(faces, key=lambda f: f[2] * f[3])
        return np.asarray(largest, dtype=np.float64) / self.scale
