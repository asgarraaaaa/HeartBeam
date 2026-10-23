"""Draws the bpm, regions and waveform on the frame."""

from __future__ import annotations

import cv2
import numpy as np

from heartbeam.roi import Box
from heartbeam.rppg import Estimate

_COLOURS = {"high": (90, 200, 90), "medium": (60, 190, 230), "low": (80, 80, 220)}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw(
    frame: np.ndarray,
    face: Box | None,
    regions: list[Box],
    estimate: Estimate | None,
    wave: np.ndarray | None,
    warming_up: float,
) -> np.ndarray:
    out = frame.copy()
    if face is not None:
        cv2.rectangle(out, (face.x, face.y), (face.x + face.w, face.y + face.h), (200, 200, 200), 1)
    for r in regions:
        cv2.rectangle(out, (r.x, r.y), (r.x + r.w, r.y + r.h), (60, 220, 255), 1)

    _panel(out, estimate, wave, warming_up, has_face=face is not None or bool(regions))
    return out


def _panel(
    out: np.ndarray, estimate: Estimate | None, wave: np.ndarray | None, warming_up: float, has_face: bool
) -> None:
    h, w = out.shape[:2]
    top = h - 90
    shade = out[top:h].astype(np.float32) * 0.35
    out[top:h] = shade.astype(np.uint8)

    if not has_face:
        cv2.putText(out, "Looking for a face...", (16, top + 52), _FONT, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
        return
    if estimate is None:
        label = f"Hold still... {warming_up:0.0f}%"
        cv2.putText(out, label, (16, top + 52), _FONT, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
        return

    colour = _COLOURS[estimate.confidence]
    beat = 0.5 + 0.5 * float(wave[-1] / (np.abs(wave).max() or 1.0)) if wave is not None else 0.5
    cv2.circle(out, (36, top + 45), int(12 + 8 * beat), colour, -1, cv2.LINE_AA)
    cv2.putText(out, f"{estimate.bpm:5.1f} BPM", (64, top + 58), _FONT, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"signal {estimate.confidence}", (68, top + 80), _FONT, 0.45, colour, 1, cv2.LINE_AA)

    if wave is not None and len(wave) > 1:
        _waveform(out, wave[-150:], x0=260, y0=top + 10, width=w - 276, height=70, colour=colour)


def _waveform(out: np.ndarray, wave: np.ndarray, x0: int, y0: int, width: int, height: int, colour) -> None:
    if width <= 10:
        return
    scale = np.abs(wave).max() or 1.0
    xs = x0 + np.linspace(0, width, len(wave))
    ys = y0 + height / 2 - (wave / scale) * (height / 2 - 4)
    points = np.column_stack([xs, ys]).astype(np.int32)
    cv2.polylines(out, [points], isClosed=False, color=colour, thickness=2, lineType=cv2.LINE_AA)
