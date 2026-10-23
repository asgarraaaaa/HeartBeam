"""Command line interface."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from heartbeam import overlay
from heartbeam.roi import Box, FaceFinder, mean_skin_rgb, skin_regions
from heartbeam.tracker import PulseTracker

UPDATE_EVERY_S = 0.5
WINDOW_TITLE = "heartbeam - press q to quit"


@dataclass(frozen=True)
class Reading:
    t: float
    bpm: float
    snr_db: float
    confidence: str


def measure(
    capture: cv2.VideoCapture,
    *,
    is_file: bool,
    roi: Box | None = None,
    window_s: float = 10.0,
    show: bool = False,
    mirror: bool = False,
) -> list[Reading]:
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = PulseTracker(window_s=window_s, min_s=min(5.0, window_s))
    finder = None if roi is not None else FaceFinder()
    readings: list[Reading] = []
    estimate = None
    next_update = 0.0
    frame_index = 0
    start = time.monotonic()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        # for webcams use real time, the fps they report is often wrong
        t = frame_index / fps if is_file else time.monotonic() - start
        frame_index += 1

        face = finder.update(frame) if finder is not None else None
        regions = [roi] if roi is not None else (skin_regions(face) if face is not None else [])
        rgb = mean_skin_rgb(frame, regions) if regions else None
        if rgb is None:
            tracker.reset()
            estimate = None
        else:
            tracker.push(t, rgb)

        if t >= next_update:
            next_update = t + UPDATE_EVERY_S
            estimate = tracker.estimate()
            if estimate is not None and np.isfinite(estimate.bpm):
                readings.append(
                    Reading(round(t, 2), round(estimate.bpm, 1), round(estimate.snr_db, 2), estimate.confidence)
                )

        if show:
            progress = 100.0 * min(1.0, tracker.seconds / tracker.min_s)
            cv2.imshow(WINDOW_TITLE, overlay.draw(frame, face, regions, estimate, tracker.wave, progress))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    return readings


def summarise(readings: list[Reading]) -> float | None:
    good = [r.bpm for r in readings if r.confidence != "low"]
    return median(good) if good else None


def write_csv(path: Path, readings: list[Reading]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_seconds", "bpm", "snr_db", "confidence"])
        writer.writerows((r.t, r.bpm, r.snr_db, r.confidence) for r in readings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="heartbeam",
        description="Measure heart rate from a webcam or video, without touching anyone.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    source.add_argument("--video", type=Path, help="analyse a video file instead of the webcam")
    parser.add_argument(
        "--roi", type=Box.parse, metavar="X,Y,W,H", help="measure this fixed region instead of detecting a face"
    )
    parser.add_argument("--window", type=float, default=10.0, metavar="SECONDS", help="analysis window (default: 10)")
    parser.add_argument("--csv", type=Path, metavar="PATH", help="save every reading to a CSV file")
    parser.add_argument("--headless", action="store_true", help="do not open a window, just print the result")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.video is not None:
        if not args.video.exists():
            print(f"heartbeam: no such file: {args.video}", file=sys.stderr)
            return 2
        capture = cv2.VideoCapture(str(args.video))
    else:
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print("heartbeam: could not open the video source", file=sys.stderr)
        return 1

    try:
        readings = measure(
            capture,
            is_file=args.video is not None,
            roi=args.roi,
            window_s=args.window,
            show=not args.headless,
            mirror=args.video is None,
        )
    except KeyboardInterrupt:
        readings = []
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()

    if args.csv:
        write_csv(args.csv, readings)
    bpm = summarise(readings)
    if bpm is None:
        print("heartbeam: no reliable reading. Try more light, less motion, or a longer clip.", file=sys.stderr)
        return 1
    print(f"{bpm:.1f} BPM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
