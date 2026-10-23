import numpy as np
import pytest

from heartbeam.synth import skin_trace
from heartbeam.tracker import PulseTracker


def test_needs_enough_history_before_estimating():
    tracker = PulseTracker(window_s=10, min_s=5)
    for i, rgb in enumerate(skin_trace(70, seconds=3)):
        tracker.push(i / 30, rgb)
    assert tracker.estimate() is None


def test_tracks_heart_rate_from_irregular_frames():
    rng = np.random.default_rng(0)
    rgb = skin_trace(66, seconds=15, fs=60, noise=0.1)
    # Simulate a webcam that delivers roughly every other sample, unevenly.
    keep = np.sort(rng.choice(len(rgb), size=len(rgb) // 2, replace=False))
    tracker = PulseTracker(window_s=10)
    for i in keep:
        tracker.push(i / 60, rgb[i])
    estimate = tracker.estimate()
    assert estimate is not None
    assert estimate.bpm == pytest.approx(66, abs=2)
    assert tracker.wave is not None


def test_window_slides_forward():
    tracker = PulseTracker(window_s=4, min_s=2)
    for i, rgb in enumerate(skin_trace(70, seconds=10)):
        tracker.push(i / 30, rgb)
    assert tracker.seconds <= 4


def test_reset_clears_everything():
    tracker = PulseTracker(window_s=10, min_s=5)
    for i, rgb in enumerate(skin_trace(70, seconds=8)):
        tracker.push(i / 30, rgb)
    tracker.estimate()
    tracker.reset()
    assert tracker.seconds == 0
    assert tracker.wave is None
    assert tracker.estimate() is None


def test_out_of_order_frames_are_ignored():
    tracker = PulseTracker()
    tracker.push(1.0, [1, 1, 1])
    tracker.push(0.5, [2, 2, 2])
    assert tracker.seconds == 0
