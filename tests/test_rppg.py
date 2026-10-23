import numpy as np
import pytest

from heartbeam.rppg import bandpass, estimate_bpm, heart_rate, pos, resample_uniform
from heartbeam.synth import skin_trace

FS = 30.0


@pytest.mark.parametrize("bpm", [48, 60, 72, 95, 130, 170])
def test_recovers_known_heart_rate(bpm):
    rgb = skin_trace(bpm, seconds=20, fs=FS, noise=0.1, seed=bpm)
    estimate = heart_rate(rgb, FS)
    assert estimate.bpm == pytest.approx(bpm, abs=1.5)
    assert estimate.confidence == "high"


def test_survives_strong_light_flicker():
    # Brightness drifts by 5% while the pulse is only 0.3%: raw green is swamped,
    # but POS projects intensity changes away.
    rgb = skin_trace(75, seconds=20, fs=FS, flicker=0.05, noise=0.1, seed=1)
    assert heart_rate(rgb, FS).bpm == pytest.approx(75, abs=1.5)


def test_green_channel_alone_is_fooled_by_flicker():
    # The classic approach (just the green channel) locks onto the lamp instead.
    rgb = skin_trace(75, seconds=20, fs=FS, flicker=0.05, noise=0.1, seed=2)
    green = estimate_bpm(bandpass(rgb[:, 1] - rgb[:, 1].mean(), FS), FS)
    with_pos = estimate_bpm(bandpass(pos(rgb, FS), FS), FS)
    assert abs(green.bpm - 75) > 10
    assert with_pos.bpm == pytest.approx(75, abs=1.5)


@pytest.mark.parametrize("seed", range(10))
def test_pure_noise_is_reported_as_low_confidence(seed):
    rng = np.random.default_rng(seed)
    rgb = 150 + rng.standard_normal((300, 3))
    assert heart_rate(rgb, FS).confidence == "low"


def test_pos_ignores_uniform_brightness_change():
    t = np.arange(300) / FS
    brightness = 1 + 0.2 * np.sin(2 * np.pi * 1.2 * t)
    rgb = brightness[:, None] * np.array([200.0, 140.0, 115.0])
    assert np.abs(pos(rgb, FS)).max() < 1e-9


def test_pos_rejects_bad_shapes():
    with pytest.raises(ValueError):
        pos(np.zeros((100, 4)), FS)
    with pytest.raises(ValueError):
        pos(np.ones((10, 3)), FS)


def test_resample_uniform_handles_jittery_timestamps():
    rng = np.random.default_rng(3)
    t = np.cumsum(1 / FS + rng.uniform(-0.01, 0.01, 900))
    values = np.column_stack([np.sin(2 * np.pi * 1.0 * t)] * 3)
    grid, out = resample_uniform(t, values, FS)
    assert np.allclose(np.diff(grid), 1 / FS)
    assert np.allclose(out[:, 0], np.sin(2 * np.pi * 1.0 * grid), atol=0.01)
