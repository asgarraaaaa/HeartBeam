import cv2
import numpy as np
import pytest

from heartbeam.cli import main, measure, summarise
from heartbeam.roi import Box, mean_skin_rgb, skin_regions
from heartbeam.synth import write_video


def test_end_to_end_on_a_synthetic_video(tmp_path):
    path = tmp_path / "clip.avi"
    x, y, w, h = write_video(path, bpm=84, seconds=15)
    capture = cv2.VideoCapture(str(path))
    try:
        readings = measure(capture, is_file=True, roi=Box(x, y, w, h))
    finally:
        capture.release()
    assert summarise(readings) == pytest.approx(84, abs=2)


def test_cli_prints_heart_rate_and_writes_csv(tmp_path, capsys):
    path = tmp_path / "clip.avi"
    x, y, w, h = write_video(path, bpm=62, seconds=12)
    out_csv = tmp_path / "readings.csv"
    code = main(["--video", str(path), "--roi", f"{x},{y},{w},{h}", "--headless", "--csv", str(out_csv)])
    assert code == 0
    bpm = float(capsys.readouterr().out.split()[0])
    assert bpm == pytest.approx(62, abs=2)
    assert out_csv.read_text().startswith("t_seconds,bpm,snr_db,confidence")


def test_cli_reports_missing_file(tmp_path, capsys):
    assert main(["--video", str(tmp_path / "nope.mp4"), "--headless"]) == 2
    assert "no such file" in capsys.readouterr().err


def test_box_parse_validates_input():
    assert Box.parse("1,2,3,4") == Box(1, 2, 3, 4)
    with pytest.raises(ValueError):
        Box.parse("1,2,0,4")


def test_skin_regions_stay_inside_the_face():
    face = Box(100, 50, 200, 240)
    for r in skin_regions(face):
        assert face.x <= r.x and r.x + r.w <= face.x + face.w
        assert face.y <= r.y and r.y + r.h <= face.y + face.h


def test_mean_skin_rgb_ignores_non_skin_pixels():
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[:, :] = (115, 140, 200)  # skin tone, BGR
    frame[:20, :] = (255, 0, 0)  # a strip of saturated blue, not skin
    rgb = mean_skin_rgb(frame, [Box(0, 0, 100, 100)])
    assert rgb == pytest.approx([200, 140, 115], abs=1)


def test_mean_skin_rgb_clips_regions_to_the_frame():
    frame = np.full((50, 50, 3), 128, np.uint8)
    assert mean_skin_rgb(frame, [Box(40, 40, 30, 30)]) is not None
    assert mean_skin_rgb(frame, [Box(60, 60, 10, 10)]) is None


@pytest.mark.parametrize("state", ["no_face", "warming_up", "reading"])
def test_overlay_draws_every_state(state):
    from heartbeam.overlay import draw
    from heartbeam.rppg import Estimate

    frame = np.zeros((240, 320, 3), np.uint8)
    face = None if state == "no_face" else Box(80, 40, 160, 160)
    regions = [] if face is None else skin_regions(face)
    estimate = Estimate(72.0, 4.0) if state == "reading" else None
    wave = np.sin(np.linspace(0, 20, 300)) if state == "reading" else None
    out = draw(frame, face, regions, estimate, wave, warming_up=40)
    assert out.shape == frame.shape
    assert out.any()
    assert not frame.any(), "the input frame must not be modified"
