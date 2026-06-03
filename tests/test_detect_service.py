"""Tests for the detect service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from watermark.consts import VIDEO_SIZE, WATERMARK_BBOX
from watermark.services import detect as detect_svc


def _make_blank_frame(out: Path) -> Path:
    """Save a 1080x1920 black image as the diagnostic frame."""
    Image.new("RGB", VIDEO_SIZE, (10, 10, 10)).save(out)
    return out


def _make_frame_with_bright_box(out: Path, bbox: tuple[int, int, int, int]) -> Path:
    """Save a frame that has bright pixels in the bbox (mimics a light watermark)."""
    frame = Image.new("RGB", VIDEO_SIZE, (10, 10, 10))
    left, top, right, bottom = bbox
    for x in range(left, right):
        for y in range(top, bottom):
            frame.putpixel((x, y), (255, 255, 255))
    frame.save(out)
    return out


def _make_frame_with_dark_watermark(out: Path, bbox: tuple[int, int, int, int]) -> Path:
    """Save a frame with a small dark watermark (RGB ~10) on a slightly different background."""
    frame = Image.new("RGB", VIDEO_SIZE, (40, 35, 38))
    left, top, right, bottom = bbox
    # Paint only a small ✦-like shape in the center of the bbox
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            frame.putpixel((cx + dx, cy + dy), (10, 8, 12))
    frame.save(out)
    return out


def test_extract_first_frame_invokes_ffmpeg(tmp_path: Path) -> None:
    """extract_first_frame calls ffmpeg with the correct arguments."""
    fake_ffmpeg = "/usr/bin/ffmpeg"
    video = tmp_path / "in.mp4"
    out = tmp_path / "frame.png"
    with patch("watermark.services.detect.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        # create the expected output file so downstream open works
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", VIDEO_SIZE, (0, 0, 0)).save(out)
        detect_svc.extract_first_frame(video, out, ffmpeg=fake_ffmpeg)
    args = mock_run.call_args[0][0]
    assert args[0] == fake_ffmpeg
    assert "-frames:v" in args
    assert "1" in args
    assert str(out) in args


def test_extract_first_frame_propagates_failure(tmp_path: Path) -> None:
    """Non-zero ffmpeg returncode raises FrameExtractionError."""
    with patch("watermark.services.detect.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "boom"
        with pytest.raises(detect_svc.FrameExtractionError, match="boom"):
            detect_svc.extract_first_frame(tmp_path / "in.mp4", tmp_path / "out.png", ffmpeg="/usr/bin/ffmpeg")


def test_resolve_ffmpeg_raises_when_missing() -> None:
    """resolve_ffmpeg raises FFmpegNotFoundError when no binary is on PATH."""
    with (
        patch("watermark.services.detect.shutil.which", return_value=None),
        pytest.raises(detect_svc.FFmpegNotFoundError),
    ):
        detect_svc.resolve_ffmpeg()


def test_open_frame_rejects_wrong_size(tmp_path: Path) -> None:
    """open_frame raises if the PNG is not VIDEO_SIZE."""
    bad = tmp_path / "bad.png"
    Image.new("RGB", (100, 100), (0, 0, 0)).save(bad)
    with pytest.raises(ValueError, match="size"):
        detect_svc.open_frame(bad)


def test_crop_watermark_raises_when_bbox_too_uniform(tmp_path: Path) -> None:
    """An all-same-color crop raises WatermarkNotFoundError."""
    frame = Image.new("RGB", VIDEO_SIZE, (0, 0, 0))
    with pytest.raises(detect_svc.WatermarkNotFoundError, match="watermark not found"):
        detect_svc.crop_watermark(frame, WATERMARK_BBOX, out=tmp_path / "crop.png")


def test_crop_watermark_saves_when_bbox_has_bright_pixels(tmp_path: Path) -> None:
    """If the bbox contains a small bright mark on a darker background, the crop is saved."""
    frame = Image.new("RGB", VIDEO_SIZE, (10, 10, 10))
    left, top, right, bottom = WATERMARK_BBOX
    # Paint only a small ✦-like shape in the center of the bbox
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            frame.putpixel((cx + dx, cy + dy), (240, 240, 240))
    out = tmp_path / "crop.png"
    crop = detect_svc.crop_watermark(frame, WATERMARK_BBOX, out=out)
    assert out.is_file()
    assert crop.size == (right - left, bottom - top)


def test_crop_watermark_saves_for_dark_watermark(tmp_path: Path) -> None:
    """A dark watermark (RGB~10) on a slightly different background is also detected."""
    out = tmp_path / "frame.png"
    _make_frame_with_dark_watermark(out, WATERMARK_BBOX)
    frame = Image.open(out)
    crop = detect_svc.crop_watermark(frame, WATERMARK_BBOX, out=tmp_path / "crop.png")
    assert (tmp_path / "crop.png").is_file()
    assert crop.size == (WATERMARK_BBOX[2] - WATERMARK_BBOX[0], WATERMARK_BBOX[3] - WATERMARK_BBOX[1])


def test_crop_watermark_rejects_inverted_bbox() -> None:
    """A malformed bbox raises ValueError immediately."""
    frame = Image.new("RGB", VIDEO_SIZE, (0, 0, 0))
    with pytest.raises(ValueError, match="invalid bbox"):
        detect_svc.crop_watermark(frame, (100, 100, 50, 50))
