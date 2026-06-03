"""Tests for the Click `detect` command."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner
from PIL import Image

from watermark.commands.detect import detect_cmd
from watermark.config import Settings
from watermark.consts import VIDEO_SIZE, WATERMARK_BBOX


def test_detect_cmd_missing_file(tmp_path: Path) -> None:
    """`watermark detect` exits 2 if the video file does not exist (Click argument check)."""
    runner = CliRunner()
    result = runner.invoke(detect_cmd, [str(tmp_path / "missing.mp4")])
    assert result.exit_code == 2  # Click argument error


def test_detect_cmd_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`watermark detect` prints the crop path on success."""
    fake_settings = Settings(
        project_root=tmp_path,
        frames_dir=tmp_path / "frames",
    )
    (tmp_path / "frames").mkdir()

    # Build a frame with a bright watermark inside WATERMARK_BBOX
    frame = Image.new("RGB", VIDEO_SIZE, (10, 10, 10))
    left, top, right, bottom = WATERMARK_BBOX
    for x in range(left, right):
        for y in range(top, bottom):
            frame.putpixel((x, y), (255, 255, 255))
    frame.save(fake_settings.frames_dir / "frame_0000.png")

    from watermark.commands import detect as detect_module

    monkeypatch.setattr(detect_module, "settings", fake_settings)

    def fake_detect(video: Path, frames_dir: Path, bbox: tuple[int, int, int, int]) -> Path:
        return frames_dir / "watermark_crop.png"

    monkeypatch.setattr(detect_module.detect_svc, "detect", fake_detect)

    video = tmp_path / "in.mp4"
    video.write_bytes(b"")

    runner = CliRunner()
    result = runner.invoke(detect_cmd, [str(video)])
    assert result.exit_code == 0, result.output
    assert "watermark_crop.png" in result.output
