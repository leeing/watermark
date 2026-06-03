"""Tests for the Click `mask` command."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner
from PIL import Image

from watermark.commands.mask import mask_cmd
from watermark.config import Settings
from watermark.consts import VIDEO_SIZE


def test_mask_cmd_missing_first_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`watermark mask` exits 1 if no first frame exists (user must run detect first)."""
    fake_settings = Settings(project_root=tmp_path, frames_dir=tmp_path / "frames")
    (tmp_path / "frames").mkdir()

    from watermark.commands import mask as mask_module

    monkeypatch.setattr(mask_module, "settings", fake_settings)

    runner = CliRunner()
    result = runner.invoke(mask_cmd, [])
    assert result.exit_code == 1
    assert "frame_0000.png" in result.output or "detect" in result.output


def test_mask_cmd_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`watermark mask` prints both mask and overlay paths on success."""
    fake_settings = Settings(
        project_root=tmp_path,
        frames_dir=tmp_path / "frames",
        mask_path=tmp_path / "mask.png",
    )
    (tmp_path / "frames").mkdir()
    Image.new("RGB", VIDEO_SIZE, (0, 0, 0)).save(fake_settings.frames_dir / "frame_0000.png")

    from watermark.commands import mask as mask_module

    monkeypatch.setattr(mask_module, "settings", fake_settings)

    mask_p = fake_settings.mask_path
    overlay_p = fake_settings.frames_dir / "mask_overlay.png"

    def fake_build(*args: object, **kwargs: object) -> tuple[Path, Path]:
        mask_p.write_bytes(b"")
        overlay_p.write_bytes(b"")
        return mask_p, overlay_p

    monkeypatch.setattr(mask_module.mask_svc, "build", fake_build)

    runner = CliRunner()
    result = runner.invoke(mask_cmd, [])
    assert result.exit_code == 0, result.output
    assert "mask:" in result.output
    assert "overlay:" in result.output


def test_mask_cmd_passes_mode_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`watermark mask --mode rectangle` forwards the selected mask mode."""
    fake_settings = Settings(
        project_root=tmp_path,
        frames_dir=tmp_path / "frames",
        mask_path=tmp_path / "mask.png",
    )
    fake_settings.frames_dir.mkdir()
    Image.new("RGB", VIDEO_SIZE, (0, 0, 0)).save(fake_settings.frames_dir / "frame_0000.png")

    from watermark.commands import mask as mask_module

    monkeypatch.setattr(mask_module, "settings", fake_settings)
    captured: list[dict[str, object]] = []

    def fake_build(*args: object, **kwargs: object) -> tuple[Path, Path]:
        captured.append(kwargs)
        return fake_settings.mask_path, fake_settings.frames_dir / "mask_overlay.png"

    monkeypatch.setattr(mask_module.mask_svc, "build", fake_build)

    runner = CliRunner()
    result = runner.invoke(mask_cmd, ["--mode", "rectangle"])
    assert result.exit_code == 0, result.output
    assert captured[-1]["mode"] == "rectangle"
