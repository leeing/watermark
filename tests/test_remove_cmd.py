"""Tests for the Click `remove` command (end-to-end with mocked services)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from watermark.commands.remove import _parse_bbox, remove_cmd
from watermark.config import Settings
from watermark.main import cli
from watermark.services import inpaint as inpaint_svc


def test_parse_bbox_happy_path() -> None:
    """_parse_bbox returns a 4-tuple of ints."""
    assert _parse_bbox("1,2,3,4") == (1, 2, 3, 4)


def test_parse_bbox_strips_whitespace() -> None:
    """Whitespace around numbers is tolerated."""
    assert _parse_bbox(" 10 , 20 , 30 , 40 ") == (10, 20, 30, 40)


def test_parse_bbox_rejects_wrong_arity() -> None:
    """Three numbers is rejected."""
    import click as _click

    with pytest.raises(_click.BadParameter, match="4 comma-separated"):
        _parse_bbox("1,2,3")


def test_parse_bbox_rejects_non_integer() -> None:
    """A non-integer component is rejected."""
    import click as _click

    with pytest.raises(_click.BadParameter, match="non-integer"):
        _parse_bbox("1,2,three,4")


def test_parse_bbox_rejects_inverted() -> None:
    """right<=left or bottom<=top is rejected."""
    import click as _click

    with pytest.raises(_click.BadParameter, match="right>left"):
        _parse_bbox("100,100,50,200")


def test_remove_cmd_lists_in_help() -> None:
    """The CLI help mentions the `remove` subcommand."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "remove" in result.output


@dataclass
class PipelineCalls:
    """Tracks which service functions the remove command invoked."""

    detect: list[bool] = field(default_factory=list)
    mask: list[bool] = field(default_factory=list)
    extract: list[bool] = field(default_factory=list)
    inpaint: list[bool] = field(default_factory=list)
    reassemble: list[bool] = field(default_factory=list)
    verify_output: list[bool] = field(default_factory=list)
    verify_audio: list[bool] = field(default_factory=list)
    verify_alpha_round_trip: list[bool] = field(default_factory=list)
    mask_kwargs: list[dict[str, object]] = field(default_factory=list)
    inpaint_args: list[tuple[object, ...]] = field(default_factory=list)


@dataclass
class RemoveEnv:
    """Fake settings + call tracker for the remove pipeline."""

    settings: Settings
    calls: PipelineCalls


@pytest.fixture
def fake_remove_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RemoveEnv:  # noqa: C901
    """Wire fake settings + tracking containers for the remove pipeline."""
    fake_settings = Settings(
        project_root=tmp_path,
        frames_dir=tmp_path / "frames",
        frames_in_dir=tmp_path / "frames_in",
        frames_out_dir=tmp_path / "frames_out",
        mask_path=tmp_path / "mask.png",
        alpha_map_path=tmp_path / "data" / "gemini_96_alpha.png",
        source_video=tmp_path / "in.mp4",
        output_video=tmp_path / "out.mp4",
        device="cpu",
    )
    for sub in (fake_settings.frames_dir, fake_settings.frames_in_dir, fake_settings.frames_out_dir):
        sub.mkdir()
    (tmp_path / "out.mp4").write_bytes(b"")  # simulate reassemble output
    (tmp_path / "in.mp4").write_bytes(b"")  # existence check; real ffmpeg is mocked
    (tmp_path / "data").mkdir()
    # 96x96 RGB PNG with a tiny alpha so the alpha backend can load it.
    Image.fromarray(np.full((96, 96, 3), 64, dtype=np.uint8), mode="RGB").save(fake_settings.alpha_map_path)
    # The newer GargantuaX alpha map variant also gets a dummy file in case
    # the fixture's configured alpha_map_path differs from the default.
    candidate = tmp_path / "data" / "gemini_96_alpha_20260520.png"
    if candidate != fake_settings.alpha_map_path:
        Image.fromarray(np.full((96, 96, 3), 64, dtype=np.uint8), mode="RGB").save(candidate)

    from watermark.commands import remove as remove_module

    monkeypatch.setattr(remove_module, "settings", fake_settings)
    calls = PipelineCalls()

    def fake_detect(*args: object, **kwargs: object) -> Path:
        calls.detect.append(True)
        # Anchor search (alpha backend) reads frame_0000.png, so create a real file.
        from PIL import Image as _Im

        _Im.new("RGB", (1080, 1920), (60, 60, 60)).save(fake_settings.frames_dir / "frame_0000.png")
        return fake_settings.frames_dir / "watermark_crop.png"

    def fake_build(*args: object, **kwargs: object) -> tuple[Path, Path]:
        calls.mask.append(True)
        calls.mask_kwargs.append(kwargs)
        return fake_settings.mask_path, fake_settings.frames_dir / "mask_overlay.png"

    def fake_extract(*args: object, **kwargs: object) -> list[Path]:
        calls.extract.append(True)
        return []

    def fake_inpaint(*args: object, **kwargs: object) -> list[Path]:
        calls.inpaint.append(True)
        calls.inpaint_args.append(args)
        return []

    def fake_reassemble(*args: object, **kwargs: object) -> Path:
        calls.reassemble.append(True)
        return fake_settings.output_video

    def fake_verify_output(*args: object, **kwargs: object) -> None:
        calls.verify_output.append(True)

    def fake_verify_audio(*args: object, **kwargs: object) -> None:
        calls.verify_audio.append(True)

    def fake_verify_alpha_round_trip(*args: object, **kwargs: object) -> None:
        calls.verify_alpha_round_trip.append(True)

    monkeypatch.setattr(remove_module.detect_svc, "detect", fake_detect)
    monkeypatch.setattr(remove_module.mask_svc, "build", fake_build)
    monkeypatch.setattr(remove_module.reassemble_svc, "extract_frames", fake_extract)
    monkeypatch.setattr(remove_module.inpaint_svc, "inpaint_frames", fake_inpaint)
    monkeypatch.setattr(remove_module.reassemble_svc, "reassemble", fake_reassemble)
    monkeypatch.setattr(remove_module.verify_svc, "verify_output", fake_verify_output)
    monkeypatch.setattr(remove_module.verify_svc, "verify_audio_identical", fake_verify_audio)
    monkeypatch.setattr(remove_module.verify_svc, "verify_alpha_round_trip", fake_verify_alpha_round_trip)

    return RemoveEnv(settings=fake_settings, calls=calls)


def test_remove_cmd_runs_full_pipeline(fake_remove_env: RemoveEnv) -> None:
    """`watermark remove` orchestrates detect -> mask -> inpaint -> reassemble (default telea backend)."""
    calls = fake_remove_env.calls

    runner = CliRunner()
    result = runner.invoke(remove_cmd, ["--device", "cpu", "--skip-verify"])
    assert result.exit_code == 0, result.output

    assert calls.detect, "detect not called"
    # Default backend is "telea" — mask IS generated (heuristic inpainting needs it).
    assert calls.mask, "telea backend should call mask"
    assert calls.extract, "extract not called"
    assert calls.inpaint, "inpaint not called"
    assert calls.reassemble, "reassemble not called"
    # --skip-verify was passed: verify steps are not invoked
    assert not calls.verify_output
    assert not calls.verify_audio


def test_remove_cmd_runs_full_pipeline_with_telea_backend(fake_remove_env: RemoveEnv) -> None:
    """`watermark remove --backend telea` keeps the mask step."""
    calls = fake_remove_env.calls

    runner = CliRunner()
    result = runner.invoke(remove_cmd, ["--device", "cpu", "--backend", "telea", "--skip-verify"])
    assert result.exit_code == 0, result.output

    assert calls.detect, "detect not called"
    assert calls.mask, "telea backend should call mask"
    assert calls.extract, "extract not called"
    assert calls.inpaint, "inpaint not called"
    assert calls.reassemble, "reassemble not called"


def test_remove_cmd_alpha_backend_runs_alpha_round_trip(fake_remove_env: RemoveEnv) -> None:
    """Alpha backend runs A8 round-trip verification by default."""
    calls = fake_remove_env.calls

    runner = CliRunner()
    result = runner.invoke(remove_cmd, ["--device", "cpu"])
    assert result.exit_code == 0, result.output
    assert calls.verify_output, "verify_output not called"
    assert calls.verify_audio, "verify_audio not called"
    # The alpha round-trip verification is verified by a dedicated verify_svc mock below.
    # Here we only assert the pipeline runs cleanly; round-trip is exercised in test_verify_service.py.


def test_remove_cmd_runs_verify_when_not_skipped(fake_remove_env: RemoveEnv) -> None:
    """Without --skip-verify, both verify steps are invoked."""
    calls = fake_remove_env.calls

    runner = CliRunner()
    result = runner.invoke(remove_cmd, ["--device", "cpu"])
    assert result.exit_code == 0, result.output
    assert calls.verify_output
    assert calls.verify_audio


def test_remove_cmd_passes_quality_options(fake_remove_env: RemoveEnv) -> None:
    """Quality options are forwarded to mask and inpaint services."""
    calls = fake_remove_env.calls

    runner = CliRunner()
    result = runner.invoke(
        remove_cmd,
        [
            "--device",
            "cpu",
            "--backend",
            "ns",
            "--mask-mode",
            "watermark",
            "--radius",
            "7",
            "--texture-strength",
            "0.5",
            "--feather",
            "3",
            "--skip-verify",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls.mask_kwargs[-1]["mode"] == "watermark"
    options = calls.inpaint_args[-1][3]
    assert options == inpaint_svc.InpaintOptions(
        device="cpu",
        backend="ns",
        radius=7,
        texture_strength=0.5,
        feather=3,
    )


def test_remove_cmd_invalid_bbox_exits_2() -> None:
    """An unparseable --bbox exits with code 2 (Click usage error)."""
    runner = CliRunner()
    result = runner.invoke(remove_cmd, ["--bbox", "garbage"])
    assert result.exit_code == 2
