"""Tests for the reassemble service (ffmpeg extraction & muxing)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from watermark.services import reassemble as reassemble_svc


def test_resolve_ffmpeg_raises_when_missing() -> None:
    """resolve_ffmpeg raises RuntimeError when ffmpeg is not on PATH."""
    with (
        patch("watermark.services.reassemble.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="ffmpeg"),
    ):
        reassemble_svc.resolve_ffmpeg()


def test_extract_frames_calls_ffmpeg_with_correct_args(tmp_path: Path) -> None:
    """extract_frames runs ffmpeg with -vsync 0 and a %04d.png output pattern."""
    in_dir = tmp_path / "frames"
    in_dir.mkdir()
    video = tmp_path / "in.mp4"

    with patch("watermark.services.reassemble.subprocess.run") as mock_run:
        # Simulate ffmpeg producing some files
        (in_dir / "0000.png").write_bytes(b"")
        (in_dir / "0001.png").write_bytes(b"")
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""

        result = reassemble_svc.extract_frames(video, in_dir, ffmpeg="/usr/bin/ffmpeg")

    args = mock_run.call_args[0][0]
    assert args[0] == "/usr/bin/ffmpeg"
    assert "-vsync" in args
    assert "0" in args
    assert "-start_number" in args
    assert "%04d.png" in " ".join(args)
    assert len(result) == 2


def test_extract_frames_propagates_failure(tmp_path: Path) -> None:
    """Non-zero returncode raises RuntimeError with stderr context."""
    with patch("watermark.services.reassemble.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "extraction failed"
        with pytest.raises(RuntimeError, match="extraction failed"):
            reassemble_svc.extract_frames(tmp_path / "in.mp4", tmp_path / "out", ffmpeg="/usr/bin/ffmpeg")


def test_reassemble_uses_crf_and_audio_copy(tmp_path: Path) -> None:
    """reassemble invokes ffmpeg with -c:a copy and the expected crf/preset/pix_fmt."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    source = tmp_path / "source.mp4"
    out = tmp_path / "out.mp4"

    with patch("watermark.services.reassemble.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        out.write_bytes(b"")  # simulate ffmpeg writing the file
        reassemble_svc.reassemble(frames_dir, source, out, ffmpeg="/usr/bin/ffmpeg")

    args = mock_run.call_args[0][0]
    assert "-framerate" in args
    assert "24" in args
    assert "-c:a" in args
    assert "copy" in args
    assert "-crf" in args
    assert "16" in args
    assert "-preset" in args
    assert "medium" in args
    assert "-pix_fmt" in args
    assert "yuv420p" in args
    # -map 0:v -map 1:a to combine video from frames with audio from source
    assert "0:v" in args
    assert "1:a" in args


def test_reassemble_propagates_failure(tmp_path: Path) -> None:
    """Non-zero returncode raises RuntimeError."""
    with patch("watermark.services.reassemble.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "reassemble failed"
        with pytest.raises(RuntimeError, match="reassemble failed"):
            reassemble_svc.reassemble(tmp_path / "f", tmp_path / "s", tmp_path / "o", ffmpeg="/usr/bin/ffmpeg")
