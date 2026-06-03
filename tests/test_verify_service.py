"""Tests for the verify service."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from watermark.consts import VIDEO_SIZE
from watermark.services import verify as verify_svc


def _make_probe_payload(  # noqa: PLR0913
    *,
    width: int = VIDEO_SIZE[0],
    height: int = VIDEO_SIZE[1],
    nb_frames: int = 240,
    duration: float = 10.0,
    codec: str = "h264",
    has_audio: bool = True,
) -> str:
    streams: list[dict[str, object]] = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": codec,
            "width": width,
            "height": height,
            "nb_frames": nb_frames,
        }
    ]
    if has_audio:
        streams.append({"index": 1, "codec_type": "audio", "codec_name": "aac"})
    return json.dumps({"streams": streams, "format": {"duration": str(duration)}})


def test_probe_parses_ffprobe_json(tmp_path: Path) -> None:
    """probe() returns a ProbeResult parsed from ffprobe JSON output."""
    payload = _make_probe_payload()
    with patch("watermark.services.verify.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = payload
        result = verify_svc.probe(tmp_path / "video.mp4", ffprobe="/usr/bin/ffprobe")
    assert result.width == 1080
    assert result.height == 1920
    assert result.nb_frames == 240
    assert result.duration == 10.0
    assert result.codec_name == "h264"
    assert result.has_audio is True


def test_probe_raises_on_no_video_stream(tmp_path: Path) -> None:
    """probe() raises VerificationError when there's no video stream."""
    payload = json.dumps({"streams": [{"codec_type": "audio"}], "format": {"duration": "1"}})
    with patch("watermark.services.verify.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = payload
        with pytest.raises(verify_svc.VerificationError, match="no video stream"):
            verify_svc.probe(tmp_path / "video.mp4", ffprobe="/usr/bin/ffprobe")


def test_audio_md5_parses_hash(tmp_path: Path) -> None:
    """audio_md5 extracts the 32-char hex hash from ffmpeg's MD5 output."""
    fake_stdout = b"MD5=abcdef0123456789abcdef0123456789\n"
    with patch("watermark.services.verify.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_stdout
        result = verify_svc.audio_md5(tmp_path / "video.mp4", ffmpeg="/usr/bin/ffmpeg")
    assert result == "abcdef0123456789abcdef0123456789"


def test_audio_md5_raises_when_no_hash(tmp_path: Path) -> None:
    """audio_md5 raises if the output doesn't match the expected format."""
    with patch("watermark.services.verify.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = b"no hash here"
        with pytest.raises(RuntimeError, match="could not parse MD5"):
            verify_svc.audio_md5(tmp_path / "video.mp4", ffmpeg="/usr/bin/ffmpeg")


def test_verify_output_happy_path(tmp_path: Path) -> None:
    """verify_output passes when probe result matches all SPEC criteria."""
    payload = _make_probe_payload()
    with patch("watermark.services.verify.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = payload
        result = verify_svc.verify_output(tmp_path / "video.mp4")
    assert result.codec_name == "h264"


def test_verify_output_rejects_wrong_size(tmp_path: Path) -> None:
    """verify_output raises when dimensions don't match VIDEO_SIZE."""
    payload = _make_probe_payload(width=720, height=1280)
    with patch("watermark.services.verify.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = payload
        with pytest.raises(verify_svc.VerificationError, match="dimensions"):
            verify_svc.verify_output(tmp_path / "video.mp4")


def test_verify_output_rejects_wrong_duration(tmp_path: Path) -> None:
    """verify_output raises when duration is off by more than the tolerance."""
    payload = _make_probe_payload(duration=12.0)
    with patch("watermark.services.verify.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = payload
        with pytest.raises(verify_svc.VerificationError, match="duration"):
            verify_svc.verify_output(tmp_path / "video.mp4")


def test_verify_output_rejects_wrong_codec(tmp_path: Path) -> None:
    """verify_output raises when codec is not h264."""
    payload = _make_probe_payload(codec="hevc")
    with patch("watermark.services.verify.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = payload
        with pytest.raises(verify_svc.VerificationError, match="codec_name"):
            verify_svc.verify_output(tmp_path / "video.mp4")


def test_verify_output_rejects_missing_audio(tmp_path: Path) -> None:
    """verify_output raises when no audio stream is present."""
    payload = _make_probe_payload(has_audio=False)
    with patch("watermark.services.verify.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = payload
        with pytest.raises(verify_svc.VerificationError, match="audio"):
            verify_svc.verify_output(tmp_path / "video.mp4")


def test_verify_audio_identical_passes_on_match(tmp_path: Path) -> None:
    """verify_audio_identical is a no-op when the two MD5s match."""
    fake_stdout = b"MD5=deadbeefdeadbeefdeadbeefdeadbeef\n"
    with patch("watermark.services.verify.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_stdout
        verify_svc.verify_audio_identical(tmp_path / "a.mp4", tmp_path / "b.mp4")


def test_verify_audio_identical_raises_on_mismatch(tmp_path: Path) -> None:
    """verify_audio_identical raises if the two audio MD5s differ."""
    with patch("watermark.services.verify.subprocess.run") as mock_run:
        # First call returns one valid 32-char hash, second returns another
        mock_run.side_effect = [
            type("R", (), {"returncode": 0, "stdout": b"MD5=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"})(),
            type("R", (), {"returncode": 0, "stdout": b"MD5=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"})(),
        ]
        with pytest.raises(verify_svc.VerificationError, match="MD5 mismatch"):
            verify_svc.verify_audio_identical(tmp_path / "a.mp4", tmp_path / "b.mp4")
