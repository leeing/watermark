"""Service: verify the cleaned video meets all SPEC acceptance criteria."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import structlog

from watermark.consts import VIDEO_SIZE

log = structlog.get_logger()

_DURATION_TOLERANCE = 0.05  # seconds


class VerificationError(RuntimeError):
    """Raised when a video fails to meet one of the SPEC acceptance criteria."""


@dataclass(frozen=True)
class ProbeResult:
    """Subset of ffprobe JSON we care about."""

    width: int
    height: int
    nb_frames: int
    duration: float
    codec_name: str
    has_audio: bool


def resolve_ffprobe() -> str:
    """Return absolute path to ffprobe or raise."""
    path = shutil.which("ffprobe")
    if path is None:
        msg = "ffprobe binary not found on PATH"
        raise RuntimeError(msg)
    return path


def probe(video: Path, *, ffprobe: str | None = None) -> ProbeResult:
    """Run ffprobe and parse a minimal subset of stream/format metadata."""
    binary = ffprobe or resolve_ffprobe()
    cmd: list[str] = [
        binary,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,nb_frames",
        "-show_entries",
        "format=duration",
        str(video),
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)  # noqa: S603
    if completed.returncode != 0:
        msg = f"ffprobe failed (rc={completed.returncode}): {completed.stderr}"
        raise RuntimeError(msg)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    fmt = payload.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        msg = f"no video stream in {video}"
        raise VerificationError(msg)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    return ProbeResult(
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        nb_frames=int(video_stream.get("nb_frames", 0)),
        duration=float(fmt.get("duration", 0.0)),
        codec_name=str(video_stream.get("codec_name", "")),
        has_audio=audio_stream is not None,
    )


_MD5_LINE = re.compile(rb"MD5=(?P<hash>[0-9a-fA-F]{32})")


def audio_md5(video: Path, *, ffmpeg: str | None = None) -> str:
    """Compute MD5 of the audio stream of *video* (lossless passthrough)."""
    binary = ffmpeg or shutil.which("ffmpeg")
    if binary is None:
        msg = "ffmpeg binary not found on PATH"
        raise RuntimeError(msg)
    cmd: list[str] = [binary, "-v", "error", "-i", str(video), "-map", "0:a", "-c", "copy", "-f", "md5", "-"]
    completed = subprocess.run(cmd, check=False, capture_output=True)  # noqa: S603
    if completed.returncode != 0:
        msg = f"ffmpeg md5 failed (rc={completed.returncode}): {completed.stderr.decode(errors='ignore')}"
        raise RuntimeError(msg)
    match = _MD5_LINE.search(completed.stdout)
    if match is None:
        msg = f"could not parse MD5 from ffmpeg output: {completed.stdout!r}"
        raise RuntimeError(msg)
    return match.group("hash").decode().lower()


def verify_output(video: Path, *, expected_size: tuple[int, int] = VIDEO_SIZE) -> ProbeResult:
    """Run probe + raise if SPEC acceptance criteria A2-A5 fail."""
    result = probe(video)
    if (result.width, result.height) != expected_size:
        msg = f"{video}: dimensions {result.width}x{result.height} != expected {expected_size}"
        raise VerificationError(msg)
    if result.nb_frames <= 0:
        msg = f"{video}: nb_frames={result.nb_frames} (expected > 0)"
        raise VerificationError(msg)
    if abs(result.duration - 10.0) > _DURATION_TOLERANCE:
        msg = f"{video}: duration {result.duration:.3f}s differs from 10.0 by more than {_DURATION_TOLERANCE}s"
        raise VerificationError(msg)
    if result.codec_name != "h264":
        msg = f"{video}: codec_name={result.codec_name!r} != 'h264'"
        raise VerificationError(msg)
    if not result.has_audio:
        msg = f"{video}: missing audio stream"
        raise VerificationError(msg)
    log.info("verify_pass", **result.__dict__)
    return result


def verify_audio_identical(a: Path, b: Path) -> None:
    """SPEC A6: audio MD5 of cleaned video must match the source video."""
    md5_a = audio_md5(a)
    md5_b = audio_md5(b)
    if md5_a != md5_b:
        msg = f"audio MD5 mismatch: source={md5_a} output={md5_b}"
        raise VerificationError(msg)
    log.info("audio_md5_match", md5=md5_a)
