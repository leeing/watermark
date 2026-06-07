"""Service: ffmpeg-based frame extraction & video reassembly."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import structlog

from watermark.consts import CRF, FPS, PIX_FMT, PRESET

log = structlog.get_logger()


def resolve_ffmpeg() -> str:
    """Return absolute path to ffmpeg or raise."""
    path = shutil.which("ffmpeg")
    if path is None:
        msg = "ffmpeg binary not found on PATH; install via `brew install ffmpeg`"
        raise RuntimeError(msg)
    return path


def extract_frames(
    video: Path,
    out_dir: Path,
    *,
    fps: int = FPS,
    ffmpeg: str | None = None,
) -> list[Path]:
    """Extract every video frame to *out_dir* as 0000.png, 0001.png, ...

    Uses `-vsync 0 -start_number 0` so we always get a clean, sequential
    sequence (no dropped/duplicated frames).
    """
    binary = ffmpeg or resolve_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "%04d.png"
    cmd: list[str] = [
        binary,
        "-y",
        "-i",
        str(video),
        "-vsync",
        "0",
        "-start_number",
        "0",
        "-frame_pts",
        "1",
        str(pattern),
    ]
    log.info("extracting_frames", src=str(video), out=str(out_dir), fps=fps)
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)  # noqa: S603
    if completed.returncode != 0:
        msg = f"ffmpeg extract failed (rc={completed.returncode}): {completed.stderr[-500:]}"
        raise RuntimeError(msg)
    return sorted(out_dir.glob("*.png"))


def reassemble(  # noqa: PLR0913
    frames_dir: Path,
    source_video: Path,
    out_video: Path,
    *,
    fps: int = FPS,
    crf: int | None = None,
    preset: str = PRESET,
    pix_fmt: str = PIX_FMT,
    ffmpeg: str | None = None,
) -> Path:
    """Mux *frames_dir* + the audio track from *source_video* into *out_video*.

    Audio is passed through with `-c:a copy` — byte-identical, no re-encode.
    """
    binary = ffmpeg or resolve_ffmpeg()
    resolved_crf = crf if crf is not None else CRF
    out_video.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "%04d.png")
    cmd: list[str] = [
        binary,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-i",
        str(source_video),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-crf",
        str(resolved_crf),
        "-preset",
        preset,
        "-pix_fmt",
        pix_fmt,
        "-c:a",
        "copy",
        str(out_video),
    ]
    log.info("reassembling", frames=str(frames_dir), out=str(out_video), crf=resolved_crf, preset=preset)
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)  # noqa: S603
    if completed.returncode != 0:
        msg = f"ffmpeg reassemble failed (rc={completed.returncode}): {completed.stderr[-500:]}"
        raise RuntimeError(msg)
    return out_video
