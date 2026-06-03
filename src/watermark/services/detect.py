"""Service: detect the watermark by extracting & cropping the first frame.

Pipeline:
  omni.mp4 --[ffmpeg]--> first_frame.png --[PIL crop]--> watermark_crop.png
"""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

from watermark.consts import VIDEO_SIZE

# Sanity-check threshold: a healthy watermark region should have noticeable
# color variance (the ✦ itself is dark RGB ~10-15 against a slightly
# different background). Use variance as a bbox-correctness proxy that works
# for both light and dark watermarks.
MIN_STD_DEV_IN_BBOX = 2.0  # grayscale std-dev lower bound (0..255)


class FFmpegNotFoundError(RuntimeError):
    """Raised when the ffmpeg binary is not on PATH."""


class FrameExtractionError(RuntimeError):
    """Raised when ffmpeg fails to extract the first frame."""


class WatermarkNotFoundError(RuntimeError):
    """Raised when the bbox crop is suspiciously empty (< MIN_BRIGHT_PIXELS)."""


def resolve_ffmpeg() -> str:
    """Return absolute path to ffmpeg, or raise FFmpegNotFoundError."""
    path = shutil.which("ffmpeg")
    if path is None:
        msg = "ffmpeg binary not found on PATH; install via `brew install ffmpeg`"
        raise FFmpegNotFoundError(msg)
    return path


def extract_first_frame(video: Path, out: Path, *, ffmpeg: str | None = None) -> Path:
    """Use ffmpeg to write the very first frame of *video* to *out* (PNG)."""
    binary = ffmpeg or resolve_ffmpeg()
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [
        binary,
        "-y",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(out),
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)  # noqa: S603
    if completed.returncode != 0:
        msg = f"ffmpeg frame extraction failed (rc={completed.returncode}): {completed.stderr}"
        raise FrameExtractionError(msg)
    return out


def open_frame(path: Path) -> Image.Image:
    """Open *path* as a PIL Image, asserting it matches VIDEO_SIZE."""
    img = Image.open(path)
    img.load()
    if img.size != VIDEO_SIZE:
        msg = f"frame size {img.size} != expected {VIDEO_SIZE}"
        raise ValueError(msg)
    return img


def _grayscale_stddev(image: Image.Image) -> float:
    """Return the grayscale standard deviation of *image* (0..255 scale)."""
    grayscale = image.convert("L")
    pixels: list[int] = list(grayscale.getdata())
    n = len(pixels)
    if n == 0:
        return 0.0
    mean: float = sum(pixels) / n
    variance: float = sum((p - mean) ** 2 for p in pixels) / n
    return math.sqrt(variance)


def crop_watermark(
    frame: Image.Image,
    bbox: tuple[int, int, int, int],
    out: Path | None = None,
) -> Image.Image:
    """Crop *frame* to *bbox* (left, top, right, bottom) and optionally save.

    Raises WatermarkNotFoundError if the cropped region is too uniform
    (low grayscale std-dev), which is a strong signal that the bbox missed
    the watermark.
    """
    left, top, right, bottom = bbox
    if (right - left) <= 0 or (bottom - top) <= 0:
        msg = f"invalid bbox dimensions: {bbox}"
        raise ValueError(msg)
    crop = frame.crop(bbox)
    stddev = _grayscale_stddev(crop)
    if stddev < MIN_STD_DEV_IN_BBOX:
        msg = (
            f"bbox {bbox} produced too-uniform crop (stddev={stddev:.2f} "
            f"< {MIN_STD_DEV_IN_BBOX}); watermark not found, "
            "adjust WATERMARK_BBOX in src/watermark/consts.py"
        )
        raise WatermarkNotFoundError(msg)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out)
    return crop


def detect(video: Path, frames_dir: Path, bbox: tuple[int, int, int, int]) -> Path:
    """Run the full detect step: extract frame, crop, save diagnostic images.

    Returns the path to the saved watermark_crop.png.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    first_frame_path = frames_dir / "frame_0000.png"
    crop_path = frames_dir / "watermark_crop.png"

    extract_first_frame(video, first_frame_path)
    frame = open_frame(first_frame_path)
    crop_watermark(frame, bbox, out=crop_path)
    return crop_path


def _bbox_argv_choices() -> Iterable[str]:
    """Placeholder for CLI --help epilog."""
    return ()
