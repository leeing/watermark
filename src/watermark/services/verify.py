"""Service: verify the cleaned video meets all SPEC acceptance criteria."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
from PIL import Image

from watermark.consts import VIDEO_SIZE

log = structlog.get_logger()

_DURATION_TOLERANCE = 0.05  # seconds

# Default pixel-difference tolerance for the alpha round-trip check (A8).
# Accounts for the lossy H.264 re-encode (CRF 16) of omni_clean.mp4.
_DEFAULT_ALPHA_TOLERANCE = 2


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


def extract_frame(video: Path, time_seconds: float, *, ffmpeg: str | None = None) -> Image.Image:
    """Extract a single frame from *video* at *time_seconds* (float seconds).

    Returns a PIL Image in RGB mode. Used by `verify_alpha_round_trip`.
    """

    binary = ffmpeg or shutil.which("ffmpeg")
    if binary is None:
        msg = "ffmpeg binary not found on PATH"
        raise RuntimeError(msg)
    cmd: list[str] = [
        binary,
        "-y",
        "-ss",
        f"{time_seconds:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-update",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True)  # noqa: S603
    if completed.returncode != 0:
        msg = f"ffmpeg frame extract failed (rc={completed.returncode}): {completed.stderr.decode(errors='ignore')}"
        raise RuntimeError(msg)
    import io

    return Image.open(io.BytesIO(completed.stdout)).convert("RGB")


def verify_alpha_round_trip(  # noqa: PLR0913
    source_video: Path,
    clean_video: Path,
    alpha_map: np.ndarray,
    bbox: tuple[int, int, int, int],
    logo_rgb: tuple[int, int, int] = (255, 255, 255),
    *,
    frame_time: float = 5.0,
    pixel_tolerance: int = _DEFAULT_ALPHA_TOLERANCE,
) -> None:
    """SPEC A8: forward(clean) with α + logo must equal source frame at bbox region.

    Procedure:
        1. Extract one frame at *frame_time* seconds from `source_video` and
           `clean_video` (using ffmpeg to bypass decode/encode loss in the
           PNG→PNG comparison).
        2. Apply forward alpha at the bbox region only (outside the bbox the
           formula has no effect — α=0 there):
              `expected[top:bottom, left:right] = clean * (1 - alpha) + logo * alpha`
        3. Compare with the source frame at the bbox.
        4. Allow up to *pixel_tolerance* uint8 drift from H.264 re-encode.

    Raises:
        VerificationError: if any pixel differs by more than the tolerance.
    """
    import numpy as np

    src_frame = extract_frame(source_video, frame_time)
    clean_frame = extract_frame(clean_video, frame_time)
    src_np = np.asarray(src_frame, dtype=np.float32)
    clean_np = np.asarray(clean_frame, dtype=np.float32)
    left, top, right, bottom = bbox
    # alpha_map shape must match the bbox (sanity check, same as inpaint_alpha_frame)
    if alpha_map.shape != (bottom - top, right - left):
        msg = f"alpha_map shape {alpha_map.shape} doesn't match bbox size ({bottom - top}, {right - left})"
        raise VerificationError(msg)
    alpha = alpha_map[..., np.newaxis]  # (h, w, 1)
    logo = np.array(logo_rgb, dtype=np.float32)  # (3,)
    # Forward alpha: original * (1 - alpha) + logo * alpha
    # Only applied to the bbox region; outside the bbox, clean should match source
    expected_region = clean_np[top:bottom, left:right] * (1.0 - alpha) + logo * alpha
    actual_region = src_np[top:bottom, left:right]
    diff = np.abs(expected_region - actual_region).astype(np.float32)
    max_diff = float(diff.max())
    if max_diff > pixel_tolerance:
        msg = (
            f"A8 round-trip failed at t={frame_time}s, bbox={bbox}: "
            f"max pixel diff {max_diff:.2f} > tolerance {pixel_tolerance}"
        )
        raise VerificationError(msg)
    log.info(
        "alpha_round_trip_pass",
        frame_time=frame_time,
        bbox=bbox,
        max_pixel_diff=max_diff,
        tolerance=pixel_tolerance,
    )


@dataclass(frozen=True)
class PngRoundTripResult:
    """Result of verifying alpha round-trip on PNG frames."""

    frame_index: int
    max_diff: float
    mean_diff: float
    alpha_region_max_diff: float
    alpha_region_mean_diff: float
    non_alpha_max_diff: float
    passed: bool


def verify_alpha_round_trip_png(  # noqa: PLR0913
    frames_in_dir: Path,
    frames_out_dir: Path,
    alpha_map: np.ndarray,
    bbox: tuple[int, int, int, int],
    logo_rgb: tuple[int, int, int] = (255, 255, 255),
    *,
    frame_indices: list[int] | None = None,
    max_diff_tolerance: float = 2.0,
    mean_diff_tolerance: float = 0.5,
) -> list[PngRoundTripResult]:
    """Verify alpha round-trip at the PNG layer (before H.264 re-encode).

    For each sampled frame index, loads the source PNG from *frames_in_dir* and
    the cleaned PNG from *frames_out_dir*, applies forward alpha at the bbox,
    and compares with the source. This avoids H.264 re-encode noise that would
    be present in verify_alpha_round_trip.

    Acceptance criteria (per frame):
      - alpha_region_max_diff <= max_diff_tolerance
      - alpha_region_mean_diff <= mean_diff_tolerance
      - non_alpha_max_diff == 0 (outside bbox must be byte-identical)

    Args:
        frames_in_dir: directory containing source PNG frames (frame_XXXX.png).
        frames_out_dir: directory containing cleaned PNG frames (0000.png, etc.).
        alpha_map: (H, W) float32 array in [0, 1].
        bbox: (left, top, right, bottom) matching alpha_map shape.
        logo_rgb: per-channel logo color.
        frame_indices: which frame numbers to verify (default: [0, 60, 120, 180, 239]).
        max_diff_tolerance: maximum allowed per-pixel uint8 diff in alpha region.
        mean_diff_tolerance: maximum allowed mean per-pixel diff in alpha region.

    Returns:
        List of PngRoundTripResult for each verified frame.

    Raises:
        VerificationError: if any frame fails the tolerance criteria.
    """
    left, top, right, bottom = bbox
    if alpha_map.shape != (bottom - top, right - left):
        msg = f"alpha_map shape {alpha_map.shape} doesn't match bbox size ({bottom - top}, {right - left})"
        raise VerificationError(msg)

    if frame_indices is None:
        frame_indices = [0, 60, 120, 180, 239]

    alpha_3d = alpha_map[..., np.newaxis]
    logo = np.array(logo_rgb, dtype=np.float32)

    results: list[PngRoundTripResult] = []

    for idx in frame_indices:
        src_path = frames_in_dir / f"{idx:04d}.png"
        clean_path = frames_out_dir / f"{idx:04d}.png"
        if not src_path.is_file():
            log.warning("png_round_trip_missing_source", index=idx, path=str(src_path))
            continue
        if not clean_path.is_file():
            log.warning("png_round_trip_missing_clean", index=idx, path=str(clean_path))
            continue

        src_np = np.asarray(Image.open(src_path).convert("RGB"), dtype=np.float32)
        clean_np = np.asarray(Image.open(clean_path).convert("RGB"), dtype=np.float32)

        # Forward alpha in bbox: expected = clean * (1-alpha) + logo * alpha
        expected_region = clean_np[top:bottom, left:right] * (1.0 - alpha_3d) + logo * alpha_3d
        actual_region = src_np[top:bottom, left:right]
        alpha_diff = np.abs(expected_region - actual_region)

        # Outside bbox: should be unchanged (byte-identical).
        outside_mask = np.ones(clean_np.shape[:2], dtype=bool)
        outside_mask[top:bottom, left:right] = False
        non_alpha_diff = np.abs(clean_np - src_np)
        non_alpha_max = float(non_alpha_diff[outside_mask].max()) if outside_mask.any() else 0.0

        alpha_max = float(alpha_diff.max())
        alpha_mean = float(alpha_diff.mean())

        passed = alpha_max <= max_diff_tolerance and alpha_mean <= mean_diff_tolerance and non_alpha_max == 0.0

        result = PngRoundTripResult(
            frame_index=idx,
            max_diff=alpha_max,
            mean_diff=alpha_mean,
            alpha_region_max_diff=alpha_max,
            alpha_region_mean_diff=alpha_mean,
            non_alpha_max_diff=non_alpha_max,
            passed=passed,
        )
        results.append(result)

        if not passed:
            msg = (
                f"PNG round-trip failed at frame {idx:04d}: "
                f"alpha max_diff={alpha_max:.2f} (tol={max_diff_tolerance}), "
                f"alpha mean_diff={alpha_mean:.4f} (tol={mean_diff_tolerance}), "
                f"non-alpha max_diff={non_alpha_max:.2f}"
            )
            raise VerificationError(msg)

        log.info(
            "png_round_trip_pass",
            frame=idx,
            bbox=bbox,
            alpha_max_diff=alpha_max,
            alpha_mean_diff=alpha_mean,
            non_alpha_max_diff=non_alpha_max,
        )

    return results
