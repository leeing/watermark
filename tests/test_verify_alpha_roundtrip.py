"""Tests for the alpha round-trip verification (A8 and PNG-layer)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from watermark.consts import VIDEO_SIZE
from watermark.services import verify as verify_svc


def _fake_frame_array(rgb: tuple[int, int, int], size: tuple[int, int] = VIDEO_SIZE) -> np.ndarray:
    """Build a solid-color (H, W, 3) uint8 array."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[..., 0] = rgb[0]
    arr[..., 1] = rgb[1]
    arr[..., 2] = rgb[2]
    return arr


def _save_png(path: Path, arr: np.ndarray) -> None:
    """Save a uint8 (H, W, 3) array as PNG via PIL."""
    from PIL import Image

    Image.fromarray(arr, mode="RGB").save(path)


def test_verify_alpha_round_trip_passes_when_clean_is_lossless(tmp_path: Path) -> None:
    """A8: when `clean` was produced by exact reverse, forward(clean) ≈ source (within tolerance)."""
    # Build source and clean frames (no re-encode noise)
    bbox = (10, 10, 26, 26)  # 16x16 inner region
    # The "original" pixel is what we want to recover
    original = np.full((16, 16, 3), [100, 120, 140], dtype=np.float32)
    alpha = np.full((16, 16), 0.4, dtype=np.float32)
    alpha_3d = alpha[..., np.newaxis]
    logo = np.array([200, 220, 240], dtype=np.float32)
    # Forward alpha to create the source frame
    src_region = original * (1.0 - alpha_3d) + logo * alpha_3d
    # Reverse alpha to create the clean frame
    clean_region = (src_region - alpha_3d * logo) / (1.0 - alpha_3d)

    # Embed in full-frame arrays (1080x1920) and clip to uint8
    src_arr = _fake_frame_array((0, 0, 0))
    src_arr[bbox[1] : bbox[3], bbox[0] : bbox[2]] = np.clip(src_region, 0, 255).astype(np.uint8)
    clean_arr = _fake_frame_array((0, 0, 0))
    clean_arr[bbox[1] : bbox[3], bbox[0] : bbox[2]] = np.clip(clean_region, 0, 255).astype(np.uint8)

    from PIL import Image

    def fake_extract(video: Path, time_seconds: float, *, ffmpeg: str | None = None) -> object:
        return Image.fromarray(src_arr if str(video).endswith("src.mp4") else clean_arr, mode="RGB")

    # H.264 re-encode introduces ~2-3 uint8 drift; the test setup itself adds
    # another ~1 from float32→uint8 round-trip. Use 5 to give a small safety margin.
    with patch.object(verify_svc, "extract_frame", fake_extract):
        # Should not raise
        verify_svc.verify_alpha_round_trip(
            source_video=tmp_path / "src.mp4",
            clean_video=tmp_path / "clean.mp4",
            alpha_map=alpha,
            bbox=bbox,
            logo_rgb=(200, 220, 240),
            pixel_tolerance=5,
        )


def test_verify_alpha_round_trip_raises_on_large_drift(tmp_path: Path) -> None:
    """A8: a tampered clean frame whose forward != source must raise."""
    bbox = (10, 10, 26, 26)
    src_arr = _fake_frame_array((100, 100, 100))
    # Intentionally create a "clean" that does NOT match the source.
    # Use alpha=0, so forward(clean) = clean (i.e., clean IS the original).
    # To make it fail, make clean differ from source by 50 units.
    clean_arr = _fake_frame_array((150, 150, 150))

    alpha = np.full((16, 16), 0.4, dtype=np.float32)

    from PIL import Image

    def fake_extract(video: Path, time_seconds: float, *, ffmpeg: str | None = None) -> object:
        return Image.fromarray(src_arr if str(video).endswith("src.mp4") else clean_arr, mode="RGB")

    with (
        patch.object(verify_svc, "extract_frame", fake_extract),
        pytest.raises(verify_svc.VerificationError, match="max pixel diff"),
    ):
        verify_svc.verify_alpha_round_trip(
            source_video=tmp_path / "src.mp4",
            clean_video=tmp_path / "clean.mp4",
            alpha_map=alpha,
            bbox=bbox,
            logo_rgb=(200, 220, 240),
            pixel_tolerance=2,
        )


def test_verify_alpha_round_trip_png_passes(tmp_path: Path) -> None:
    """PNG-layer verifier: exact round-trip should pass with strict tolerances."""
    bbox = (10, 10, 26, 26)  # 16×16 region
    original = np.full((16, 16, 3), [100, 120, 140], dtype=np.float32)
    alpha = np.full((16, 16), 0.4, dtype=np.float32)
    alpha_3d = alpha[..., np.newaxis]
    logo = np.array([200, 220, 240], dtype=np.float32)

    # Create source frame at 32×32 (small for test speed)
    src_region = original * (1.0 - alpha_3d) + logo * alpha_3d
    clean_region = (src_region - alpha_3d * logo) / (1.0 - alpha_3d)

    frame_size = (32, 32)
    src_arr = _fake_frame_array((0, 0, 0), size=frame_size)
    src_arr[bbox[1] : bbox[3], bbox[0] : bbox[2]] = np.clip(src_region, 0, 255).astype(np.uint8)
    clean_arr = _fake_frame_array((0, 0, 0), size=frame_size)
    clean_arr[bbox[1] : bbox[3], bbox[0] : bbox[2]] = np.clip(clean_region, 0, 255).astype(np.uint8)

    # Write PNGs in the expected naming convention.
    frames_in = tmp_path / "frames_in"
    frames_out = tmp_path / "frames_out"
    frames_in.mkdir()
    frames_out.mkdir()
    _save_png(frames_in / "0000.png", src_arr)
    _save_png(frames_out / "0000.png", clean_arr)
    _save_png(frames_in / "0060.png", src_arr)
    _save_png(frames_out / "0060.png", clean_arr)

    results = verify_svc.verify_alpha_round_trip_png(
        frames_in,
        frames_out,
        alpha,
        bbox,
        logo_rgb=(200, 220, 240),
        frame_indices=[0, 60],
        max_diff_tolerance=2.0,
        mean_diff_tolerance=0.5,
    )
    assert len(results) == 2
    for r in results:
        assert r.passed
        assert r.alpha_region_max_diff <= 2.0
        assert r.alpha_region_mean_diff <= 0.5


def test_verify_alpha_round_trip_png_raises_on_drift(tmp_path: Path) -> None:
    """PNG-layer verifier raises when clean frame is tampered."""
    bbox = (10, 10, 26, 26)
    frame_size = (32, 32)
    alpha = np.full((16, 16), 0.4, dtype=np.float32)

    src_arr = _fake_frame_array((100, 100, 100), size=frame_size)
    clean_arr = _fake_frame_array((150, 150, 150), size=frame_size)  # tampered

    frames_in = tmp_path / "frames_in"
    frames_out = tmp_path / "frames_out"
    frames_in.mkdir()
    frames_out.mkdir()
    _save_png(frames_in / "0000.png", src_arr)
    _save_png(frames_out / "0000.png", clean_arr)

    with pytest.raises(verify_svc.VerificationError, match="PNG round-trip failed"):
        verify_svc.verify_alpha_round_trip_png(
            frames_in,
            frames_out,
            alpha,
            bbox,
            logo_rgb=(200, 220, 240),
            frame_indices=[0],
            max_diff_tolerance=2.0,
            mean_diff_tolerance=0.5,
        )


def test_verify_alpha_round_trip_png_non_alpha_unchanged(tmp_path: Path) -> None:
    """PNG-layer verifier checks that non-alpha region is byte-identical."""
    bbox = (10, 10, 26, 26)
    frame_size = (32, 32)
    alpha = np.full((16, 16), 0.4, dtype=np.float32)
    alpha_3d = alpha[..., np.newaxis]
    logo = np.array([200, 220, 240], dtype=np.float32)

    # Source with watermark, clean with reverse applied.
    original = np.full((16, 16, 3), [30, 40, 50], dtype=np.float32)
    src_region = original * (1.0 - alpha_3d) + logo * alpha_3d
    clean_region = (src_region - alpha_3d * logo) / (1.0 - alpha_3d)

    src_arr = _fake_frame_array((0, 0, 0), size=frame_size)
    src_arr[bbox[1] : bbox[3], bbox[0] : bbox[2]] = np.clip(src_region, 0, 255).astype(np.uint8)
    clean_arr = _fake_frame_array((0, 0, 0), size=frame_size)
    clean_arr[bbox[1] : bbox[3], bbox[0] : bbox[2]] = np.clip(clean_region, 0, 255).astype(np.uint8)

    # Tamper a pixel outside the bbox to be different.
    clean_arr[0, 0, 0] = 99  # different from source's 0
    src_arr[0, 0, 0] = 0

    frames_in = tmp_path / "frames_in"
    frames_out = tmp_path / "frames_out"
    frames_in.mkdir()
    frames_out.mkdir()
    _save_png(frames_in / "0000.png", src_arr)
    _save_png(frames_out / "0000.png", clean_arr)

    with pytest.raises(verify_svc.VerificationError, match="non-alpha"):
        verify_svc.verify_alpha_round_trip_png(
            frames_in,
            frames_out,
            alpha,
            bbox,
            logo_rgb=(200, 220, 240),
            frame_indices=[0],
            max_diff_tolerance=2.0,
            mean_diff_tolerance=0.5,
        )
