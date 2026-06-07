"""Tests for the alpha service (reverse alpha blending)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from watermark.services import alpha as alpha_svc


def _make_alpha_png(path: Path, size: int, *, max_value: int = 128) -> Path:
    """Write a square RGB PNG where every pixel's max channel equals max_value."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[..., 0] = max_value
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def test_load_alpha_map_decodes_max_channel(tmp_path: Path) -> None:
    """A 96×96 PNG with max channel = 255 yields α = 1.0 everywhere."""
    path = _make_alpha_png(tmp_path / "alpha.png", size=96, max_value=255)
    alpha = alpha_svc.load_alpha_map(path)
    assert alpha.shape == (96, 96)
    assert alpha.dtype == np.float32
    assert np.allclose(alpha, 1.0)


def test_load_alpha_map_normalizes_to_zero(tmp_path: Path) -> None:
    """A 96×96 PNG with max channel = 0 yields α = 0.0 (no watermark)."""
    path = _make_alpha_png(tmp_path / "alpha.png", size=96, max_value=0)
    alpha = alpha_svc.load_alpha_map(path)
    assert np.allclose(alpha, 0.0)


def test_load_alpha_map_picks_max_across_channels(tmp_path: Path) -> None:
    """The max of R/G/B is used (not luminance or first channel)."""
    path = tmp_path / "alpha.png"
    size = 32
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[..., 0] = 50
    arr[..., 1] = 200  # should win
    arr[..., 2] = 100
    Image.fromarray(arr, mode="RGB").save(path)
    alpha = alpha_svc.load_alpha_map(path)
    assert np.allclose(alpha, 200.0 / 255.0)


def test_load_alpha_map_rejects_non_square(tmp_path: Path) -> None:
    """A non-square image raises AlphaMapError."""
    path = tmp_path / "wide.png"
    Image.new("RGB", (96, 48), (128, 128, 128)).save(path)
    with pytest.raises(alpha_svc.AlphaMapError, match="must be square"):
        alpha_svc.load_alpha_map(path)


def test_load_alpha_map_raises_on_missing_file(tmp_path: Path) -> None:
    """A nonexistent path raises AlphaMapError."""
    with pytest.raises(alpha_svc.AlphaMapError, match="not found"):
        alpha_svc.load_alpha_map(tmp_path / "does_not_exist.png")


def test_load_logo_map_rejects_shape_mismatch(tmp_path: Path) -> None:
    """Logo map must match the alpha map's H×W×3 shape."""
    path = tmp_path / "logo.npy"
    np.save(path, np.zeros((16, 16, 3), dtype=np.float32))
    with pytest.raises(alpha_svc.AlphaMapError, match="shape"):
        alpha_svc.load_logo_map(path, expected_shape=(32, 32))


def test_inpaint_alpha_frame_mathematical_inverse(tmp_path: Path) -> None:
    """Applying reverse then forward returns the original frame within 2 uint8."""
    size = 32
    alpha = np.full((size, size), 0.4, dtype=np.float32)
    original = np.random.RandomState(42).randint(0, 256, (size, size, 3), dtype=np.uint8)
    logo = np.array([200, 220, 240], dtype=np.float32)
    alpha_3d = alpha[..., np.newaxis]
    wm = (original.astype(np.float32) * (1.0 - alpha_3d) + logo * alpha_3d).astype(np.uint8)
    frame = Image.fromarray(wm, mode="RGB")
    out = alpha_svc.inpaint_alpha_frame(frame, alpha, bbox=(0, 0, size, size), logo_rgb=(200, 220, 240))
    restored = np.array(out)
    # Tolerance: float32 -> uint8 cast may round up to 2 levels
    assert np.abs(restored.astype(int) - original.astype(int)).max() <= 2


def test_inpaint_alpha_frame_outside_bbox_unchanged() -> None:
    """Pixels outside the bbox are passed through byte-identically.

    Bbox is the full image; surrounding "outside" is a 4-px frame around
    a 24x24 inner block whose alpha is non-zero. Those 4-px strips must
    be unchanged because alpha=0 there.
    """
    size = 32
    bbox = (0, 0, size, size)  # full image so size matches
    # Inner 24x24 has α=0.5, surrounding 4-px frame has α=0
    alpha = np.zeros((size, size), dtype=np.float32)
    alpha[4:28, 4:28] = 0.5

    arr = np.random.RandomState(0).randint(0, 256, (size, size, 3), dtype=np.uint8)
    frame = Image.fromarray(arr, mode="RGB")
    out = np.array(alpha_svc.inpaint_alpha_frame(frame, alpha, bbox=bbox))

    # Outside the 24x24 inner block (where alpha was 0) must be unchanged
    top_strip = out[0:4, :, :]
    bottom_strip = out[28:32, :, :]
    left_strip = out[4:28, 0:4, :]
    right_strip = out[4:28, 28:32, :]
    assert np.array_equal(top_strip, arr[0:4, :, :])
    assert np.array_equal(bottom_strip, arr[28:32, :, :])
    assert np.array_equal(left_strip, arr[4:28, 0:4, :])
    assert np.array_equal(right_strip, arr[4:28, 28:32, :])


def test_inpaint_alpha_frame_rejects_size_mismatch(tmp_path: Path) -> None:
    """Bbox whose size doesn't match the alpha map raises."""
    alpha = np.zeros((32, 32), dtype=np.float32)
    frame = Image.new("RGB", (64, 64), (10, 10, 10))
    with pytest.raises(alpha_svc.AlphaMapError, match="doesn't match"):
        alpha_svc.inpaint_alpha_frame(frame, alpha, bbox=(0, 0, 16, 16))


def test_adaptive_reverse_blend_uses_wider_default_feather() -> None:
    """Default alpha feathering spans beyond 2px to avoid abrupt edge jumps."""
    assert alpha_svc.DEFAULT_ALPHA_FEATHER_WIDTH >= 6


def test_adaptive_reverse_blend_uses_per_pixel_safe_alpha_on_dark_region() -> None:
    """Dark frames should not clamp all active alpha pixels to global bg_min/255."""
    size = 16
    bbox = (0, 0, size, size)
    alpha = np.zeros((size, size), dtype=np.float32)
    alpha[4:12, 4:12] = 0.53
    alpha[6:10, 6:10] = 0.16
    frame_arr = np.full((size, size, 3), 40, dtype=np.uint8)
    # Simulate watermarked pixels: the high-alpha center is brighter than the
    # dark background and therefore has much more safe headroom than bg_min/255.
    frame_arr[4:12, 4:12] = 138
    frame_arr[6:10, 6:10] = 73
    frame = Image.fromarray(frame_arr, mode="RGB")

    restored = np.array(
        alpha_svc.adaptive_reverse_blend(
            frame,
            alpha,
            bbox,
            feather_width=0,
        ),
    )

    global_limited = np.array(
        alpha_svc.adaptive_reverse_blend(
            frame,
            np.minimum(alpha, 40.0 / 255.0),
            bbox,
            feather_width=0,
        ),
    )

    high_alpha_center = restored[5, 5, 0]
    global_limited_center = global_limited[5, 5, 0]
    assert high_alpha_center < global_limited_center - 100
    assert high_alpha_center <= 7


def test_adaptive_reverse_blend_supports_black_shadow_logo_map() -> None:
    """A per-pixel black logo map removes dark shadow pixels losslessly."""
    size = 16
    bbox = (0, 0, size, size)
    alpha = np.zeros((size, size), dtype=np.float32)
    alpha[4:12, 4:12] = 0.25
    logo_map = np.full((size, size, 3), 255, dtype=np.float32)
    logo_map[4:12, 4:12] = 0

    original_value = 128
    watermarked_value = int(original_value * (1.0 - 0.25))
    frame_arr = np.full((size, size, 3), original_value, dtype=np.uint8)
    frame_arr[4:12, 4:12] = watermarked_value
    frame = Image.fromarray(frame_arr, mode="RGB")

    restored = np.array(
        alpha_svc.adaptive_reverse_blend(
            frame,
            alpha,
            bbox,
            logo_map=logo_map,
            feather_width=0,
        ),
    )

    assert np.abs(restored[6, 6, 0].astype(int) - original_value) <= 1


def test_adaptive_reverse_blend_does_not_feather_internal_alpha_edge() -> None:
    """Feathering should not blend the watermark's own anti-aliased edge back in."""
    size = 32
    bbox = (0, 0, size, size)
    alpha = np.zeros((size, size), dtype=np.float32)
    alpha[12:20, 12:20] = 0.5
    original_value = 40
    watermarked_value = int(255 * 0.5 + original_value * 0.5)
    frame_arr = np.full((size, size, 3), original_value, dtype=np.uint8)
    frame_arr[12:20, 12:20] = watermarked_value
    frame = Image.fromarray(frame_arr, mode="RGB")

    restored = np.array(
        alpha_svc.adaptive_reverse_blend(
            frame,
            alpha,
            bbox,
            feather_width=8,
        ),
    )

    assert restored[12, 16, 0] <= original_value + 2


def test_find_anchor_locates_bright_pattern(tmp_path: Path) -> None:
    """find_anchor uses round-trip error to locate the watermark position."""
    size = 96
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    center = size / 2
    d2 = (xx - center) ** 2 + (yy - center) ** 2
    alpha = np.clip(1.0 - d2 / (size * 1.5), 0, 1).astype(np.float32)
    logo = np.array([255, 255, 255], dtype=np.float32)

    # Build a 200x200 source frame with alpha-composited watermark at (50, 50).
    bg = np.full((200, 200, 3), 30, dtype=np.float32)
    placed_top, placed_left = 50, 50
    region = bg[placed_top : placed_top + size, placed_left : placed_left + size, :]
    region[:] = region * (1.0 - alpha[..., np.newaxis]) + logo * alpha[..., np.newaxis]
    frame = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), mode="RGB")

    # Catalog says (100, 100); find_anchor should find (50, 50) via round-trip.
    catalog = (100, 100, 196, 196)
    found = alpha_svc.find_anchor(frame, alpha, catalog, search_radius=120)
    assert found == (50, 50, 146, 146), f"expected (50,50,146,146) got {found}"


def test_find_anchor_finds_position_within_small_search_radius(tmp_path: Path) -> None:
    """When the actual position is close to catalog, find_anchor returns it."""
    size = 96
    # Distinctive alpha pattern: bright spot in one corner, gradient to zero.
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # Offset the peak to (20, 30) within the 96x96 — makes the pattern asymmetric.
    d2 = (xx - 20) ** 2 + (yy - 30) ** 2
    alpha = np.clip(np.exp(-d2 / 200.0), 0, 1).astype(np.float32)
    logo = np.array([255, 255, 255], dtype=np.float32)

    # Build frame using alpha compositing at (105, 110).
    bg = np.full((300, 300, 3), 20, dtype=np.float32)
    target_top, target_left = 105, 110
    region = bg[target_top : target_top + size, target_left : target_left + size, :]
    region[:] = region * (1.0 - alpha[..., np.newaxis]) + logo * alpha[..., np.newaxis]
    frame = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), mode="RGB")

    catalog = (100, 100, 196, 196)
    found = alpha_svc.find_anchor(frame, alpha, catalog, search_radius=20)
    assert found == (110, 105, 206, 201)


def test_calibrate_from_video_recovers_known_alpha(tmp_path: Path) -> None:
    """Self-calibration: given a synthetic video with a known watermark, recover it.

    We construct a tiny 192x192 video (4 frames at distinct orig values), plant
    a known alpha map inside a 96x96 bbox, apply forward alpha, and verify
    `calibrate_from_video` recovers alpha within 0.05 mean abs error.
    """
    import subprocess

    size = 96
    # Synthetic alpha map: a small bright region in the center
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    true_alpha = np.clip(1.0 - ((xx - 48) ** 2 + (yy - 48) ** 2) / 800.0, 0, 1).astype(np.float32)

    # Build 4 frames at 200x200, each with different orig under the bbox
    frames = []
    orig_colors = [10, 50, 100, 180]  # different orig values
    for orig_val in orig_colors:
        canvas = np.full((200, 200, 3), orig_val, dtype=np.uint8)
        # Add some non-uniformity so orig varies spatially
        for y in range(200):
            for x in range(200):
                canvas[y, x] = min(255, max(0, int(orig_val) + (x + y) % 10))
        # Apply watermark
        bbox_top, bbox_left = 50, 50
        for dy in range(size):
            for dx in range(size):
                a = true_alpha[dy, dx]
                for c in range(3):
                    canvas[bbox_top + dy, bbox_left + dx, c] = np.clip(
                        int(a * 255 + (1 - a) * canvas[bbox_top + dy, bbox_left + dx, c]),
                        0,
                        255,
                    )
        frames.append(canvas)

    # Save as a 24-frame mp4 (24 seconds at 1 fps) so explicit sample times
    # always land on a valid frame.
    import PIL.Image as PILImage

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for i, f in enumerate(frames):
        # Repeat each frame 6 times to get 24 total frames at 1 fps
        PILImage.fromarray(f).save(raw_dir / f"{i:02d}.png")
    # Duplicate frames to reach 24
    for i in range(len(frames), 24):
        PILImage.fromarray(frames[i % len(frames)]).save(raw_dir / f"{i:02d}.png")
    video_path = tmp_path / "test.mp4"
    ffmpeg_bin = "/opt/homebrew/bin/ffmpeg"
    subprocess.run(  # noqa: S603
        [
            ffmpeg_bin,
            "-y",
            "-framerate",
            "1",
            "-i",
            str(raw_dir / "%02d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )

    # Calibrate with explicit sample times spread across the 24s video
    bbox = (bbox_left, bbox_top, bbox_left + size, bbox_top + size)
    sample_times = [0.5, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    recovered = alpha_svc.calibrate_from_video(video_path, bbox, sample_times=sample_times)

    # Compare: should match the true_alpha within reasonable tolerance
    mean_err = float(np.abs(recovered - true_alpha).mean())
    max_err = float(np.abs(recovered - true_alpha).max())
    assert mean_err < 0.15, f"mean abs err {mean_err:.3f} too high"
    assert max_err < 0.3, f"max abs err {max_err:.3f} too high"
    assert recovered.shape == (size, size)
    assert recovered.dtype == np.float32


def test_calibrate_from_solid_video_recovers_alpha_from_grey(tmp_path: Path) -> None:
    """Solid-background calibration: known bg + watermark → recover alpha."""
    import subprocess

    size = 96
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d2 = (xx - 48) ** 2 + (yy - 48) ** 2
    true_alpha = np.clip(np.exp(-d2 / 400.0), 0, 1).astype(np.float32)

    bg_rgb = (126, 126, 127)
    logo_rgb = (255, 255, 255)
    bg = np.array(bg_rgb, dtype=np.float32)
    logo = np.array(logo_rgb, dtype=np.float32)

    # Build a short solid video: 10 identical frames at 200x200 with the
    # watermark at a known position.
    wm_top, wm_left = 50, 50
    frames = []
    for _ in range(10):
        canvas = np.full((200, 200, 3), bg, dtype=np.uint8)
        region = canvas[wm_top : wm_top + size, wm_left : wm_left + size, :]
        region_f = region.astype(np.float32)
        region[:] = np.clip(
            region_f * (1.0 - true_alpha[..., np.newaxis]) + logo * true_alpha[..., np.newaxis],
            0,
            255,
        ).astype(np.uint8)
        frames.append(canvas)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for i, f in enumerate(frames):
        from PIL import Image as PILImage

        PILImage.fromarray(f).save(raw_dir / f"{i:02d}.png")
    # Duplicate to 24 frames at 1 fps.
    for i in range(len(frames), 24):
        from PIL import Image as PILImage

        PILImage.fromarray(frames[i % len(frames)]).save(raw_dir / f"{i:02d}.png")
    video_path = tmp_path / "solid.mp4"
    ffmpeg_bin = "/opt/homebrew/bin/ffmpeg"
    subprocess.run(  # noqa: S603
        [
            ffmpeg_bin,
            "-y",
            "-framerate",
            "1",
            "-i",
            str(raw_dir / "%02d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )

    alpha, meta = alpha_svc.calibrate_from_solid_video(
        video_path,
        bg_rgb,
        logo_rgb=logo_rgb,
        alpha_size=size,
    )
    assert alpha.shape == (size, size)
    assert alpha.dtype == np.float32
    assert meta["background_rgb"] == [126, 126, 127]
    assert meta["logo_rgb"] == [255, 255, 255]

    # Compare recovered alpha to ground truth (allows H.264 compression drift).
    mean_err = float(np.abs(alpha - true_alpha).mean())
    assert mean_err < 0.05, f"mean α error {mean_err:.4f} > 0.05"


def test_calibrate_from_solid_video_rejects_same_logo_and_bg(tmp_path: Path) -> None:
    """When logo ≈ background, division by ~0 raises ValueError."""
    import subprocess

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    from PIL import Image as PILImage

    for i in range(24):
        PILImage.new("RGB", (200, 200), (100, 100, 100)).save(raw_dir / f"{i:02d}.png")
    video_path = tmp_path / "same.mp4"
    subprocess.run(  # noqa: S603
        [
            "/opt/homebrew/bin/ffmpeg",
            "-y",
            "-framerate",
            "1",
            "-i",
            str(raw_dir / "%02d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="too close"):
        alpha_svc.calibrate_from_solid_video(video_path, (100, 100, 100), logo_rgb=(100, 101, 100))


def test_refine_subpixel_shift() -> None:
    """refine_subpixel_shift recovers a known subpixel shift from synthetic frame."""
    size = 32
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    center = size / 2.0
    d2 = (xx - center) ** 2 + (yy - center) ** 2
    alpha = np.clip(1.0 - d2 / 100.0, 0, 1).astype(np.float32)
    logo = np.full((size, size, 3), 255.0, dtype=np.float32)

    # Sub-pixel shift to plant: dx=-0.3, dy=0.2
    planted_dx, planted_dy = -0.3, 0.2

    import cv2

    h, w = alpha.shape
    m_mat = np.float32([[1, 0, planted_dx], [0, 1, planted_dy]])
    alpha_shifted = cv2.warpAffine(
        alpha, m_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    logo_shifted = cv2.warpAffine(
        logo, m_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )

    # Background: dark spatial gradient so that shift mismatch causes underflow
    yy_bg, xx_bg = np.mgrid[0:64, 0:64].astype(np.float32)
    bg = np.zeros((64, 64, 3), dtype=np.float32)
    bg[..., 0] = xx_bg * 0.1
    bg[..., 1] = yy_bg * 0.1
    bg[..., 2] = (xx_bg + yy_bg) * 0.05

    bbox = (10, 10, 10 + size, 10 + size)
    left, top, right, bottom = bbox

    # Apply forward blend with subpixel-shifted maps
    region = bg[top:bottom, left:right]
    alpha_3d = alpha_shifted[..., np.newaxis]
    region[:] = region * (1.0 - alpha_3d) + logo_shifted * alpha_3d

    frame = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))

    # Recover sub-pixel shift
    dx, dy = alpha_svc.refine_subpixel_shift([frame], alpha, None, bbox)

    assert abs(dx - planted_dx) <= 0.06, f"expected {planted_dx} got {dx}"
    assert abs(dy - planted_dy) <= 0.06, f"expected {planted_dy} got {dy}"
