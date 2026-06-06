"""Reverse alpha blending — mathematically lossless watermark removal.

Per-pixel formula (Lossless.md §2.2):

    original = (watermarked - alpha * logo) / (1 - alpha)

The alpha map is loaded from a pre-computed PNG (calibrated externally, MIT
licensed from GargantuaX/gemini-watermark-remover). The logo color is pure
white (255, 255, 255) per the upstream calibration.

Anchor search: in practice the watermark position in `omni.mp4` may not match
the catalog's predicted (left, top) by a few pixels (sometimes 50+ px when
Gemini ships a new version before GargantuaX recalibrates). `find_anchor`
uses `cv2.matchTemplate` to locate the actual watermark position within a
search window around the catalog bbox. `find_anchor_multiframe` extends this
with multi-frame voting for robust position detection.

Adaptive blending: `adaptive_reverse_blend` applies per-pixel alpha clamping
to prevent numerical underflow on dark backgrounds. For each pixel, the safe
alpha limit is ``min_channel(pixel) / max(logo)`` — this is less conservative
than global clamping because watermark pixels are brighter than background
(the white logo was blended in). Combined with Gaussian edge feathering,
this produces seamless removal even on very dark frames.

Self-calibration: when no calibrated alpha map is available, `calibrate_from_video`
extracts one from the video itself by exploiting the fact that the watermark
is static and the original content varies across frames. The 4 corners of
the 96x96 bbox are alpha=0 control points that reveal the original directly;
the rest of the bbox is then solved for alpha.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC
from pathlib import Path

import numpy as np
import structlog
from PIL import Image

log = structlog.get_logger()

# Epsilon for guarding against division by (1 - alpha) ≈ 0 in reverse alpha.
_DENOM_EPSILON: float = 1e-6

# Threshold below which alpha is treated as zero to avoid numerical noise.
_ALPHA_ZERO_THRESHOLD: float = 1e-4

# Maximum 8-bit channel value.
_MAX_CHANNEL_VALUE: float = 255.0

# Default alpha-edge feather radius.  The watermark's semi-transparent edge
# spans more than the previous 2px ramp; 8px avoids abrupt boundary jumps while
# keeping the core at full reverse-blend strength.
DEFAULT_ALPHA_FEATHER_WIDTH: int = 8


class AlphaMapError(RuntimeError):
    """Raised when the alpha map is missing, malformed, or wrong shape."""


def load_alpha_map(path: Path) -> np.ndarray:
    """Load an NxN alpha map and return float32 array in [0, 1].

    Supports two formats (auto-detected by extension):
        - `.png` / `.jpg`: square RGB PNG; per-pixel alpha is the maximum of
          the three RGB channels divided by 255.0. Matches GargantuaX's
          `calculateAlphaMap` decoding.
        - `.npy`: raw float32 numpy array saved by `calibrate_from_video`.

    Args:
        path: filesystem path to a square RGB PNG or .npy file.

    Returns:
        A `(H, W)` float32 numpy array in [0, 1].

    Raises:
        AlphaMapError: if the file is missing, unreadable, or non-square.
    """
    if not path.is_file():
        msg = f"alpha map not found: {path}"
        raise AlphaMapError(msg)
    if path.suffix == ".npy":
        npy_alpha = np.load(path).astype(np.float32)
        if npy_alpha.ndim != 2 or npy_alpha.shape[0] != npy_alpha.shape[1]:  # noqa: PLR2004
            msg = f"alpha map must be square, got {npy_alpha.shape}"
            raise AlphaMapError(msg)
        log.info(
            "alpha_map_loaded",
            path=str(path),
            shape=npy_alpha.shape,
            alpha_max=float(npy_alpha.max()),
            source="npy",
        )
        return npy_alpha  # type: ignore[no-any-return]
    img = Image.open(path)
    width, height = img.size
    if width != height:
        msg = f"alpha map must be square, got {width}x{height}"
        raise AlphaMapError(msg)
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    max_channel = arr.max(axis=2)
    alpha: np.ndarray = max_channel / 255.0
    log.info("alpha_map_loaded", path=str(path), shape=alpha.shape, alpha_max=float(alpha.max()))
    return alpha


def load_logo_map(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    """Load a per-pixel RGB logo map for mixed white/highlight and black/shadow layers.

    Args:
        path: filesystem path to a ``.npy`` array of shape ``(H, W, 3)``.
        expected_shape: expected ``(H, W)`` matching the alpha map.

    Returns:
        A float32 ``(H, W, 3)`` array in byte-value space ``[0, 255]``.

    Raises:
        AlphaMapError: if the file is missing or has the wrong shape.
    """
    if not path.is_file():
        msg = f"logo map not found: {path}"
        raise AlphaMapError(msg)
    logo_map = np.load(path).astype(np.float32)
    expected = (*expected_shape, 3)
    if logo_map.shape != expected:
        msg = f"logo map shape must be {expected}, got {logo_map.shape}"
        raise AlphaMapError(msg)
    logo_map = np.clip(logo_map, 0.0, _MAX_CHANNEL_VALUE)
    log.info("logo_map_loaded", path=str(path), shape=logo_map.shape)
    return np.asarray(logo_map, dtype=np.float32)


def inpaint_alpha_frame(
    frame_rgb: Image.Image,
    alpha_map: np.ndarray,
    bbox: tuple[int, int, int, int],
    logo_rgb: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Apply reverse alpha blending to a single frame at *bbox*.

    The alpha map is anchored at the catalog-predicted bbox (no dilate).
    Outside the bbox, pixels are passed through unchanged.

    Args:
        frame_rgb: a PIL Image in RGB mode.
        alpha_map: `(H, W)` float32 array from `load_alpha_map`.
        bbox: `(left, top, right, bottom)` matching the alpha map shape.
        logo_rgb: per-channel logo color (default white).

    Returns:
        A new PIL Image with the alpha region restored.
    """
    h, w = alpha_map.shape
    left, top, right, bottom = bbox
    if (right - left) != w or (bottom - top) != h:
        msg = f"bbox {(left, top, right, bottom)} doesn't match alpha map shape {(h, w)}"
        raise AlphaMapError(msg)

    frame_np = np.asarray(frame_rgb.convert("RGB"), dtype=np.float32)
    alpha = alpha_map[..., np.newaxis]  # (H, W, 1)
    logo = np.array(logo_rgb, dtype=np.float32)  # (3,)
    out = frame_np.copy()
    region = out[top:bottom, left:right]
    # (wm - alpha * logo) / (1 - alpha), per Lossless.md §2.2
    region[:] = (region - alpha * logo) / (1.0 - alpha)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def adaptive_reverse_blend(  # noqa: PLR0913
    frame_rgb: Image.Image,
    alpha_map: np.ndarray,
    bbox: tuple[int, int, int, int],
    logo_rgb: tuple[int, int, int] = (255, 255, 255),
    *,
    logo_map: np.ndarray | None = None,
    corner_size: int = 8,  # noqa: ARG001 — kept for API compat
    feather_width: int = DEFAULT_ALPHA_FEATHER_WIDTH,
) -> Image.Image:
    """Apply reverse alpha blending with per-pixel background-aware clamping.

    Unlike ``inpaint_alpha_frame``, this function clamps the effective alpha
    **per pixel** so that the reverse formula never produces negative values.
    The per-pixel safe limit is ``min_channel(pixel) / max(logo)``, which is
    much less conservative than the old global ``bg_min / logo_max`` approach
    — watermark pixels are brighter than their background (because the white
    logo was blended in), so their per-pixel limit is naturally higher.

    Edge feathering uses a distance-based blend at the boundary of the
    alpha region to prevent visible seams between modified and unmodified
    pixels.

    Args:
        frame_rgb: a PIL Image in RGB mode.
        alpha_map: ``(H, W)`` float32 array from ``load_alpha_map``.
        bbox: ``(left, top, right, bottom)`` matching the alpha map shape.
        logo_rgb: per-channel logo color (default white).
        logo_map: optional ``(H, W, 3)`` per-pixel logo color map. Use this
            for watermarks that mix white highlight pixels with black shadow
            pixels.
        corner_size: unused (kept for backward compatibility).
        feather_width: edge feather radius in pixels (0 to disable).

    Returns:
        A new PIL Image with the alpha region restored.
    """
    h, w = alpha_map.shape
    left, top, right, bottom = bbox
    if (right - left) != w or (bottom - top) != h:
        msg = f"bbox {(left, top, right, bottom)} doesn't match alpha map shape {(h, w)}"
        raise AlphaMapError(msg)

    frame_np = np.asarray(frame_rgb.convert("RGB"), dtype=np.float32)
    out = frame_np.copy()
    region = out[top:bottom, left:right]
    if logo_map is not None:
        if logo_map.shape != (h, w, 3):
            msg = f"logo_map shape must be {(h, w, 3)}, got {logo_map.shape}"
            raise AlphaMapError(msg)
        logo = np.clip(logo_map.astype(np.float32), 0.0, _MAX_CHANNEL_VALUE)
    else:
        logo = np.broadcast_to(np.array(logo_rgb, dtype=np.float32), (h, w, 3))

    # Per-pixel safe alpha: for each pixel, the maximum alpha that keeps the
    # reverse formula within [0, 255]. Since restored = (wm - α·logo)/(1-α),
    # the lower bound matters for white logo pixels and the upper bound matters
    # for black shadow pixels. Using the minimum across channels is strictest.
    # This is much less conservative than using a single global limit because
    # watermarked pixels carry their own safe headroom after compositing.
    lower_safe = np.ones_like(region, dtype=np.float32)
    positive_logo = logo > 0.0
    lower_safe = np.where(positive_logo, region / np.where(positive_logo, logo, 1.0), lower_safe)
    upper_safe = np.ones_like(region, dtype=np.float32)
    below_white_logo = logo < _MAX_CHANNEL_VALUE
    upper_safe = np.where(
        below_white_logo,
        (_MAX_CHANNEL_VALUE - region) / np.where(below_white_logo, _MAX_CHANNEL_VALUE - logo, 1.0),
        upper_safe,
    )
    per_channel_safe = np.minimum(lower_safe, upper_safe)
    per_pixel_safe = per_channel_safe.min(axis=2)  # (H, W) — strictest channel
    per_pixel_safe = np.clip(per_pixel_safe, 0.0, 1.0)

    effective_alpha = np.minimum(alpha_map, per_pixel_safe).astype(np.float32)
    clamped_count = int((alpha_map > per_pixel_safe).sum())
    if clamped_count > 0:
        log.debug(
            "alpha_per_pixel_clamped",
            clamped_pixels=clamped_count,
            original_alpha_max=float(alpha_map.max()),
            effective_alpha_max=float(effective_alpha.max()),
        )

    alpha_3d = effective_alpha[..., np.newaxis]  # (H, W, 1)
    denom = 1.0 - alpha_3d + _DENOM_EPSILON
    restored = (region - alpha_3d * logo) / denom

    # Edge feathering: linear ramp only at the outer bbox boundary. The alpha
    # map already encodes the anti-aliased star boundary; feathering that
    # internal boundary would blend the original watermark edge back in.
    if feather_width > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        dist_to_bbox_edge = np.minimum.reduce([xx + 1, yy + 1, w - xx, h - yy]).astype(np.float32)
        feather_mask = np.clip(dist_to_bbox_edge / float(feather_width), 0.0, 1.0)
        feather_3d = feather_mask[..., np.newaxis]
        region[:] = restored * feather_3d + region * (1.0 - feather_3d)
    else:
        region[:] = restored

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def find_anchor(  # noqa: PLR0913
    frame_rgb: Image.Image,
    alpha_map: np.ndarray,
    catalog_bbox: tuple[int, int, int, int],
    *,
    search_radius: int = 80,
    logo_rgb: tuple[int, int, int] = (255, 255, 255),  # noqa: ARG001 — kept for API compat
) -> tuple[int, int, int, int]:
    """Locate the actual watermark position via cv2.matchTemplate (ZNCC).

    Converts the alpha map to a luminance template and searches for the
    brightest matching position around *catalog_bbox*. ZNCC is robust to
    global brightness shifts and is the method used by GargantuaX.

    Round-trip error (forward(reverse(frame)) vs frame) is NOT used as the
    objective because it is a mathematical identity: for any position,
    forward(reverse(x)) = x exactly (up to float32 epsilon), regardless of
    whether the alpha map is correctly aligned.

    Args:
        frame_rgb: a PIL Image in RGB mode.
        alpha_map: `(H, W)` float32 array from `load_alpha_map`.
        catalog_bbox: catalog-predicted `(left, top, right, bottom)`.
        search_radius: pixels to search around catalog top-left in each direction.
        logo_rgb: unused (kept for backward API compatibility).

    Returns:
        The best `(left, top, right, bottom)` bbox found.
    """
    import cv2

    cat_left, cat_top, _cat_right, _cat_bottom = catalog_bbox
    h, w = alpha_map.shape
    img_w, img_h = frame_rgb.size

    # Convert frame to grayscale; alpha map as 0..255 uint8 template.
    # The watermark is always brighter than background (wm = bg*(1-a) + 255*a > bg),
    # so the alpha map directly encodes the expected brightness pattern.
    frame_l = np.asarray(frame_rgb.convert("L"), dtype=np.uint8)
    template = np.clip(alpha_map * 255.0, 0, 255).astype(np.uint8)

    # Extract the search window expanded by search_radius, clamped to image.
    win_left = max(0, cat_left - search_radius)
    win_top = max(0, cat_top - search_radius)
    win_right = min(img_w, cat_left + w + search_radius)
    win_bottom = min(img_h, cat_top + h + search_radius)
    if win_right - win_left < w or win_bottom - win_top < h:
        msg = f"search window too small for catalog_bbox={catalog_bbox} + {search_radius}px radius"
        raise AlphaMapError(msg)
    search_img = frame_l[win_top:win_bottom, win_left:win_right]

    # ZNCC: zero-mean normalized cross-correlation. 1.0 = perfect match.
    result = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    abs_left = win_left + max_loc[0]
    abs_top = win_top + max_loc[1]
    dx = abs_left - cat_left
    dy = abs_top - cat_top
    log.info(
        "anchor_found",
        catalog_bbox=catalog_bbox,
        search_radius=search_radius,
        best_dx=dx,
        best_dy=dy,
        best_score=float(max_val),
    )
    return (cat_left + dx, cat_top + dy, cat_left + dx + w, cat_top + dy + h)


def find_anchor_multiframe(
    frames: list[Image.Image],
    alpha_map: np.ndarray,
    hint_bbox: tuple[int, int, int, int],
    *,
    search_radius: int = 120,
) -> tuple[int, int, int, int]:
    """Multi-frame anchor search with majority voting.

    Runs ``find_anchor`` on each frame independently, then returns the
    median (left, top) position. This is more robust than single-frame
    matching when the video content creates local brightness patterns that
    confuse the ZNCC template match.

    Args:
        frames: list of PIL Images in RGB mode (at least 1).
        alpha_map: ``(H, W)`` float32 array from ``load_alpha_map``.
        hint_bbox: initial search hint ``(left, top, right, bottom)``.
        search_radius: pixels to search around hint top-left.

    Returns:
        The voted ``(left, top, right, bottom)`` bbox.
    """
    if not frames:
        msg = "find_anchor_multiframe requires at least 1 frame"
        raise ValueError(msg)

    h, w = alpha_map.shape
    lefts: list[int] = []
    tops: list[int] = []
    scores: list[float] = []

    for frame in frames:
        found = find_anchor(frame, alpha_map, hint_bbox, search_radius=search_radius)
        lefts.append(found[0])
        tops.append(found[1])
        # Re-extract score for logging (find_anchor logs it, but we want the value)
        import cv2

        frame_l = np.asarray(frame.convert("L"), dtype=np.uint8)
        template = np.clip(alpha_map * 255.0, 0, 255).astype(np.uint8)
        img_w, img_h = frame.size
        cat_left, cat_top = hint_bbox[0], hint_bbox[1]
        wl = max(0, cat_left - search_radius)
        wt = max(0, cat_top - search_radius)
        wr = min(img_w, cat_left + w + search_radius)
        wb = min(img_h, cat_top + h + search_radius)
        search_img = frame_l[wt:wb, wl:wr]
        result = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        scores.append(float(max_val))

    voted_left = int(np.median(lefts))
    voted_top = int(np.median(tops))
    log.info(
        "anchor_multiframe_voted",
        n_frames=len(frames),
        lefts=lefts,
        tops=tops,
        scores=[round(s, 4) for s in scores],
        voted_left=voted_left,
        voted_top=voted_top,
        mean_score=round(float(np.mean(scores)), 4),
    )
    return (voted_left, voted_top, voted_left + w, voted_top + h)


def calibrate_from_video(  # noqa: PLR0913
    video: Path,
    bbox: tuple[int, int, int, int],
    *,
    logo_rgb: tuple[int, int, int] = (255, 255, 255),
    sample_times: Iterable[float] | None = None,
    ffmpeg: str | None = None,
    min_alpha: float = 0.05,
    erode_pixels: int = 2,
) -> np.ndarray:
    """Derive an alpha map directly from the video (no external calibration needed).

    Procedure:
        1. Sample multiple frames of the video at the given times (default: 24
           evenly-spaced frames across the duration).
        2. For each frame, the 4 corners of *bbox* (a 96×96 square) are
           assumed to be alpha=0 (the ✦ watermark never reaches the corners).
           Their mean RGB is the original at that frame in that region.
        3. For every pixel in *bbox*, alpha = median over frames of
           `(wm - orig) / (logo - orig)` (per channel, then averaged).

    Args:
        video: path to the source video.
        bbox: `(left, top, right, bottom)` matching the catalog position of
            the watermark.
        logo_rgb: per-channel logo color (default pure white).
        sample_times: optional explicit timestamps (seconds). When None,
            24 frames evenly spaced across the duration are used.
        ffmpeg: optional explicit ffmpeg path.

    Returns:
        A `(H, W)` float32 alpha map, where H×W matches *bbox*'s size.
    """
    import shutil
    import subprocess
    import tempfile

    binary = ffmpeg or shutil.which("ffmpeg")
    if binary is None:
        msg = "ffmpeg binary not found on PATH"
        raise RuntimeError(msg)

    left, top, right, bottom = bbox
    h, w = bottom - top, right - left

    if sample_times is None:
        # Probe duration, then sample 24 frames evenly
        ffprobe_bin = shutil.which("ffprobe")
        if ffprobe_bin is None:
            msg = "ffprobe not found; cannot auto-detect duration"
            raise RuntimeError(msg)
        probe = subprocess.run(  # noqa: S603
            [ffprobe_bin, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
            check=True,
            capture_output=True,
            text=True,
        )
        duration = float(probe.stdout.strip())
        sample_times = [duration * i / 24 for i in range(24)]

    region_stack: list[np.ndarray] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, t in enumerate(sample_times):
            out_png = Path(tmpdir) / f"f_{i:03d}.png"
            cmd: list[str] = [
                binary,
                "-y",
                "-ss",
                f"{t:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-update",
                "1",
                str(out_png),
            ]
            subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603
            frame = np.array(Image.open(out_png).convert("RGB"), dtype=np.float32)
            region_stack.append(frame[top:bottom, left:right, :])
    regions = np.stack(region_stack)  # (T, h, w, 3)
    t_count = regions.shape[0]

    logo = np.asarray(logo_rgb, dtype=np.float32)

    # For each frame, estimate orig from the 4 corners of the bbox.
    def _corner_orig(region: np.ndarray) -> np.ndarray:
        corners: np.ndarray = np.median(
            [
                region[0:8, 0:8, :].mean(axis=(0, 1)),
                region[0:8, w - 8 : w, :].mean(axis=(0, 1)),
                region[h - 8 : h, 0:8, :].mean(axis=(0, 1)),
                region[h - 8 : h, w - 8 : w, :].mean(axis=(0, 1)),
            ],
            axis=0,
        )
        return corners

    # Per-frame alpha estimate, then median over frames.
    alpha_accum: list[np.ndarray] = []
    for t_idx in range(t_count):
        region = regions[t_idx]
        orig = _corner_orig(region)  # (3,)
        # alpha[i,j] = (wm - orig) / (logo - orig), per channel
        denom = (logo - orig)[None, None, :]
        # Guard against divide-by-zero when orig ≈ logo at the corners
        denom = np.where(np.abs(denom) < 1.0, 1.0, denom)
        alpha_t = (region - orig[None, None, :]) / denom
        alpha_t = np.clip(alpha_t, 0.0, 1.0)
        # Average the 3 channels (the alpha should be channel-invariant for
        # a white logo)
        alpha_accum.append(alpha_t.mean(axis=2))
    alpha_stack: np.ndarray = np.stack(alpha_accum)
    alpha: np.ndarray = np.median(alpha_stack, axis=0)
    # Threshold low-alpha noise to zero. Per-pixel median estimation tends to
    # leave residual alpha at the ✦'s edges; thresholding erodes those edges
    # so reverse blending doesn't subtract noise from non-watermark pixels.
    if min_alpha > 0:
        alpha = np.where(alpha < min_alpha, 0.0, alpha)
    # Morphological erosion: removes any remaining fuzzy edge by N pixels,
    # leaving only the high-confidence core of the watermark.
    if erode_pixels > 0:
        import cv2

        mask = (alpha * 255).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_pixels * 2 + 1,) * 2)
        mask = cv2.erode(mask, kernel, iterations=1)
        alpha = mask.astype(np.float32) / 255.0
    log.info(
        "alpha_self_calibrated",
        video=str(video),
        bbox=bbox,
        frames=t_count,
        alpha_max=float(alpha.max()),
        alpha_mean=float(alpha.mean()),
        alpha_nonzero=int((alpha > 0).sum()),
        min_alpha_threshold=min_alpha,
        erode_pixels=erode_pixels,
    )
    return alpha.astype(np.float32)


def calibrate_from_solid_video(  # noqa: PLR0913, PLR0915
    video: Path,
    background_rgb: tuple[int, int, int],
    *,
    logo_rgb: tuple[int, int, int] = (255, 255, 255),
    alpha_size: int = 96,
    detect_threshold: float = 2.0,
    ffmpeg: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Calibrate alpha map from a solid-background video (e.g. grey.mp4, black.mp4).

    Procedure:
        1. Extract all frames from *video* and compute the per-pixel median.
           The median frame averages out H.264 compression noise, giving a clean
           watermark view.
        2. Auto-detect the watermark bbox: find pixels whose luminance deviates
           from *background_rgb* by more than *detect_threshold*, compute the
           tight bounding box of those pixels, then expand evenly to *alpha_size*
           × *alpha_size* centered on the tight cluster.
        3. For each pixel in the bbox, derive alpha per channel:
              α_c = (wm_c - bg_c) / (logo_c - bg_c)
           then average across channels.
        4. Clip to [0, 1] and set alpha=0 where (1-alpha) would underflow.

    Args:
        video: path to a solid-background video with the watermark visible.
        background_rgb: the (R, G, B) of the solid background (e.g. ~126 for grey).
        logo_rgb: per-channel logo color (default pure white).
        alpha_size: target alpha map size (default 96, matching Gemini catalog).
        detect_threshold: luminance deviation from background to consider as watermark.
        ffmpeg: optional explicit ffmpeg path.

    Returns:
        (alpha_map, meta) where:
          - alpha_map: (alpha_size, alpha_size) float32 array in [0, 1].
          - meta: dict with keys: background_rgb, logo_rgb, bbox, source,
            alpha_shape, alpha_max, alpha_mean, alpha_nonzero, created_at.
    """
    import shutil
    import subprocess
    import tempfile
    from datetime import datetime

    binary = ffmpeg or shutil.which("ffmpeg")
    if binary is None:
        msg = "ffmpeg binary not found on PATH"
        raise RuntimeError(msg)

    bg = np.array(background_rgb, dtype=np.float32)
    logo = np.array(logo_rgb, dtype=np.float32)

    # Check that logo != background per channel (otherwise denominator is 0).
    denom = logo - bg
    if np.any(np.abs(denom) < 1.0):
        msg = f"logo_rgb {logo_rgb} too close to background_rgb {background_rgb}; cannot derive alpha (division by ~0)"
        raise ValueError(msg)

    # --- Step 1: extract all frames, compute median ---
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        extract_cmd: list[str] = [
            binary,
            "-y",
            "-i",
            str(video),
            "-vsync",
            "0",
            str(tmp / "%04d.png"),
        ]
        subprocess.run(extract_cmd, check=True, capture_output=True)  # noqa: S603
        pngs = sorted(tmp.glob("*.png"))
        if not pngs:
            msg = f"no frames extracted from {video}"
            raise RuntimeError(msg)
        frames = [np.array(Image.open(p).convert("RGB"), dtype=np.float32) for p in pngs]
    stack = np.stack(frames, axis=0)  # (T, H, W, 3)
    median_frame: np.ndarray = np.median(stack, axis=0)  # (H, W, 3)
    h_full, w_full = median_frame.shape[:2]
    log.info(
        "solid_video_median_computed",
        video=str(video),
        frames=len(pngs),
        median_mean=float(median_frame.mean()),
        median_std=float(median_frame.std()),
    )

    # --- Step 2: auto-detect watermark bbox ---
    # Compute luminance and per-pixel deviation from background luminance.
    bg_luminance = float(bg.mean())
    frame_luminance: np.ndarray = median_frame.mean(axis=2)  # (H, W)
    dev_map = np.abs(frame_luminance - bg_luminance)
    wm_mask = dev_map > detect_threshold
    dys, dxs = np.where(wm_mask)
    if len(dys) == 0:
        msg = f"no watermark pixels detected at threshold {detect_threshold}; try lowering it"
        raise RuntimeError(msg)

    tight_left = int(dxs.min())
    tight_right = int(dxs.max())
    tight_top = int(dys.min())
    tight_bottom = int(dys.max())
    tight_w = tight_right - tight_left + 1
    tight_h = tight_bottom - tight_top + 1
    log.info(
        "solid_video_watermark_detected",
        tight_bbox=(tight_left, tight_top, tight_right + 1, tight_bottom + 1),
        tight_size=(tight_w, tight_h),
        wm_pixels=len(dys),
    )

    # Expand to alpha_size × alpha_size centered on the tight cluster.
    cx = (tight_left + tight_right) // 2
    cy = (tight_top + tight_bottom) // 2
    half = alpha_size // 2
    left = cx - half
    top = cy - half
    right = left + alpha_size
    bottom = top + alpha_size

    # Clamp to frame bounds (shouldn't happen for normal placements, but defensive).
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > w_full:
        left -= right - w_full
        right = w_full
    if bottom > h_full:
        top -= bottom - h_full
        bottom = h_full
    bbox = (left, top, right, bottom)
    log.info("solid_video_bbox_expanded", bbox=bbox, alpha_size=alpha_size)

    # --- Step 3: derive alpha per pixel from median frame ---
    region = median_frame[top:bottom, left:right, :]  # (H_a, W_a, 3)
    # α_c = (wm_c - bg_c) / (logo_c - bg_c)  per channel
    alpha_per_channel = (region - bg[None, None, :]) / denom[None, None, :]
    alpha_per_channel = np.clip(alpha_per_channel, 0.0, 1.0)
    # Average across channels (watermark is white, α should be channel-invariant).
    alpha = alpha_per_channel.mean(axis=2).astype(np.float32)  # (H_a, W_a)

    # Threshold near-zero alpha to zero to avoid tiny numerical noise triggering
    # (1-alpha) underflow in reverse blending.
    alpha = np.where(alpha < _ALPHA_ZERO_THRESHOLD, 0.0, alpha)

    meta = {
        "background_rgb": list(background_rgb),
        "logo_rgb": list(logo_rgb),
        "bbox": list(bbox),
        "source": str(video.name) if isinstance(video, Path) else str(video),
        "alpha_shape": list(alpha.shape),
        "alpha_max": float(alpha.max()),
        "alpha_mean": float(alpha.mean()),
        "alpha_nonzero": int((alpha > 0).sum()),
        "created_at": datetime.now(UTC).isoformat(),
    }
    log.info(
        "alpha_solid_calibrated",
        video=str(video),
        bbox=bbox,
        background_rgb=background_rgb,
        alpha_max=meta["alpha_max"],
        alpha_mean=meta["alpha_mean"],
        alpha_nonzero=meta["alpha_nonzero"],
    )
    return alpha, meta
