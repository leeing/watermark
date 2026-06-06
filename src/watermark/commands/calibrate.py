"""Click command: `watermark calibrate` — calibrate alpha map from video.

Two modes:
  1. **Solid-background** (--background-rgb): fast, reliable calibration from a
     solid-color video (e.g. grey.mp4, black.mp4). Uses the median frame to
     derive alpha = (wm - bg) / (logo - bg) per pixel.
  2. **Self-calibration** (default): estimates alpha from the video itself by
     exploiting corner pixels as alpha=0 control points. Less reliable for
     complex backgrounds.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import structlog

from watermark.config import settings
from watermark.consts import ALPHA_BBOX_1080X1920
from watermark.services import alpha as alpha_svc

log = structlog.get_logger()


def _parse_rgb(raw: str) -> tuple[int, int, int]:
    """Parse an R,G,B string into a 3-tuple of ints."""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:  # noqa: PLR2004
        msg = f"--background-rgb expects 3 comma-separated ints (R,G,B), got {len(parts)}"
        raise click.BadParameter(msg)
    try:
        r, g, b = (int(p) for p in parts)
    except ValueError as exc:
        msg = f"--background-rgb contains a non-integer: {raw!r}"
        raise click.BadParameter(msg) from exc
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):  # noqa: PLR2004
        msg = f"--background-rgb values must be 0-255, got {raw!r}"
        raise click.BadParameter(msg)
    return r, g, b


def _parse_bbox(raw: str) -> tuple[int, int, int, int]:
    """Parse left,top,right,bottom string into a 4-tuple."""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:  # noqa: PLR2004
        msg = f"--bbox expects 4 comma-separated ints, got {len(parts)}: {raw!r}"
        raise click.BadParameter(msg)
    try:
        left, top, right, bottom = (int(p) for p in parts)
    except ValueError as exc:
        msg = f"--bbox contains a non-integer: {raw!r}"
        raise click.BadParameter(msg) from exc
    if right <= left or bottom <= top:
        msg = f"--bbox must satisfy right>left and bottom>top, got {raw!r}"
        raise click.BadParameter(msg)
    return left, top, right, bottom


@click.command()
@click.option(
    "--video",
    default=settings.source_video,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Source video to calibrate from.",
)
@click.option(
    "--bbox",
    default=f"{ALPHA_BBOX_1080X1920[0]},{ALPHA_BBOX_1080X1920[1]},{ALPHA_BBOX_1080X1920[2]},{ALPHA_BBOX_1080X1920[3]}",
    show_default=True,
    help="Watermark bbox as left,top,right,bottom. For solid mode, set to 'auto' to auto-detect.",
)
@click.option(
    "--background-rgb",
    default=None,
    help="Solid background RGB as R,G,B (e.g. 126,126,127 for grey.mp4). Enables solid-background mode.",
)
@click.option(
    "--out-alpha",
    default=settings.project_root / "data" / "gemini_alpha_96_from_grey.npy",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output path for the calibrated alpha map (.npy).",
)
@click.option(
    "--out-meta",
    default=settings.project_root / "data" / "gemini_alpha_meta.json",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output path for calibration metadata (.json).",
)
@click.option(
    "--min-alpha",
    default=0.05,
    show_default=True,
    type=float,
    help="(self-calibration only) Threshold below which alpha is set to 0.",
)
@click.option(
    "--erode-pixels",
    default=2,
    show_default=True,
    type=int,
    help="(self-calibration only) Morphological erosion radius in pixels.",
)
def calibrate_cmd(  # noqa: PLR0913
    video: Path,
    bbox: str,
    background_rgb: str | None,
    out_alpha: Path,
    out_meta: Path,
    min_alpha: float,
    erode_pixels: int,
) -> None:
    """Calibrate an alpha map from VIDEO.

    Two modes:
      --background-rgb R,G,B : Solid-background mode (recommended). Uses a video
        with a known uniform background color (e.g. grey.mp4 with R=126,G=126,B=127).
        Auto-detects bbox if --bbox is 'auto'.
      (default): Self-calibration mode. Estimates alpha from the video itself by
        assuming the 4 bbox corners reveal the original background.
    """
    if background_rgb is not None:
        # --- Solid-background mode ---
        bg_rgb = _parse_rgb(background_rgb)
        log.info("calibrate_solid_start", video=str(video), background_rgb=bg_rgb)

        if bbox == "auto":
            # Auto-detect bbox from the solid video; pass a dummy catalog bbox
            # that won't be used (calibrate_from_solid_video does its own detection).
            bbox_tuple = ALPHA_BBOX_1080X1920
            auto_bbox = True
        else:
            bbox_tuple = _parse_bbox(bbox)
            auto_bbox = False

        alpha, meta = alpha_svc.calibrate_from_solid_video(video, bg_rgb)

        # If bbox was auto-detected, update meta with the detected bbox.
        if auto_bbox:
            log.info("calibrate_auto_bbox", detected_bbox=meta["bbox"])

        out_alpha.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_alpha, alpha)
        with open(out_meta, "w") as f:
            json.dump(meta, f, indent=2)
        click.echo(
            f"OK (solid): alpha={out_alpha} (shape={alpha.shape}, max={alpha.max():.4f}, "
            f"mean={alpha.mean():.4f}), meta={out_meta}"
        )
        log.info("calibrate_solid_done", path=str(out_alpha), alpha_max=float(alpha.max()))
    else:
        # --- Self-calibration mode ---
        bbox_tuple = _parse_bbox(bbox)
        log.info("calibrate_self_start", video=str(video), bbox=bbox_tuple)

        alpha = alpha_svc.calibrate_from_video(video, bbox_tuple, min_alpha=min_alpha, erode_pixels=erode_pixels)
        out_alpha.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_alpha, alpha)
        click.echo(f"OK (self-cal): {out_alpha} (shape={alpha.shape}, max={alpha.max():.4f}, mean={alpha.mean():.4f})")
        log.info("calibrate_self_done", path=str(out_alpha), alpha_max=float(alpha.max()))
