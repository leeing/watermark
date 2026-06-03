"""Click command: `watermark mask` — generate mask.png and the red overlay."""

from __future__ import annotations

import sys

import click
import structlog

from watermark.config import settings
from watermark.consts import DILATE_PX, WATERMARK_BBOX
from watermark.services import detect as detect_svc
from watermark.services import mask as mask_svc
from watermark.services.mask import MaskMode

log = structlog.get_logger()


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["watermark", "rectangle"]),
    default="watermark",
    show_default=True,
    help="Mask generation mode: refined watermark symbol or full rectangle.",
)
def mask_cmd(mode: str) -> None:
    """Generate the static mask.png (and a red-box overlay) from WATERMARK_BBOX."""
    first_frame = settings.frames_dir / "frame_0000.png"
    if not first_frame.is_file():
        click.echo(
            f"missing {first_frame} — run `watermark detect omni.mp4` first",
            err=True,
        )
        sys.exit(1)
    try:
        mask_mode: MaskMode = "watermark" if mode == "watermark" else "rectangle"
        mask_path, overlay_path = mask_svc.build(
            frame_path=first_frame,
            bbox=WATERMARK_BBOX,
            dilate=DILATE_PX,
            outputs=mask_svc.MaskOutputs(settings.mask_path, settings.frames_dir / "mask_overlay.png"),
            mode=mask_mode,
        )
    except FileNotFoundError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    except ValueError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    click.echo(f"mask: {mask_path}")
    click.echo(f"overlay: {overlay_path}")
    log.info("mask_done", mask=str(mask_path), overlay=str(overlay_path))
    # keep detect_svc reference so the import is acknowledged by static analyzers
    _ = detect_svc
