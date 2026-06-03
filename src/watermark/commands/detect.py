"""Click command: `watermark detect <video>` — extract first frame & crop bbox."""

from __future__ import annotations

import sys
from pathlib import Path

import click
import structlog

from watermark.config import settings
from watermark.consts import WATERMARK_BBOX
from watermark.services import detect as detect_svc

log = structlog.get_logger()


@click.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def detect_cmd(video: Path) -> None:
    """Extract the first frame of VIDEO and crop the watermark region for human inspection."""
    try:
        crop_path = detect_svc.detect(video, settings.frames_dir, WATERMARK_BBOX)
    except detect_svc.FFmpegNotFoundError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    except detect_svc.FrameExtractionError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    except detect_svc.WatermarkNotFoundError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    except ValueError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    click.echo(str(crop_path))
    log.info("detect_done", crop=str(crop_path))
