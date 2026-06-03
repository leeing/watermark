"""Click command: `watermark remove` — end-to-end inpaint + reassemble + verify."""

from __future__ import annotations

import sys

import click
import structlog

from watermark.config import settings
from watermark.consts import DILATE_PX, WATERMARK_BBOX
from watermark.services import detect as detect_svc
from watermark.services import inpaint as inpaint_svc
from watermark.services import mask as mask_svc
from watermark.services import reassemble as reassemble_svc
from watermark.services import verify as verify_svc
from watermark.services.mask import MaskMode

log = structlog.get_logger()

# A bbox is a 4-tuple of (left, top, right, bottom) integer pixel coordinates.
_BBOX_ARITY = 4


def _parse_bbox(raw: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != _BBOX_ARITY:
        msg = f"--bbox expects {_BBOX_ARITY} comma-separated ints, got {len(parts)}: {raw!r}"
        raise click.BadParameter(msg)
    try:
        values = tuple(int(p) for p in parts)
    except ValueError as exc:
        msg = f"--bbox contains a non-integer: {raw!r}"
        raise click.BadParameter(msg) from exc
    left, top, right, bottom = values
    if right <= left or bottom <= top:
        msg = f"--bbox must satisfy right>left and bottom>top, got {raw!r}"
        raise click.BadParameter(msg)
    return left, top, right, bottom


def _parse_int_option(raw: object, name: str) -> int:
    """Parse a Click option value known to be an integer."""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError as exc:
            msg = f"{name} contains a non-integer: {raw!r}"
            raise click.BadParameter(msg) from exc
    msg = f"{name} contains an unsupported value: {raw!r}"
    raise click.BadParameter(msg)


def _parse_float_option(raw: object, name: str) -> float:
    """Parse a Click option value known to be a float."""
    if isinstance(raw, int | float):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError as exc:
            msg = f"{name} contains a non-number: {raw!r}"
            raise click.BadParameter(msg) from exc
    msg = f"{name} contains an unsupported value: {raw!r}"
    raise click.BadParameter(msg)


@click.command()
@click.option(
    "--device",
    type=click.Choice(["mps", "cpu", "cuda"]),
    default=settings.device,
    show_default=True,
    help="torch device for LaMa inference.",
)
@click.option(
    "--bbox",
    default=f"{WATERMARK_BBOX[0]},{WATERMARK_BBOX[1]},{WATERMARK_BBOX[2]},{WATERMARK_BBOX[3]}",
    show_default=True,
    help="Watermark bbox as left,top,right,bottom in pixels.",
)
@click.option(
    "--dilate",
    default=DILATE_PX,
    show_default=True,
    type=int,
    help="Mask dilation in pixels (expands bbox outward).",
)
@click.option(
    "--backend",
    type=click.Choice(["telea", "ns", "lama"]),
    default="telea",
    show_default=True,
    help="Inpainting backend: telea/ns (OpenCV) or lama (slower, often better single-frame quality).",
)
@click.option(
    "--mask-mode",
    type=click.Choice(["watermark", "rectangle"]),
    default="watermark",
    show_default=True,
    help="Mask generation mode: refined watermark symbol or full rectangle.",
)
@click.option(
    "--radius",
    default=inpaint_svc.TELEA_RADIUS,
    show_default=True,
    type=int,
    help="OpenCV inpainting radius in pixels.",
)
@click.option(
    "--texture-strength",
    default=inpaint_svc.InpaintOptions().texture_strength,
    show_default=True,
    type=float,
    help="Local texture restoration strength; set 0 to disable.",
)
@click.option(
    "--feather",
    default=inpaint_svc.InpaintOptions().feather,
    show_default=True,
    type=int,
    help="Mask edge feather in pixels for repaired-region blending.",
)
@click.option(
    "--skip-verify",
    is_flag=True,
    default=False,
    help="Skip the SPEC acceptance verification step (faster iteration).",
)
def remove_cmd(**raw_options: object) -> None:
    """One-shot: detect -> mask -> inpaint -> reassemble -> verify."""
    device = str(raw_options["device"])
    bbox = str(raw_options["bbox"])
    dilate = _parse_int_option(raw_options["dilate"], "--dilate")
    backend = str(raw_options["backend"])
    radius = _parse_int_option(raw_options["radius"], "--radius")
    texture_strength = _parse_float_option(raw_options["texture_strength"], "--texture-strength")
    feather = _parse_int_option(raw_options["feather"], "--feather")
    skip_verify = bool(raw_options["skip_verify"])
    raw_mask_mode = str(raw_options["mask_mode"])
    mask_mode: MaskMode = "watermark" if raw_mask_mode == "watermark" else "rectangle"

    try:
        bbox_tuple = _parse_bbox(bbox)
    except click.BadParameter as exc:
        click.echo(str(exc), err=True)
        sys.exit(2)

    log.info(
        "remove_start",
        device=device,
        bbox=bbox_tuple,
        dilate=dilate,
        backend=backend,
        mask_mode=mask_mode,
        radius=radius,
        texture_strength=texture_strength,
        feather=feather,
    )

    try:
        # 1. detect (extract first frame + verify bbox catches the watermark)
        detect_svc.detect(settings.source_video, settings.frames_dir, bbox_tuple)

        # 2. mask
        mask_svc.build(
            frame_path=settings.frames_dir / "frame_0000.png",
            bbox=bbox_tuple,
            dilate=dilate,
            outputs=mask_svc.MaskOutputs(settings.mask_path, settings.frames_dir / "mask_overlay.png"),
            mode=mask_mode,
        )

        # 3. extract all frames
        reassemble_svc.extract_frames(
            settings.source_video,
            settings.frames_in_dir,
        )

        # 4. inpaint
        inpaint_svc.inpaint_frames(
            settings.frames_in_dir,
            settings.mask_path,
            settings.frames_out_dir,
            inpaint_svc.InpaintOptions(
                device=device,
                backend=backend,
                radius=radius,
                texture_strength=texture_strength,
                feather=feather,
            ),
        )

        # 5. reassemble
        reassemble_svc.reassemble(
            settings.frames_out_dir,
            settings.source_video,
            settings.output_video,
        )

        # 6. verify
        if not skip_verify:
            verify_svc.verify_output(settings.output_video)
            verify_svc.verify_audio_identical(settings.source_video, settings.output_video)
    except (verify_svc.VerificationError, ValueError, RuntimeError) as exc:
        click.echo(f"FAILED: {exc}", err=True)
        sys.exit(1)

    click.echo(f"OK: {settings.output_video}")
    log.info("remove_done", out=str(settings.output_video))
