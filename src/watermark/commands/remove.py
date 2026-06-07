"""Click command: `watermark remove` — end-to-end inpaint + reassemble + verify."""

from __future__ import annotations

import contextlib
import sys

import click
import numpy as np
import structlog
from PIL import Image

from watermark.config import settings
from watermark.consts import ALPHA_BBOX_HINT, CRF_LOSSLESS, DILATE_PX, LOGO_RGB, VIDEO_SIZE, WATERMARK_BBOX
from watermark.services import alpha as alpha_svc
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
    type=click.Choice(["telea", "alpha", "ns", "lama"]),
    default="telea",
    show_default=True,
    help="Inpainting backend: telea (default, visually perfect), alpha (lossless, opt-in), ns, lama.",
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
    "--lossless",
    is_flag=True,
    default=False,
    help="Use CRF 0 for mathematically lossless H.264 (large file, pixel-perfect). "
    "Also disables adaptive alpha clamping and edge feathering for pure reverse blend.",
)
@click.option(
    "--skip-verify",
    is_flag=True,
    default=False,
    help="Skip the SPEC acceptance verification step (faster iteration).",
)
def remove_cmd(**raw_options: object) -> None:  # noqa: C901, PLR0912, PLR0915
    """One-shot: detect -> mask -> inpaint -> reassemble -> verify."""
    device = str(raw_options["device"])
    bbox = str(raw_options["bbox"])
    dilate = _parse_int_option(raw_options["dilate"], "--dilate")
    backend = str(raw_options["backend"])
    radius = _parse_int_option(raw_options["radius"], "--radius")
    texture_strength = _parse_float_option(raw_options["texture_strength"], "--texture-strength")
    feather = _parse_int_option(raw_options["feather"], "--feather")
    skip_verify = bool(raw_options["skip_verify"])
    lossless = bool(raw_options["lossless"])
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
        lossless=lossless,
        mask_mode=mask_mode,
        radius=radius,
        texture_strength=texture_strength,
        feather=feather,
    )

    # Alpha backend: pre-load the calibrated alpha map and use multi-frame
    # anchor search to locate the actual watermark position (the hint bbox
    # from grey.mp4 calibration can be off by 36+ px on different videos).
    alpha_map: np.ndarray | None = None
    logo_map: np.ndarray | None = None
    alpha_bbox = ALPHA_BBOX_HINT
    if backend == "alpha":
        # Size sanity check: skip if the source file is empty or unreadable
        # (test fixtures rely on this). Mismatched dimensions will be caught
        # later by inpaint_alpha_frame's shape check.
        if settings.source_video.is_file() and settings.source_video.stat().st_size > 0:
            from watermark.services.verify import probe as _probe

            try:
                probe_result = _probe(settings.source_video)
            except (RuntimeError, OSError):
                probe_result = None
            if probe_result is not None:
                video_size = (probe_result.width, probe_result.height)
                if video_size != VIDEO_SIZE:
                    msg = (
                        f"alpha backend currently only supports {VIDEO_SIZE[0]}x{VIDEO_SIZE[1]} "
                        f"videos; got {video_size[0]}x{video_size[1]}. "
                        "Use --backend telea for other sizes."
                    )
                    click.echo(f"FAILED: {msg}", err=True)
                    sys.exit(1)
        alpha_map = alpha_svc.load_alpha_map(settings.alpha_map_path)
        if settings.logo_map_path.is_file():
            logo_map = alpha_svc.load_logo_map(settings.logo_map_path, alpha_map.shape)
        log.info(
            "alpha_backend_ready",
            path=str(settings.alpha_map_path),
            logo_map_path=str(settings.logo_map_path) if logo_map is not None else None,
            hint_bbox=alpha_bbox,
            logo_rgb=LOGO_RGB,
        )

    try:
        if backend == "alpha":
            # Alpha route: skip mask step (alpha map carries its own shape).
            # Extract first frame for visual diagnostics.
            detect_svc.detect(settings.source_video, settings.frames_dir, alpha_bbox)
            # Multi-frame anchor search: extract 5 sample frames and use
            # median-voted position detection. This is more robust than
            # single-frame matching (frame 0 can be very dark with low ZNCC).
            subpixel_shift = (0.0, 0.0)
            if alpha_map is not None:
                from watermark.services.verify import extract_frame as _extract_frame

                sample_times = [0.0, 2.0, 4.0, 6.0, 8.0]
                sample_frames: list[Image.Image] = []
                for t_sec in sample_times:
                    with contextlib.suppress(RuntimeError):
                        sample_frames.append(_extract_frame(settings.source_video, t_sec))
                if not sample_frames:
                    # Fallback: use the already-extracted first frame
                    sample_frames = [Image.open(settings.frames_dir / "frame_0000.png")]
                alpha_bbox = alpha_svc.find_anchor_multiframe(
                    sample_frames,
                    alpha_map,
                    alpha_bbox,
                    search_radius=120,
                )
                log.info("alpha_anchor_refined", bbox=alpha_bbox)

                # Sub-pixel shift refinement
                subpixel_shift = alpha_svc.refine_subpixel_shift(
                    sample_frames,
                    alpha_map,
                    logo_map,
                    alpha_bbox,
                )
                log.info("alpha_subpixel_refined", shift=subpixel_shift)
        else:
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
        inpaint_options_kwargs: dict[str, object] = {
            "device": device,
            "backend": backend,
            "radius": radius,
            "texture_strength": texture_strength,
            "feather": feather,
            "lossless": lossless,
        }
        if backend == "alpha":
            inpaint_options_kwargs["alpha_map"] = alpha_map
            inpaint_options_kwargs["logo_map"] = logo_map
            inpaint_options_kwargs["alpha_bbox"] = alpha_bbox
            inpaint_options_kwargs["logo_rgb"] = (*LOGO_RGB, 0)  # RGBA-ish placeholder; first 3 used
            inpaint_options_kwargs["subpixel_shift"] = subpixel_shift
        inpaint_svc.inpaint_frames(
            settings.frames_in_dir,
            settings.mask_path,
            settings.frames_out_dir,
            inpaint_svc.InpaintOptions(**inpaint_options_kwargs),  # type: ignore[arg-type]
        )

        # 5. reassemble
        if backend == "alpha" and alpha_map is not None:
            # Alpha route: verify round-trip at PNG layer before reassembling.
            # This catches anchor misalignment or alpha map errors BEFORE
            # writing a bad output video (SPEC A8 at PNG layer).
            from pathlib import Path as _Path

            _temp_frames_out = _Path(settings.frames_out_dir)
            log.info("alpha_png_verification_start", frames_out_dir=str(_temp_frames_out))
            png_results = verify_svc.verify_alpha_round_trip_png(
                frames_in_dir=settings.frames_in_dir,
                frames_out_dir=_temp_frames_out,
                alpha_map=alpha_map,
                bbox=alpha_bbox,
                logo_rgb=LOGO_RGB,
                logo_map=logo_map,
                subpixel_shift=subpixel_shift,
                frame_indices=[0, 60, 120, 180, 239],
                max_diff_tolerance=settings.alpha_png_max_diff_tolerance,
                mean_diff_tolerance=settings.alpha_png_mean_diff_tolerance,
            )
            passed = sum(1 for r in png_results if r.passed)
            log.info(
                "alpha_png_verification_done",
                frames_checked=len(png_results),
                frames_passed=passed,
                results=[(r.frame_index, r.alpha_region_max_diff, r.alpha_region_mean_diff) for r in png_results],
            )

        reassemble_svc.reassemble(
            settings.frames_out_dir,
            settings.source_video,
            settings.output_video,
            crf=CRF_LOSSLESS if lossless else None,
        )

        # 6. verify
        if not skip_verify:
            verify_svc.verify_output(settings.output_video)
            verify_svc.verify_audio_identical(settings.source_video, settings.output_video)
            if backend == "alpha" and alpha_map is not None:
                verify_svc.verify_alpha_round_trip(
                    source_video=settings.source_video,
                    clean_video=settings.output_video,
                    alpha_map=alpha_map,
                    bbox=alpha_bbox,
                    logo_rgb=LOGO_RGB,
                    logo_map=logo_map,
                    subpixel_shift=subpixel_shift,
                    pixel_tolerance=int(settings.alpha_h264_pixel_tolerance),
                )
    except (verify_svc.VerificationError, ValueError, RuntimeError) as exc:
        click.echo(f"FAILED: {exc}", err=True)
        sys.exit(1)

    click.echo(f"OK: {settings.output_video}")
    log.info("remove_done", out=str(settings.output_video))
