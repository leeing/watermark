"""Service: per-frame inpainting.

Uses OpenCV's Telea algorithm (`cv2.inpaint`) which is well-suited for small
static watermarks on relatively low-frequency backgrounds. It runs in tens of
milliseconds per frame even at 1080×1920, and gives clean results for the
small static corner-mark case.

A LaMa backend (via `simple_lama_inpainting`) is also supported via the
`backend="lama"` argument, but is much slower on CPU (~5 frames/min) and
requires extra model downloads.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import structlog
from PIL import Image

from watermark.services import alpha as alpha_svc

log = structlog.get_logger()

# Pattern that matches the ffmpeg-style %04d.png naming used by the extract step.
_FRAME_PATTERN = re.compile(r"^(\d+)\.png$")

# OpenCV Telea inpainting radius (pixels). 3 is the standard default and works
# well for our small static watermarks.
TELEA_RADIUS = 3


@dataclass(frozen=True)
class InpaintOptions:
    """Options controlling per-frame inpainting."""

    device: str = "cpu"
    backend: str = "telea"
    radius: int = TELEA_RADIUS
    texture_strength: float = 0.25
    feather: int = 2
    lossless: bool = False  # skip clamping + feathering for mathematically pure reverse
    # Reverse alpha blending (Lossless.md) — only used when backend == "alpha"
    alpha_map: np.ndarray | None = None
    logo_map: np.ndarray | None = None
    alpha_bbox: tuple[int, int, int, int] | None = None
    logo_rgb: tuple[int, int, int, int] = (255, 255, 255, 255)
    subpixel_shift: tuple[float, float] = (0.0, 0.0)


class Inpainter(Protocol):
    """Callable that repairs one frame with a mask."""

    def __call__(self, frame_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
        """Return an inpainted RGB frame."""


def _sorted_frames(frames_dir: Path) -> list[Path]:
    """Return frame files in frames_dir sorted by their numeric stem."""
    pairs: list[tuple[int, Path]] = []
    for entry in frames_dir.iterdir():
        if not entry.is_file():
            continue
        match = _FRAME_PATTERN.match(entry.name)
        if match is None:
            continue
        pairs.append((int(match.group(1)), entry))
    pairs.sort(key=lambda p: p[0])
    return [path for _, path in pairs]


def _inpaint_opencv(
    frame_rgb: Image.Image,
    mask_l: Image.Image,
    *,
    algorithm: str = "telea",
    options: InpaintOptions | None = None,
) -> Image.Image:
    """Inpaint using an OpenCV algorithm. Returns a new PIL Image (RGB)."""
    import cv2
    import numpy as np

    resolved = options or InpaintOptions()
    algorithms = {
        "telea": cv2.INPAINT_TELEA,
        "ns": cv2.INPAINT_NS,
    }
    if algorithm not in algorithms:
        msg = f"unsupported OpenCV inpaint algorithm {algorithm!r}; choose from {sorted(algorithms)}"
        raise ValueError(msg)
    if resolved.radius <= 0:
        msg = f"inpaint radius must be positive, got {resolved.radius}"
        raise ValueError(msg)

    frame_np = np.array(frame_rgb)
    mask_np = np.array(mask_l)
    # Telea requires uint8 mask with non-zero pixels = region to inpaint
    if mask_np.dtype != "uint8":
        mask_np = mask_np.astype("uint8")
    result_bgr = cv2.inpaint(frame_np, mask_np, resolved.radius, algorithms[algorithm])
    repaired = Image.fromarray(result_bgr)
    return _restore_local_texture(
        frame_rgb,
        repaired,
        mask_l,
        strength=resolved.texture_strength,
        feather=resolved.feather,
    )


def _restore_local_texture(
    original: Image.Image,
    repaired: Image.Image,
    mask_l: Image.Image,
    *,
    strength: float,
    feather: int,
) -> Image.Image:
    """Blend repaired pixels with locally inpainted high-frequency texture."""
    import cv2
    import numpy as np

    if strength <= 0:
        return repaired

    original_np = np.array(original.convert("RGB")).astype("float32")
    repaired_np = np.array(repaired.convert("RGB")).astype("float32")
    mask_np = np.array(mask_l.convert("L"))
    binary_mask = (mask_np > 0).astype("uint8") * 255
    if int(binary_mask.max()) == 0:
        return repaired

    low_frequency = cv2.GaussianBlur(original_np, (0, 0), sigmaX=2.0)
    residual = original_np - low_frequency
    residual_u8 = np.clip(residual + 128.0, 0, 255).astype("uint8")
    filled_residual = cv2.inpaint(residual_u8, binary_mask, 3, cv2.INPAINT_NS).astype("float32") - 128.0
    mask_bool = binary_mask > 0
    outside_residual = residual[~mask_bool]
    inside_filled = filled_residual[mask_bool]
    if outside_residual.size > 0 and inside_filled.size > 0:
        residual_too_smooth = float(inside_filled.std()) < float(outside_residual.std()) * 0.25
    else:
        residual_too_smooth = False
    if residual_too_smooth:
        ys, xs = np.where(mask_bool)
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        patch_width = x1 - x0
        source_x0 = x0 - patch_width if x0 >= patch_width else min(x1, residual.shape[1] - patch_width)
        source_x1 = source_x0 + patch_width
        borrowed = residual[y0:y1, source_x0:source_x1, :]
        if borrowed.shape[:2] != (y1 - y0, patch_width):
            borrowed = cv2.resize(borrowed, (patch_width, y1 - y0), interpolation=cv2.INTER_LINEAR)
        filled_residual[y0:y1, x0:x1, :] = np.where(
            mask_bool[y0:y1, x0:x1, np.newaxis],
            borrowed,
            filled_residual[y0:y1, x0:x1, :],
        )

    textured = repaired_np + filled_residual * strength
    if feather > 0:
        distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
        alpha = np.clip(distance / float(feather), 0.0, 1.0)
    else:
        alpha = (binary_mask > 0).astype("float32")
    alpha_3d = alpha[:, :, np.newaxis]

    blended = original_np * (1.0 - alpha_3d) + textured * alpha_3d
    final = np.where(binary_mask[:, :, np.newaxis] > 0, blended, original_np)
    return Image.fromarray(np.clip(final, 0, 255).astype("uint8"))


def _make_lama_inpainter(device: str) -> Inpainter:
    """Return a LaMa inpainter, loading the model once for the whole run."""
    from simple_lama_inpainting import SimpleLama

    model = SimpleLama()
    inner = getattr(model, "model", model)
    inner.to(device)

    def inpaint(frame_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
        result = model(frame_rgb, mask_l)
        return result  # type: ignore[no-any-return]

    return inpaint


def _inpaint_telea(frame_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    """Compatibility wrapper for tests and external callers."""
    return _inpaint_opencv(frame_rgb, mask_l, algorithm="telea", options=InpaintOptions(radius=TELEA_RADIUS))


def _inpaint_lama(frame_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    """Compatibility wrapper for one-off LaMa calls."""
    return _make_lama_inpainter("cpu")(frame_rgb, mask_l)


def _inpaint_alpha(
    frame_rgb: Image.Image,
    mask_l: Image.Image,  # noqa: ARG001 - unused, signature uniform with other backends
    options: InpaintOptions,
) -> Image.Image:
    """Reverse alpha blending — mathematically lossless (Lossless.md §2).

    Uses adaptive_reverse_blend which handles dark backgrounds without
    producing black spots (numerical underflow).
    """
    if options.alpha_map is None or options.alpha_bbox is None:
        msg = "backend='alpha' requires alpha_map and alpha_bbox in InpaintOptions"
        raise ValueError(msg)
    return alpha_svc.adaptive_reverse_blend(
        frame_rgb,
        options.alpha_map,
        options.alpha_bbox,
        (options.logo_rgb[0], options.logo_rgb[1], options.logo_rgb[2]),
        logo_map=options.logo_map,
        lossless=options.lossless,
        subpixel_shift=options.subpixel_shift,
    )


def _make_alpha_inpainter(options: InpaintOptions) -> Inpainter:
    def inpaint(frame_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
        return _inpaint_alpha(frame_rgb, mask_l, options)

    return inpaint


_INPAINTERS = {
    "telea": _inpaint_telea,
    "lama": _inpaint_lama,
}


def _make_opencv_inpainter(options: InpaintOptions, algorithm: str) -> Inpainter:
    """Return an OpenCV inpainter bound to one algorithm and radius."""

    def inpaint(frame_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
        return _inpaint_opencv(
            frame_rgb,
            mask_l,
            algorithm=algorithm,
            options=options,
        )

    return inpaint


def inpaint_frames(
    frames_in: Path,
    mask_path: Path,
    frames_out: Path,
    options: InpaintOptions | None = None,
) -> list[Path]:
    """Inpaint every frame in *frames_in* with a static mask, write to *frames_out*.

    Returns the list of written output paths in frame order.
    """
    resolved = options or InpaintOptions()
    backend = resolved.backend
    supported = {"telea", "ns", "lama", "alpha"}
    if backend not in supported and backend not in _INPAINTERS:
        msg = f"unsupported backend {backend!r}; choose from {sorted(supported | set(_INPAINTERS))}"
        raise ValueError(msg)
    if backend == "ns":
        inpaint_fn = _make_opencv_inpainter(resolved, "ns")
    elif backend == "telea" and _INPAINTERS.get("telea") is _inpaint_telea:
        inpaint_fn = _make_opencv_inpainter(resolved, "telea")
    elif backend == "lama" and _INPAINTERS.get("lama") is _inpaint_lama:
        inpaint_fn = _make_lama_inpainter(resolved.device)
    elif backend == "alpha":
        inpaint_fn = _make_alpha_inpainter(resolved)
    else:
        inpaint_fn = _INPAINTERS[backend]

    frames_out.mkdir(parents=True, exist_ok=True)
    in_paths = _sorted_frames(frames_in)
    if not in_paths:
        msg = f"no frames found in {frames_in}"
        raise ValueError(msg)

    log.info("loading_inpainter", backend=backend, device=resolved.device, radius=resolved.radius, frames=len(in_paths))

    out_paths: list[Path] = []
    total = len(in_paths)
    for index, src in enumerate(in_paths):
        with Image.open(src) as frame_raw:
            frame = frame_raw.convert("RGB")
        if backend == "alpha":
            # Alpha backend ignores the mask (alpha map carries its own shape).
            # The Inpainter Protocol still requires a non-None mask_l argument,
            # so we synthesize a blank one here. Its content is unused.
            blank_mask = Image.new("L", frame.size, 0)
            result = inpaint_fn(frame, blank_mask)
        else:
            mask_img = Image.open(mask_path).convert("L")
            result = inpaint_fn(frame, mask_img)
        dst = frames_out / f"{index:04d}.png"
        result.save(dst)
        out_paths.append(dst)
        if (index + 1) % 30 == 0 or (index + 1) == total:
            log.info("inpaint_progress", done=index + 1, total=total)
    return out_paths


def iter_frame_indices(frames_dir: Path) -> Iterator[int]:
    """Yield the sorted numeric indices of frames in *frames_dir*."""
    for path in _sorted_frames(frames_dir):
        match = _FRAME_PATTERN.match(path.name)
        if match is not None:  # always true, defensive
            yield int(match.group(1))
