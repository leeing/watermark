"""Service: build the static inpainting mask.png from a bounding box."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw

from watermark.consts import VIDEO_SIZE

MaskMode = Literal["rectangle", "watermark"]


@dataclass(frozen=True)
class MaskOutputs:
    """Output paths produced by mask build."""

    mask: Path
    overlay: Path


def _clamp(value: int, low: int, high: int) -> int:
    """Clamp *value* into [low, high]."""
    return max(low, min(high, value))


def make_mask(
    bbox: tuple[int, int, int, int],
    dilate: int,
    out: Path,
    *,
    size: tuple[int, int] = VIDEO_SIZE,
) -> Path:
    """Write a binary (grayscale) mask: white = inpaint, black = keep.

    The bbox is expanded outward by *dilate* pixels on each side, then clamped
    to the image bounds.
    """
    left, top, right, bottom = bbox
    width, height = size
    x0 = _clamp(left - dilate, 0, width)
    y0 = _clamp(top - dilate, 0, height)
    x1 = _clamp(right + dilate, 0, width)
    y1 = _clamp(bottom + dilate, 0, height)

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((x0, y0, x1, y1), fill=255)
    out.parent.mkdir(parents=True, exist_ok=True)
    mask.save(out)
    return out


def make_watermark_mask(
    frame: Image.Image,
    bbox: tuple[int, int, int, int],
    dilate: int,
    out: Path,
    *,
    size: tuple[int, int] = VIDEO_SIZE,
) -> Path:
    """Write a refined binary mask for the bright Gemini watermark symbol.

    The mask is generated from the frame content inside *bbox*: bright pixels
    relative to the local background are selected, then slightly dilated. This
    keeps the repair area close to the symbol instead of filling the entire
    rectangle.
    """
    import cv2
    import numpy as np

    left, top, right, bottom = bbox
    width, height = size
    x0 = _clamp(left, 0, width)
    y0 = _clamp(top, 0, height)
    x1 = _clamp(right, 0, width)
    y1 = _clamp(bottom, 0, height)
    if x1 <= x0 or y1 <= y0:
        msg = f"invalid bbox dimensions after clamping: {bbox}"
        raise ValueError(msg)

    crop = frame.convert("L").crop((x0, y0, x1, y1))
    gray = np.array(crop)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    threshold = max(float(np.median(border)) + 25.0, float(gray.mean()) + float(gray.std()) * 0.45)
    symbol = (gray >= threshold).astype("uint8") * 255

    if dilate > 0:
        kernel_size = dilate * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        symbol = cv2.dilate(symbol, kernel, iterations=1)

    mask = Image.new("L", size, 0)
    mask.paste(Image.fromarray(symbol, mode="L"), (x0, y0))
    out.parent.mkdir(parents=True, exist_ok=True)
    mask.save(out)
    return out


def make_overlay(
    frame: Image.Image,
    bbox: tuple[int, int, int, int],
    dilate: int,
    out: Path,
) -> Path:
    """Save a debug overlay: the original frame with a red box around the bbox."""
    left, top, right, bottom = bbox
    x0 = _clamp(left - dilate, 0, frame.width)
    y0 = _clamp(top - dilate, 0, frame.height)
    x1 = _clamp(right + dilate, 0, frame.width)
    y1 = _clamp(bottom + dilate, 0, frame.height)

    overlay = frame.convert("RGBA").copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0, 255), width=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out)
    return out


def build(
    frame_path: Path,
    bbox: tuple[int, int, int, int],
    dilate: int,
    outputs: MaskOutputs,
    *,
    mode: MaskMode = "rectangle",
) -> tuple[Path, Path]:
    """Convenience: load frame, write mask.png + mask_overlay.png."""
    frame = Image.open(frame_path)
    frame.load()
    if mode == "rectangle":
        make_mask(bbox, dilate, outputs.mask)
    elif mode == "watermark":
        make_watermark_mask(frame, bbox, dilate, outputs.mask)
    else:
        msg = f"unsupported mask mode {mode!r}"
        raise ValueError(msg)
    make_overlay(frame, bbox, dilate, outputs.overlay)
    return outputs.mask, outputs.overlay
