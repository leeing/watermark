"""Tests for the mask service."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from watermark.consts import DILATE_PX, VIDEO_SIZE, WATERMARK_BBOX
from watermark.services import mask as mask_svc


def _count_white_pixels(img: Image.Image) -> int:
    grayscale = img.convert("L")
    return sum(1 for px in grayscale.getdata() if px >= 128)


def test_make_mask_writes_binary_png(tmp_path: Path) -> None:
    """make_mask produces a grayscale PNG with the expected white area.

    The actual white area equals `(clamped_x1 - clamped_x0 + 1) *
    (clamped_y1 - clamped_y0 + 1)` because PIL's Draw.rectangle fills
    pixels inclusively on both endpoints and the dilated bbox is clamped
    to the image bounds.
    """
    out = tmp_path / "mask.png"
    mask_svc.make_mask(WATERMARK_BBOX, DILATE_PX, out)
    assert out.is_file()
    img = Image.open(out)
    assert img.mode == "L"
    assert img.size == VIDEO_SIZE

    # Compute the expected white area using the same clamping the service applies.
    # PIL internally clips to the last valid index (width-1, height-1).
    left, top, right, bottom = WATERMARK_BBOX
    width, height = VIDEO_SIZE
    x0 = max(0, left - DILATE_PX)
    y0 = max(0, top - DILATE_PX)
    x1 = min(width - 1, right + DILATE_PX)
    y1 = min(height - 1, bottom + DILATE_PX)
    expected = (x1 - x0 + 1) * (y1 - y0 + 1)
    assert _count_white_pixels(img) == expected


def test_make_watermark_mask_targets_bright_symbol_not_full_rectangle(tmp_path: Path) -> None:
    """Refined mask covers the bright watermark symbol without filling the whole bbox."""
    frame = Image.new("RGB", VIDEO_SIZE, (30, 30, 30))
    bbox = (100, 100, 180, 180)
    draw = ImageDraw.Draw(frame)
    draw.polygon(
        [
            (140, 108),
            (152, 128),
            (172, 140),
            (152, 152),
            (140, 172),
            (128, 152),
            (108, 140),
            (128, 128),
        ],
        fill=(210, 210, 210),
    )
    out = tmp_path / "watermark-mask.png"

    mask_svc.make_watermark_mask(frame, bbox, dilate=2, out=out)

    img = Image.open(out)
    white = _count_white_pixels(img)
    rectangle_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    assert img.mode == "L"
    assert img.size == VIDEO_SIZE
    assert 1_000 < white < rectangle_area // 2
    assert img.getpixel((140, 140)) == 255
    assert img.getpixel((100, 100)) == 0


def test_build_uses_watermark_mask_mode(tmp_path: Path) -> None:
    """build() can produce the refined watermark mask mode."""
    frame_path = tmp_path / "frame.png"
    frame = Image.new("RGB", VIDEO_SIZE, (20, 20, 20))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((50, 50, 70, 70), fill=(220, 220, 220))
    frame.save(frame_path)
    mask_out = tmp_path / "m.png"
    overlay_out = tmp_path / "o.png"

    mask_svc.build(frame_path, (40, 40, 80, 80), 1, mask_svc.MaskOutputs(mask_out, overlay_out), mode="watermark")

    img = Image.open(mask_out)
    assert 0 < _count_white_pixels(img) < 1_000
    assert overlay_out.is_file()


def test_make_mask_clamps_to_image_bounds(tmp_path: Path) -> None:
    """If the dilated bbox overflows, the mask is clipped to the image edges."""
    out = tmp_path / "mask.png"
    # bbox that, after dilate, will overflow the bottom-right corner
    bbox = (1060, 1900, 1080, 1920)
    mask_svc.make_mask(bbox, dilate=20, out=out)
    img = Image.open(out)
    # After dilation: (1040..1080, 1880..1920), service clamps to image bounds
    # (1080, 1920). PIL then internally clips to (1040..1079, 1880..1919) since
    # the image width/height are 1080/1920 (last valid index 1079/1919).
    # = 40 x 40 = 1600 pixels (inclusive endpoints, clamped at the edge).
    x0, y0 = 1040, 1880
    x1, y1 = 1080 - 1, 1920 - 1  # PIL's internal clamp to last valid index
    expected = (x1 - x0 + 1) * (y1 - y0 + 1)
    assert _count_white_pixels(img) == expected


def test_make_overlay_draws_red_box(tmp_path: Path) -> None:
    """make_overlay returns the saved path and creates a non-empty PNG with a red outline."""
    frame = Image.new("RGB", VIDEO_SIZE, (50, 50, 50))
    out = tmp_path / "overlay.png"
    returned = mask_svc.make_overlay(frame, WATERMARK_BBOX, DILATE_PX, out)
    assert returned == out
    assert out.is_file()
    overlay = Image.open(out)
    assert overlay.size == VIDEO_SIZE

    # The red outline is at the *clamped dilated* bbox edges (3-px thick line).
    # Use clamped coordinates so we don't sample out-of-bounds pixels when the
    # bbox sits near the image edge.
    left, top, right, bottom = WATERMARK_BBOX
    x0 = max(0, left - DILATE_PX)
    y0 = max(0, top - DILATE_PX)
    x1 = min(VIDEO_SIZE[0], right + DILATE_PX)
    y1 = min(VIDEO_SIZE[1], bottom + DILATE_PX)
    mid_y = (y0 + y1) // 2
    mid_x = (x0 + x1) // 2
    edge_samples = [
        (x0, mid_y),  # left edge of outline
        (x1 - 1, mid_y),  # right edge (clamp inside)
        (mid_x, y0),  # top edge
        (mid_x, y1 - 1),  # bottom edge (clamp inside)
    ]
    red_pixels = [overlay.convert("RGB").getpixel((x, y)) for x, y in edge_samples]
    dominantly_red = [px for px in red_pixels if px[0] > 200 and px[1] < 80 and px[2] < 80]
    assert dominantly_red, f"no red edge pixel found in {red_pixels}"


def test_make_overlay_clamps_to_image_bounds(tmp_path: Path) -> None:
    """Overlay works even with a bbox near the edge."""
    frame = Image.new("RGB", VIDEO_SIZE, (0, 0, 0))
    out = tmp_path / "overlay.png"
    mask_svc.make_overlay(frame, (1060, 1900, 1080, 1920), dilate=20, out=out)
    assert out.is_file()


def test_clamp_helper() -> None:
    """Internal clamp handles values inside / below / above range."""
    assert mask_svc._clamp(5, 0, 10) == 5
    assert mask_svc._clamp(-3, 0, 10) == 0
    assert mask_svc._clamp(99, 0, 10) == 10


def test_build_writes_both_files(tmp_path: Path) -> None:
    """build() writes both mask.png and mask_overlay.png."""
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", VIDEO_SIZE, (0, 0, 0)).save(frame_path)
    mask_out = tmp_path / "m.png"
    overlay_out = tmp_path / "o.png"
    mask_svc.build(frame_path, WATERMARK_BBOX, DILATE_PX, mask_svc.MaskOutputs(mask_out, overlay_out))
    assert mask_out.is_file()
    assert overlay_out.is_file()
