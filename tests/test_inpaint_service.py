"""Tests for the inpaint service.

These tests stay at the unit level — we mock the per-frame inpainting
function so we don't need to download any model weights or run inference.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from watermark.consts import VIDEO_SIZE
from watermark.services import inpaint as inpaint_svc


def _make_frames(dir_: Path, count: int) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (32, 32), (i, i, i)).save(dir_ / f"{i:04d}.png")


def test_sorted_frames_returns_numeric_order(tmp_path: Path) -> None:
    """Files are returned in 0000, 0001, 0002 order, ignoring non-png."""
    _make_frames(tmp_path, 3)
    (tmp_path / "readme.txt").write_text("ignore me")
    frames = inpaint_svc._sorted_frames(tmp_path)
    assert [p.name for p in frames] == ["0000.png", "0001.png", "0002.png"]


def test_iter_frame_indices_yields_ints(tmp_path: Path) -> None:
    """iter_frame_indices yields int indices in sorted order."""
    _make_frames(tmp_path, 3)
    assert list(inpaint_svc.iter_frame_indices(tmp_path)) == [0, 1, 2]


def test_inpaint_frames_raises_when_no_frames(tmp_path: Path) -> None:
    """inpaint_frames raises ValueError on an empty input dir."""
    empty = tmp_path / "in"
    empty.mkdir()
    out = tmp_path / "out"
    mask = tmp_path / "mask.png"
    Image.new("L", VIDEO_SIZE, 0).save(mask)
    with pytest.raises(ValueError, match="no frames"):
        inpaint_svc.inpaint_frames(empty, mask, out)


def test_inpaint_frames_raises_on_unknown_backend(tmp_path: Path) -> None:
    """An unsupported backend name raises ValueError before any frame is processed."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    mask_path = tmp_path / "mask.png"
    _make_frames(in_dir, 1)
    Image.new("L", VIDEO_SIZE, 0).save(mask_path)
    with pytest.raises(ValueError, match="unsupported backend"):
        inpaint_svc.inpaint_frames(in_dir, mask_path, out_dir, inpaint_svc.InpaintOptions(backend="bogus"))


def test_inpaint_frames_telea_default(tmp_path: Path) -> None:
    """The default backend (telea) calls OpenCV and writes one output per input frame."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    mask_path = tmp_path / "mask.png"
    _make_frames(in_dir, 3)
    Image.new("L", VIDEO_SIZE, 0).save(mask_path)

    captured: list[tuple[Image.Image, Image.Image]] = []

    def fake_telea(frame: Image.Image, mask_l: Image.Image) -> Image.Image:
        captured.append((frame, mask_l))
        return Image.new("RGB", (32, 32), (7, 7, 7))

    with patch.dict(inpaint_svc._INPAINTERS, {"telea": fake_telea}):
        outputs = inpaint_svc.inpaint_frames(in_dir, mask_path, out_dir)

    assert len(outputs) == 3
    assert [p.name for p in outputs] == ["0000.png", "0001.png", "0002.png"]
    assert len(captured) == 3
    # First call: input frame is the first 32x32 frame; mask is the 1080x1920 L mask
    first_frame, first_mask = captured[0]
    assert first_frame.size == (32, 32)
    assert first_mask.size == VIDEO_SIZE
    assert first_mask.mode == "L"


def test_inpaint_frames_passes_opencv_algorithm_and_radius(tmp_path: Path) -> None:
    """OpenCV backends receive the selected algorithm and radius."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    mask_path = tmp_path / "mask.png"
    _make_frames(in_dir, 1)
    Image.new("L", VIDEO_SIZE, 0).save(mask_path)
    captured: list[tuple[str, int]] = []

    def fake_opencv(
        frame: Image.Image,
        mask_l: Image.Image,
        *,
        algorithm: str,
        options: inpaint_svc.InpaintOptions | None,
    ) -> Image.Image:
        assert options is not None
        captured.append((algorithm, options.radius))
        return frame.copy()

    with patch.object(inpaint_svc, "_inpaint_opencv", fake_opencv):
        outputs = inpaint_svc.inpaint_frames(
            in_dir,
            mask_path,
            out_dir,
            inpaint_svc.InpaintOptions(backend="ns", radius=9),
        )

    assert [p.name for p in outputs] == ["0000.png"]
    assert captured == [("ns", 9)]


def test_restore_local_texture_adds_detail_inside_mask_only() -> None:
    """Texture restoration adds local detail to a smooth repaired region without changing the outside."""
    original = Image.new("RGB", (64, 64), (80, 80, 80))
    for x in range(64):
        for y in range(64):
            value = 70 if (x + y) % 2 == 0 else 90
            original.putpixel((x, y), (value, value, value))
    repaired = original.copy()
    for x in range(20, 44):
        for y in range(20, 44):
            repaired.putpixel((x, y), (80, 80, 80))
    mask = Image.new("L", (64, 64), 0)
    for x in range(20, 44):
        for y in range(20, 44):
            mask.putpixel((x, y), 255)

    textured = inpaint_svc._restore_local_texture(original, repaired, mask, strength=0.8, feather=1)
    inside = textured.crop((24, 24, 40, 40)).convert("L")
    outside_before = repaired.crop((0, 0, 12, 12))
    outside_after = textured.crop((0, 0, 12, 12))

    assert inside.getextrema()[1] - inside.getextrema()[0] > 0
    assert list(outside_after.getdata()) == list(outside_before.getdata())


def test_inpaint_frames_lama_backend(tmp_path: Path) -> None:
    """The 'lama' backend calls the SimpleLama class and writes one output per frame."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    mask_path = tmp_path / "mask.png"
    _make_frames(in_dir, 2)
    Image.new("L", VIDEO_SIZE, 0).save(mask_path)

    def fake_lama(frame: Image.Image, mask_l: Image.Image) -> Image.Image:
        return Image.new("RGB", (32, 32), (3, 3, 3))

    with patch.dict(inpaint_svc._INPAINTERS, {"lama": fake_lama}):
        outputs = inpaint_svc.inpaint_frames(
            in_dir,
            mask_path,
            out_dir,
            inpaint_svc.InpaintOptions(device="cpu", backend="lama"),
        )

    assert len(outputs) == 2
    assert [p.name for p in outputs] == ["0000.png", "0001.png"]
