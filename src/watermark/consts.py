"""Project-wide constants. Single source of truth for tunable parameters."""

from typing import Final

# Source / output video geometry
VIDEO_SIZE: Final[tuple[int, int]] = (1080, 1920)  # (width, height)
FPS: Final[int] = 24
FRAME_COUNT: Final[int] = 240
DURATION_SEC: Final[float] = 10.0

# Watermark bounding box (left, top, right, bottom) in pixels.
# Single point of iteration: tweak these four numbers and re-run `uv run watermark remove`.
# Visually located by zooming into frame_0000.png (the ✦ is a light-gray
# 4-pointed star, much brighter than the surrounding dark background):
# the Gemini ✦ spans X≈900-980, Y≈1730-1820. Added 10 px safety margin.
WATERMARK_BBOX: Final[tuple[int, int, int, int]] = (890, 1720, 990, 1830)

# Refined mask dilation (pixels) — enough to catch anti-aliased watermark edges
# without turning the symbol-shaped mask back into a visible patch.
DILATE_PX: Final[int] = 4

# Reassembly encoding
CRF: Final[int] = 16
PRESET: Final[str] = "medium"
PIX_FMT: Final[str] = "yuv420p"

# Reverse alpha blending (Lossless.md)
# Logo color used by Gemini's visible watermark (pure white, scalar per channel)
LOGO_RGB: Final[tuple[int, int, int]] = (255, 255, 255)
# Gemini watermark catalog: for outputs > 1024 px, watermark is 96×96 with 64-px margins
ALPHA_WATERMARK_SIZE: Final[int] = 96
ALPHA_MARGIN_RIGHT: Final[int] = 64
ALPHA_MARGIN_BOTTOM: Final[int] = 64
# For 1080×1920 → calibrated from grey.mp4 (solid RGB 126/126/127 background).
# The catalog formula (1080-64-96, 1920-64-96, 1080-64, 1920-64) = (920, 1760, 1016, 1856)
# was off by ~70 px; the actual ✦ position in grey.mp4 is (851, 1691, 947, 1787).
# NOTE: this is only a HINT for anchor search — actual watermark positions vary
# across videos (e.g. omni.mp4 is at (887, 1727), shifted +36px from grey.mp4).
ALPHA_BBOX_HINT: Final[tuple[int, int, int, int]] = (851, 1691, 947, 1787)
# Backward-compatible alias (deprecated).
ALPHA_BBOX_1080X1920: Final[tuple[int, int, int, int]] = ALPHA_BBOX_HINT
