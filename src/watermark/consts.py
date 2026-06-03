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
