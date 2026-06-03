# watermark

Remove the bottom-right **Gemini ✦** watermark from `omni.mp4` using local, quality-first inpainting.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) ≥ 0.10
- ffmpeg ≥ 8.0 (`brew install ffmpeg`)

## Install

```bash
uv sync
```

## Usage

```bash
# Step 1: extract first frame and crop watermark region for human inspection
uv run watermark detect omni.mp4
open frames/watermark_crop.png   # verify the ✦ is fully inside the crop

# Step 2: generate a refined symbol-shaped mask.png from the bbox constants
uv run watermark mask
open frames/mask_overlay.png     # verify the red box covers the ✦

# Step 3: quality-first local inpaint + reassemble
uv run watermark remove --backend ns --mask-mode watermark --radius 3 --texture-strength 0.25 --feather 2
# output: omni_clean.mp4
```

## Tuning the bbox

If the watermark isn't fully covered, edit `src/watermark/consts.py`:

```python
WATERMARK_BBOX: tuple[int, int, int, int] = (810, 1780, 1060, 1900)  # left, top, right, bottom
```

Then re-run `uv run watermark remove`.

## Quality tuning

The default mask mode is `watermark`, which extracts the bright ✦ shape from the frame and avoids repairing a full rectangle. This is usually much less visible than the old rectangular mask.

Useful local comparisons:

```bash
uv run watermark remove --backend ns --mask-mode watermark --radius 3 --texture-strength 0.25 --feather 2
uv run watermark remove --backend telea --mask-mode watermark --radius 3 --texture-strength 0.25 --feather 2
uv run watermark remove --backend telea --mask-mode rectangle --dilate 10
```

`--backend ns` is often more natural on textured backgrounds; `telea` can be sharper on simple dark regions. `--backend lama` is available only if `simple-lama-inpainting` is installed, and is much slower.

`--texture-strength` restores a little local high-frequency detail inside the repaired region. If the patch looks too grainy, lower it; if it looks too smooth or glassy, raise it slightly. `--feather` controls the repaired-region edge blend.

## Acceptance

- `ffprobe omni_clean.mp4` reports `width=1080 height=1920 nb_frames=240 duration≈10s`
- Audio MD5 of `omni.mp4` and `omni_clean.mp4` are **byte-identical**
- ✦ is invisible in all frames

See [SPEC.md](SPEC.md) for the full spec.
