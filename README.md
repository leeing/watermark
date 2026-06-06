# watermark

Remove the bottom-right **Gemini ✦** watermark from videos (`omni.mp4`, `Flow_202606032214.mp4`, etc.) using local, quality-first inpainting or mathematically lossless reverse alpha blending.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) ≥ 0.10
- ffmpeg ≥ 8.0 (`brew install ffmpeg`)

## Install

```bash
uv sync
```

## Usage

The project supports two main workflows: **Mathematically Lossless (Alpha)** and **Heuristic Inpainting (Telea/NS/LaMa)**.

### 1. Mathematically Lossless Route (Alpha Backend) - Recommended

This route uses the reverse alpha blending formula to subtract the watermark, restoring the exact original background pixels underneath.

#### Running Watermark Removal:
You can specify the input and output videos using environment variables `WATERMARK_SOURCE_VIDEO` and `WATERMARK_OUTPUT_VIDEO`:

```bash
# Clean Flow_202606032214.mp4 using the alpha backend
WATERMARK_SOURCE_VIDEO=Flow_202606032214.mp4 WATERMARK_OUTPUT_VIDEO=Flow_clean.mp4 uv run python -m watermark.main remove --backend alpha

# Clean omni.mp4 using the alpha backend
WATERMARK_SOURCE_VIDEO=omni.mp4 WATERMARK_OUTPUT_VIDEO=omni_clean.mp4 uv run python -m watermark.main remove --backend alpha
```

#### CLI Calibrating:
If the pre-calibrated alpha map doesn't exist or you are using a new video resolution / watermark style, you can calibrate a new alpha map from a solid-background video:

```bash
uv run python -m watermark.main calibrate \
  --video data/grey.mp4 \
  --background-rgb 126,126,127 \
  --bbox auto \
  --out-alpha data/gemini_alpha_96_from_grey.npy \
  --out-meta data/gemini_alpha_meta.json
```

---

### 2. Heuristic Inpainting Route (Telea / NS / LaMa)

If the alpha map is not available or you are working with non-standard watermarks, you can use heuristic inpainting:

```bash
# Step 1: extract first frame and crop watermark region for human inspection
uv run python -m watermark.main detect omni.mp4
open frames/watermark_crop.png   # verify the ✦ is fully inside the crop

# Step 2: generate a refined symbol-shaped mask.png from the bbox constants
uv run python -m watermark.main mask
open frames/mask_overlay.png     # verify the red box covers the ✦

# Step 3: inpaint + reassemble using Telea (visually perfect, fast)
uv run python -m watermark.main remove --backend telea --mask-mode watermark --radius 3 --texture-strength 0.25 --feather 2
```

---

## The Alpha Principle (Reverse Alpha Blending)

Gemini applies watermarks to frames using a per-pixel linear alpha compositing formula:

$$\text{watermarked} = \alpha \cdot \text{logo} + (1 - \alpha) \cdot \text{original}$$

Where:
- $\alpha \in [0, 1]$ is the opacity (per-pixel alpha map, a static 96x96 matrix).
- $\text{logo}$ is the color of the watermark logo (typically pure white, `(255, 255, 255)`).
- $\text{original}$ is the unpolluted background frame.

Since the forward operation is linear, it can be mathematically inverted (Lossless):

$$\text{original} = \frac{\text{watermarked} - \alpha \cdot \text{logo}}{1 - \alpha}$$

### Advanced Optimizations in this Project:
- **Per-Pixel Adaptive Alpha Clamping**: On dark background frames, reverse alpha division can lead to numerical underflow (resulting in negative pixels). Our pipeline uses per-pixel background-aware clamping:
  $$\alpha_{\text{safe}} = \min\left(\alpha_{\text{map}}, \frac{\text{min\_channel}(\text{watermarked})}{\max(\text{logo})}\right)$$
  This guarantees that all reconstructed pixels remain in the valid $[0, 255]$ range, avoiding artifacts on dark backgrounds.
- **Robust Multi-Frame Anchor Search**: Watermark placement can slightly shift by a few pixels depending on the video. The CLI runs ZNCC (Zero-mean Normalized Cross-Correlation) sliding template-matching on multiple sample frames and uses median-voted coordinates to achieve sub-pixel alignment accuracy.
- **PNG-Level verification**: The verification step performs PNG-level round-trip reconstruction difference checking to ensure that the reconstructed frame $\text{forward}(\text{reverse}(\text{frame})) \approx \text{frame}$ holds perfectly within tolerances.

---

## Tuning the bbox

If the watermark isn't fully covered when using heuristic inpainting, edit `src/watermark/consts.py`:

```python
WATERMARK_BBOX: tuple[int, int, int, int] = (890, 1720, 990, 1830)  # left, top, right, bottom
```

Then re-run `uv run python -m watermark.main remove`.

## CLI Arguments for `remove`

- `--backend [telea|alpha|ns|lama]`: The inpainting backend. Default: `telea`.
- `--bbox TEXT`: Custom watermark bounding box (left,top,right,bottom).
- `--device [mps|cpu|cuda]`: PyTorch device for LaMa. Default: `mps`.
- `--dilate INTEGER`: Mask dilation size. Default: `4`.
- `--mask-mode [watermark|rectangle]`: Symbol-refined mask or full rectangle box. Default: `watermark`.
- `--radius INTEGER`: OpenCV inpainting radius. Default: `3`.
- `--texture-strength FLOAT`: Local texture restoration strength. Default: `0.25`.
- `--feather INTEGER`: Blending feather size. Default: `2`.
- `--skip-verify`: Skip checking constraints (not recommended for production).

## Acceptance Criteria

- `ffprobe omni_clean.mp4` reports `width=1080 height=1920 nb_frames=240 duration≈10s`
- Audio MD5 of source and clean videos are **byte-identical** (using `-c:a copy` bypasses re-encoding).
- Watermark is completely invisible in all frames.

See [SPEC.md](SPEC.md) for details.
