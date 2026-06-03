# Local Quality Inpainting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve local watermark removal quality by minimizing the repaired region and adding higher-quality local inpainting controls.

**Architecture:** Keep the existing service/command layout. Extend mask generation with a refined watermark mask mode, extend inpainting with OpenCV algorithm/radius controls and a reusable LaMa runner, and expose those controls through the existing Click commands.

**Tech Stack:** Python 3.12, Pillow, OpenCV, Click, uv, pytest, Ruff, Mypy.

---

### Task 1: Refined Watermark Mask

**Files:**
- Modify: `src/watermark/services/mask.py`
- Test: `tests/test_mask_service.py`

- [ ] Add tests for a refined mask generated from a bright star-shaped watermark crop.
- [ ] Implement `make_watermark_mask` that thresholds bright pixels inside the bbox, keeps meaningful connected regions, applies limited dilation, and writes a binary full-frame mask.
- [ ] Keep the existing rectangle mask behavior available for comparison.

### Task 2: Inpaint Backend Controls

**Files:**
- Modify: `src/watermark/services/inpaint.py`
- Test: `tests/test_inpaint_service.py`

- [ ] Add tests that `inpaint_frames` forwards OpenCV algorithm and radius options.
- [ ] Add `ns` support using `cv2.INPAINT_NS`.
- [ ] Refactor LaMa into a reusable runner so the model loads once per full run instead of once per frame.

### Task 3: CLI Wiring

**Files:**
- Modify: `src/watermark/commands/mask.py`
- Modify: `src/watermark/commands/remove.py`
- Test: `tests/test_mask_cmd.py`
- Test: `tests/test_remove_cmd.py`

- [ ] Add `--mode rectangle|watermark` to mask generation, defaulting to `watermark`.
- [ ] Add `--algorithm telea|ns` or reuse backend naming cleanly for OpenCV inpainting.
- [ ] Add `--radius` so quality experiments do not require editing constants.

### Task 4: Preview And Verification

**Files:**
- Modify: `README.md`
- Use: existing frames under `frames_in/` and `frames_out/`

- [ ] Generate a local crop montage comparing old rectangle output against refined-mask output on representative frames.
- [ ] Run `uv run pytest`, `uv run ruff check`, `uv run ruff format --check`, and `uv run mypy src/`.

**Note:** This workspace is not a git repository, so plan and implementation commits are intentionally skipped.
