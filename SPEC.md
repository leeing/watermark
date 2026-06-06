# SPEC — 去除 omni.mp4 右下角 Gemini ✦ 水印

> **业务规则源 (Source of Truth)**。代码实现与本文档冲突时，以本文档为准。

## 1. Goal

从 `/Users/qadmlee/cmblab/watermark/omni.mp4` 中去除右下角的 Google Gemini **闪光图标 (✦)** 水印，输出 `omni_clean.mp4`，保持原始分辨率、帧率、时长、音频流不变。

**范围内**：仅去除右下角 ✦ 闪光图标。
**范围外**：左下角 "AI Result" 文字（用户明确未要求去除）。

## 2. Inputs

| 字段 | 值 |
|---|---|
| 文件 | `omni.mp4` |
| 分辨率 | 1080 × 1920 (portrait) |
| 帧率 | 24 fps |
| 时长 | 10.000 s (240 frames) |
| 视频编码 | H.264 (High@L4.0) |
| 音频编码 | AAC LC, 48 kHz, 立体声, ~137 kbps |
| 大小 | ~15.8 MB |

## 3. Output

| 字段 | 值 |
|---|---|
| 文件 | `omni_clean.mp4` |
| 分辨率 | 1080 × 1920 (与源一致) |
| 帧率 | 24 fps (与源一致) |
| 时长 | ≈10.000 s (与源一致) |
| 视频编码 | H.264, CRF 16, yuv420p, preset=medium |
| 音频编码 | AAC (与源一致)，通过 `-c:a copy` 复用原字节 |
| 帧数 | 240 (与源一致) |

## 4. Method

```
omni.mp4
  └─[ffmpeg 抽帧, -vsync 0, -start_number 0]─> frames_in/0000.png ... 0239.png
  └─[PIL 静态 mask]──────────────────────────> mask.png (1080x1920, 灰度, 白=inpaint)
  └─[simple-lama-inpainting on MPS, 逐帧]───> frames_out/0000.png ... 0239.png
  └─[ffmpeg 重组, -map 1:a -c:a copy]───────> omni_clean.mp4
```

**AI 修复器**：`simple_lama_inpainting.SimpleLama` (基于 Suvorov et al. 2021 LaMa)，内部使用 `saic-mdal/lama` 权重，PyTorch 后端。
**设备**：M3 Max 48G 优先 MPS；失败回退 CPU。

## 5. Mask 定义

### Heuristic 路线 (Telea / LaMa / NS)
| 常量 | 值 | 说明 |
|---|---|---|
| `VIDEO_SIZE` | `(1080, 1920)` | 帧尺寸 (W, H) |
| `WATERMARK_BBOX` | `(890, 1720, 990, 1830)` | 水印外接矩形 — 视觉定位定稿 |
| `DILATE_PX` | `4` | mask 安全边距 (像素) |

### Lossless 路线 (alpha — v0.3.0)

| 常量 | 值 | 说明 |
|---|---|---|
| `LOGO_RGB` | `(255, 255, 255)` | Gemini 标定 logo 颜色（纯白） |
| `ALPHA_WATERMARK_SIZE` | `96` | 水印 alpha map 尺寸 |
| `ALPHA_BBOX_1080X1920` | `(851, 1691, 947, 1787)` | **从 grey.mp4 实测标定** (v0.3.0) |

**v0.3.0 变更摘要**:
- **标定源**: 使用 `grey.mp4` (RGB ~126/126/127 纯色背景) 替代 GargantuaX catalog 公式。
  - `calibrate_from_solid_video` 从 grey.mp4 的 96 帧中位数自动检测 ✦ 位置并反推 alpha。
- **anchor 搜索**: 从 `cv2.matchTemplate` (ZNCC) 改为 round-trip error minimization。
  - 每个候选位置执行 reverse → forward alpha，选误差最小的。
- **PNG 层验证**: A8 验证改为在 `frames_in/*.png` / `frames_out/*.png` 级别执行，
  避免 H.264 重编码噪声混入。容忍度: max_diff ≤ 2, mean_diff ≤ 0.5。
- **标定产物**: `data/gemini_alpha_96_from_grey.npy` + `data/gemini_alpha_meta.json`。
  - meta 记录 background_rgb, logo_rgb, bbox, source, created_at。
- **自助标定 CLI**: `watermark calibrate --background-rgb R,G,B --video grey.mp4` 支持任意纯色背景视频标定。
- **测试**: 76/76 tests pass (v0.2.0 68 tests + 8 new tests)。✅ Auto Gate 全绿。

实际绘制 mask 时使用 `bbox` 各边向外扩张 10 px，clamp 到图像边界。

**单点迭代**：bbox 偏差只需修改 `src/watermark/consts.py` 中 `WATERMARK_BBOX` 一处，重跑 `uv run watermark remove`。

### bbox 定位迭代史 (v0.1.0)

| 尝试 | bbox | 结果 | 失败原因 |
|---|---|---|---|
| 1 | (810, 1780, 1060, 1900) | ❌ 空 bbox | 用户提供的图与实际帧不符 |
| 2 | (1030, 1210, 1080, 1330) | ❌ 抓到了 bokeh | 高亮像素搜索找到的是 bokeh 光斑 |
| 3 | (880, 1330, 1070, 1470) | ❌ 还是 bokeh | 同上 |
| 4 | (990, 1490, 1080, 1610) | ❌ crop 仍然空 | 阈值过严 |
| 5 | (930, 1690, 1030, 1790) | ⚠️ 部分去除 | ✦ 在 (900-980, 1730-1820)，bbox 偏右 |
| **6** | **(890, 1720, 990, 1830)** | **✅ 完全去除** | **正确覆盖 ✦** |

## 6. Acceptance Criteria

| # | 准则 | 检查方式 | 结果 |
|---|---|---|---|
| A1 | ✦ 在所有帧中不可见 | 视觉抽帧（0s/5s/9.9s） | ✅ 通过 (telea backend) |
| A2 | 输出分辨率 = 1080 × 1920 | `ffprobe` | ✅ 1080×1920 |
| A3 | 输出帧数 = 240 | `ffprobe` | ✅ 240 |
| A4 | 输出时长 ∈ [9.95, 10.05] s | `ffprobe` | ✅ 10.000 s |
| A5 | 输出视频编码 = h264 | `ffprobe` | ✅ h264 |
| A6 | **音频 MD5 与源完全一致** | `ffmpeg -f md5 -` | ✅ `d6d165e50504f8adae20fdbf6a2de358` |
| A7 | Auto Gate 全绿 | `ruff format --check`, `ruff check`, `mypy src/`, `pytest` | ✅ **68/68** tests pass |
| A8 | **alpha 路线 PNG层 正反演算可逆** (v0.3.0) | `forward(clean) ≈ source` (max_diff ≤ 2, mean_diff ≤ 0.5, 非alpha区域逐字节一致) | ✅ **76/76** tests pass — 待 `omni.mp4` 端到端验证 |

## 7. Phases

- [x] Phase 0 — 脚手架 (`pyproject.toml`, 目录结构, `.gitignore`)
- [x] Phase 0.5 — 编写 SPEC.md
- [x] Phase 1 — 实现 Service 层 (`detect`, `mask`, `inpaint`, `reassemble`, `verify`)
- [x] Phase 2 — 实现 Click Command 层 (`detect`, `mask`, `remove`)
- [x] Phase 3 — 编写 pytest 测试 (v0.1.0 50 tests, +v0.2.0 18 tests = 68 total)
- [x] Phase 4 — Auto Gate 全绿
- [x] Phase 5 — 端到端跑 `omni.mp4` → `omni_clean.mp4`
  - v0.2.0 alpha 路线：失败 (A8 不通过，catalog 位置与实际 ✦ 偏差 ~50px)
  - v0.1.0 telea 路线：**成功** (8 秒，视觉无水印无伪影)
- [x] Phase 6 — 验收报告 (按 `acceptance` skill 模板)

## 8. 工具与版本

| 工具 | 版本 | 路径 |
|---|---|---|
| Python | ≥ 3.12 | system |
| uv | 0.10.0+ | `/opt/homebrew/bin/uv` |
| ffmpeg | 8.0.1 | `/opt/homebrew/bin/ffmpeg` |
| ffprobe | 8.0.1 | `/opt/homebrew/bin/ffprobe` |
| OpenCV (cv2) | 4.13.0 (headless) | pip — **实际使用的 backend** |
| Pillow | 12.2.0 | pip |
| pydantic-settings | 2.14.1 | pip |
| structlog | 25.5.0 | pip |
| Click | 8.1+ | pip |
| httpx | 0.28.1 | pip — fetch_alpha.py 用 |
| ruff | 0.15.15 | dev |
| mypy | 2.1.0 (strict) | dev |
| pytest | 9.0.3 | dev |
| pytest-cov | 7.1.0 | dev |

### v0.3.0 alpha 路线实验结果 ✅

**已实现**:
- `src/watermark/services/alpha.py`:
  - `calibrate_from_solid_video(grey.mp4, bg=(126,126,127))` — 从纯色背景视频标定 alpha map。自动检测 ✦ 位置，中位数帧消噪，逐像素反解 α。
  - `find_anchor` — 用 round-trip error 搜索最优对齐位置（替代 ZNCC template matching）。
  - `load_alpha_map` / `inpaint_alpha_frame` — 加载和反向 alpha 混合（不变）。
- `src/watermark/services/verify.py`:
  - `verify_alpha_round_trip_png` — **PNG 层** round-trip 验证（5 帧: 0000, 0060, 0120, 0180, 0239），避免 H.264 噪声。
  - `verify_alpha_round_trip` — 视频层验证（保留兼容）。
- `src/watermark/commands/calibrate.py`:
  - `watermark calibrate --background-rgb R,G,B --video grey.mp4` — 一键从纯色视频标定 alpha。
- `src/watermark/commands/remove.py`:
  - `--backend alpha` 管线: 标定 bbox → round-trip anchor search → alpha 反向混合 → **PNG 层验证** → 通过后再 reassemble。
- 标定产物:
  - `data/gemini_alpha_96_from_grey.npy` — 从 grey.mp4 标定的 96×96 alpha map (max α=0.557)。
  - `data/gemini_alpha_meta.json` — 记录 background_rgb, bbox=(851,1691,947,1787), source=grey.mp4, created_at。
- `src/watermark/consts.py`:
  - `ALPHA_BBOX_1080X1920` 更新为 `(851, 1691, 947, 1787)` — 从 grey.mp4 实测标定。
- 测试: 76/76 (v0.2.0 68 + 8 new tests)。✅ Auto Gate 全绿。

**grey.mp4 标定结果**:
- 检测到的紧 bbox: `(862, 1700, 938, 1780)` — 76×80 px。
- 展开到 96×96: `(851, 1691, 947, 1787)` (居中)。
- Round-trip 验证: max_diff=1.28 ≤ 2, mean_diff=0.093 ≤ 0.5。✅ **通过**。

**待验证**: `omni.mp4` 端到端 `--backend alpha` round-trip 仍需 anchor search 验证（
grey.mp4 的 ✦ 位置与 omni.mp4 可能略有偏差，find_anchor 用于纠正此偏差）。

**当前生产状态**: `--backend telea` 默认，`--backend alpha` 可选（opt-in）。

## 9. Risks & Mitigations

| 风险 | 概率 | 缓解 |
|---|---|---|
| 首版 bbox 偏了 | 中 | `consts.py` 单点改；`watermark detect` 提供裁剪图人工核对 |
| `SimpleLama` 内部默认 cuda，mps 报错 | 中 | `inpaint.py` 显式 `.to("cpu")`；`--device cpu` 兜底 |
| MPS 内存峰值超限（理论上 48G 不会） | 极低 | 回退 CPU (预计 ~10 min) |
| `-c:a copy` 漂移 | 极低 | A6 MD5 断言自动捕获，失败则非零退出 |
| 闪烁 (LaMa 逐帧在低对比纹理) | 极低 | 水印是静态低频，flicker 几乎不可见 |
| **alpha catalog 位置与实际不符** | **高** | A8 验证自动捕获，失败立即拒绝；退回 telea (v0.1.0) |
| Hook 失败 3 轮未过 | 中 | 修代码 → 最多 3 轮 → 报告用户 |

## 9. Risks & Mitigations

| 风险 | 概率 | 缓解 |
|---|---|---|
| 首版 bbox 偏了 | 中 | `consts.py` 单点改；`watermark detect` 提供裁剪图人工核对 |
| `SimpleLama` 内部默认 cuda，mps 报错 | 中 | `inpaint.py` 显式 `.to("cpu")`；`--device cpu` 兜底 |
| MPS 内存峰值超限（理论上 48G 不会） | 极低 | 回退 CPU (预计 ~10 min) |
| `-c:a copy` 漂移 | 极低 | A6 MD5 断言自动捕获，失败则非零退出 |
| LaMa CPU 太慢 | 高 | **改用 OpenCV Telea (默认 backend)**，性能 50× 提升 |
| Hook 失败 3 轮未过 | 中 | 修代码 → 最多 3 轮 → 报告用户 |

## 10. Fallback: ProPainter HF Space

若本地 Telea 输出明显瑕疵：
1. 浏览器打开 https://huggingface.co/spaces/sczhou/ProPainter
2. 上传 `omni.mp4` + `mask.png`，用点击工具绘制掩膜
3. 下载结果
4. **副作用**：Space 强制把最长边 ≤ 1280 px → 输出 720 × 1280
5. 若需原分辨率：`ffmpeg -i result.mp4 -vf scale=1080:1920:flags=lanczos omni_clean.mp4`

## 变更日志

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-06-03 | 0.1.0 | 初始 SPEC |
| 2026-06-03 | 0.1.1 | 修正 `WATERMARK_BBOX=(930,1690,1030,1790)` (视觉定位)；后端从 LaMa 改为 OpenCV Telea |
| 2026-06-03 | 0.1.2 | **修正 bbox=(890,1720,990,1830)** — 用户反馈初版 ✦ 残留，定位后再次修正 |
| 2026-06-03 | 0.2.0 | **alpha lossless 路线实验** — 加 `--backend alpha` (默认)、`data/gemini_96_alpha.png` (GargantuaX MIT 标定)、A8 验证；**实验失败**：catalog 位置与实际 ✦ 偏差 ~50 px，A8 差 70.5 > tolerance 2。回退 telea 默认 |
| 2026-06-04 | 0.3.0 | **alpha 路线重构** — 从 grey.mp4 标定 alpha map (不再是 GargantuaX catalog); `find_anchor` 改为 round-trip error 搜索; PNG 层 round-trip 验证; `watermark calibrate --background-rgb` 一键标定; 测试 76/76 ✅ |

## 8. 工具与版本

| 工具 | 版本 | 路径 |
|---|---|---|
| Python | ≥ 3.12 | system |
| uv | 0.10.0+ | `/opt/homebrew/bin/uv` |
| ffmpeg | 8.0.1 | `/opt/homebrew/bin/ffmpeg` |
| ffprobe | 8.0.1 | `/opt/homebrew/bin/ffprobe` |
| simple-lama-inpainting | ≥ 0.1.2 | pip |
| torch | ≥ 2.3 (arm64 macOS, MPS) | pip |
| Pillow | ≥ 10.4 | pip |
| Click | ≥ 8.1 | pip |
| pydantic-settings | ≥ 2.0 | pip |
| structlog | ≥ 24.1 | pip |
| ruff | ≥ 0.6 | dev |
| mypy | ≥ 1.10 (strict) | dev |
| pytest | ≥ 8.3 | dev |

## 9. Risks & Mitigations

| 风险 | 概率 | 缓解 |
|---|---|---|
| 首版 bbox 偏了 | 中 | `consts.py` 单点改；`watermark detect` 提供裁剪图人工核对 |
| `SimpleLama` 内部默认 cuda，mps 报错 | 中 | `inpaint.py` 显式 `.to(torch.device("mps"))`；`--device cpu` 兜底 |
| MPS 内存峰值超限（理论上 48G 不会） | 极低 | 回退 CPU (预计 ~10 min) |
| `-c:a copy` 漂移 | 极低 | A6 MD5 断言自动捕获，失败则非零退出 |
| 逐帧 LaMa 闪烁（小概率，静态低频场景罕见） | 极低 | 视觉抽帧验证 A1；如失败走 ProPainter HF Space fallback |
| Hook 失败 3 轮未过 | 中 | 修代码 → 最多 3 轮 → 报告用户 |

## 10. Fallback: ProPainter HF Space

若本地 LaMa 输出明显瑕疵：
1. 浏览器打开 https://huggingface.co/spaces/sczhou/ProPainter
2. 上传 `omni.mp4` + `mask.png`，用点击工具绘制掩膜
3. 下载结果
4. **副作用**：Space 强制把最长边 ≤ 1280 px → 输出 720 × 1280
5. 若需原分辨率：`ffmpeg -i result.mp4 -vf scale=1080:1920:flags=lanczos omni_clean.mp4`

## 变更日志

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-06-03 | 0.1.0 | 初始 SPEC |
