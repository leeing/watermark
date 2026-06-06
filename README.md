# watermark

使用本地高质量 inpainting 或数学无损的逆向 alpha 混合，去除视频右下角 **Gemini ✦** 水印（支持 `omni.mp4`、`Flow_202606032214.mp4` 等）。

## 环境准备

- [uv](https://github.com/astral-sh/uv) ≥ 0.10
- ffmpeg ≥ 8.0 (`brew install ffmpeg`)

## 安装

```bash
uv sync
```

## 使用方式

项目支持两种主要工作流：**数学无损（Alpha）** 和 **启发式 Inpainting（Telea / NS / LaMa）**。

### 1. 数学无损路线（Alpha Backend）—— 推荐

此路线使用逆向 alpha 混合公式减去水印，精确恢复原始背景像素。

#### 运行水印去除：

可通过环境变量 `WATERMARK_SOURCE_VIDEO` 和 `WATERMARK_OUTPUT_VIDEO` 指定输入/输出视频：

```bash
# 使用 alpha backend 清除 Flow_202606032214.mp4
WATERMARK_SOURCE_VIDEO=Flow_202606032214.mp4 WATERMARK_OUTPUT_VIDEO=Flow_clean.mp4 uv run python -m watermark.main remove --backend alpha

# 使用 alpha backend 清除 omni.mp4
WATERMARK_SOURCE_VIDEO=omni.mp4 WATERMARK_OUTPUT_VIDEO=omni_clean.mp4 uv run python -m watermark.main remove --backend alpha
```

#### CLI 标定：

如果预标定的 alpha map 不存在，或你使用了新的视频分辨率/水印样式，可以从纯色背景视频标定新的 alpha map：

```bash
uv run python -m watermark.main calibrate \
  --video data/grey.mp4 \
  --background-rgb 126,126,127 \
  --bbox auto \
  --out-alpha data/gemini_alpha_96_from_grey.npy \
  --out-meta data/gemini_alpha_meta.json
```

---

### 2. 启发式 Inpainting 路线（Telea / NS / LaMa）

如果没有 alpha map，或处理的是非标准水印，可使用启发式 inpainting：

```bash
# 第 1 步：提取首帧并裁剪水印区域供人工检查
uv run python -m watermark.main detect omni.mp4
open frames/watermark_crop.png   # 确认 ✦ 完全位于裁剪区域内

# 第 2 步：根据 bbox 常量生成精细的符号形状 mask.png
uv run python -m watermark.main mask
open frames/mask_overlay.png     # 确认红色框覆盖 ✦

# 第 3 步：使用 Telea 进行 inpainting + 重组（视觉完美，速度快）
uv run python -m watermark.main remove --backend telea --mask-mode watermark --radius 3 --texture-strength 0.25 --feather 2
```

---

## Alpha 原理（逆向 Alpha 混合）

Gemini 使用逐像素线性 alpha 合成公式对帧添加水印：

$$\text{watermarked} = \alpha \cdot \text{logo} + (1 - \alpha) \cdot \text{original}$$

其中：
- $\alpha \in [0, 1]$ 为不透明度（逐像素 alpha map，96×96 静态矩阵）。
- $\text{logo}$ 为水印 logo 的颜色（通常为纯白，`(255, 255, 255)`）。
- $\text{original}$ 为未被污染的原始背景帧。

由于正向操作是线性的，可以数学上精确逆向（无损）：

$$\text{original} = \frac{\text{watermarked} - \alpha \cdot \text{logo}}{1 - \alpha}$$

### 本项目的高级优化：

- **逐像素自适应 Alpha 钳位**：在暗背景帧上，逆向 alpha 除法可能导致数值下溢（产生负像素值）。本管线采用逐像素背景感知钳位：
  $$\alpha_{\text{safe}} = \min\left(\alpha_{\text{map}}, \frac{\text{min\_channel}(\text{watermarked})}{\max(\text{logo})}\right)$$
  这保证所有重建像素保持在 $[0, 255]$ 有效范围内，避免暗背景上的伪影。

- **鲁棒多帧锚点搜索**：水印位置可能因视频不同而偏移若干像素。CLI 对多帧采样运行 ZNCC（零均值归一化互相关）滑动模板匹配，并使用中值投票坐标实现亚像素对齐精度。

- **像素级验证**：验证步骤执行像素级往返重建差异检查，确保重建帧 $\text{forward}(\text{reverse}(\text{frame})) \approx \text{frame}$ 在容差内完全成立。

---

## 调整 bbox

若使用启发式 inpainting 时水印未完全覆盖，编辑 `src/watermark/consts.py`：

```python
WATERMARK_BBOX: tuple[int, int, int, int] = (890, 1720, 990, 1830)  # left, top, right, bottom
```

然后重新运行 `uv run python -m watermark.main remove`。

## `remove` 命令参数

- `--backend [telea|alpha|ns|lama]`：Inpainting 后端。默认：`telea`。
- `--bbox TEXT`：自定义水印边界框 (left,top,right,bottom)。
- `--device [mps|cpu|cuda]`：LaMa 使用的 PyTorch 设备。默认：`mps`。
- `--dilate INTEGER`：Mask 膨胀大小。默认：`4`。
- `--mask-mode [watermark|rectangle]`：符号细化 mask 或完整矩形框。默认：`watermark`。
- `--radius INTEGER`：OpenCV inpainting 半径。默认：`3`。
- `--texture-strength FLOAT`：局部纹理恢复强度。默认：`0.25`。
- `--feather INTEGER`：混合羽化大小。默认：`2`。
- `--skip-verify`：跳过约束检查（不推荐用于生产环境）。

## 验收标准

- `ffprobe omni_clean.mp4` 输出 `width=1080 height=1920 nb_frames=240 duration≈10s`
- 源视频与处理后视频的音频 MD5 **逐字节一致**（使用 `-c:a copy` 跳过重新编码）。
- 所有帧中水印完全不可见。

详见 [SPEC.md](SPEC.md)。
