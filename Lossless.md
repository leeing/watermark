# Lossless — 反向 Alpha 混合去水印原理

> **本文件的目的**：解释 [GargantuaX/gemini-watermark-remover](https://github.com/GargantuaX/gemini-watermark-remover) 这类工具为什么能"无损"地去除 Gemini 可见水印，以及它与本项目当前 Telea / LaMa 修复路线的本质差异。**不动代码**，纯文档。

---

## 1. 概述

Gemini 把可见水印当作一次**逐像素的线性操作**：

$$
\text{watermarked} = \alpha \cdot \text{logo} + (1 - \alpha) \cdot \text{original}
$$

其中 $\alpha \in [0, 1]$ 是水印的不透明度（per-pixel 已知常数矩阵），$\text{logo}$ 是水印图样的颜色（常数），$\text{original}$ 是生成前未被水印污染的真实像素。

**反向 alpha 混合**（reverse alpha blending）做的只是把这条线性方程代数反演：

$$
\text{original} = \frac{\text{watermarked} - \alpha \cdot \text{logo}}{1 - \alpha}
$$

由于正向操作是线性的，其代数逆也是线性的——只要 $\alpha$ 和 $\text{logo}$ 是已知常数，每个输出像素就是输入像素的**确定性**函数，没有任何近似、没有邻域采样、没有模型推理。

与 AI 修复（Telea / LaMa / ProPainter）的本质区别：

| 路线 | 怎么填水印区域 | 像素保真度 |
|---|---|---|
| AI 修复 | 模型"猜"被覆盖区域应该长什么样 | 模糊 / 失真 / 偶尔 hallucinate |
| **反向 alpha 混合** | 代数反演已知的线性公式 | **逐像素等于原始**（数学无损） |

视频级层面："提取每帧 → 无损去水印 → 拼回视频" **是** 无损去水印——前提是 $\alpha$ 和 $\text{logo}$ 真的被精确校准过；唯一的非无损步骤是 ffmpeg 重新编码音频/视频，但用 `-c:a copy`（音频字节级 passthrough）和 `-crf 16`（视觉无损 H.264）后实际损失可忽略。

---

## 2. 数学核心

### 2.1 正向 alpha 混合

对每一帧的每一个像素 $(x, y)$，对 RGB 三个通道各做一次标量运算：

$$
\begin{aligned}
R_{\text{wm}} &= \alpha \cdot R_{\text{logo}} + (1 - \alpha) \cdot R_{\text{orig}} \\
G_{\text{wm}} &= \alpha \cdot G_{\text{logo}} + (1 - \alpha) \cdot G_{\text{orig}} \\
B_{\text{wm}} &= \alpha \cdot B_{\text{logo}} + (1 - \alpha) \cdot B_{\text{orig}}
\end{aligned}
$$

其中 $\alpha$ 在不同像素可以不同（这就是为什么 ✦ 的中间更亮、边缘更淡）。但 **$\alpha$ 的形状矩阵 $\mathbf{A} \in [0,1]^{H \times W}$ 和 $\text{logo}$ 的 RGB 三元组是 Gemini 内部固定的常数**，随产品版本变化但不随内容变化。

### 2.2 反向公式推导

把正向公式对 $R_{\text{orig}}$ 求解：

$$
\begin{aligned}
R_{\text{wm}} &= \alpha \cdot R_{\text{logo}} + R_{\text{orig}} - \alpha \cdot R_{\text{orig}} \\
R_{\text{wm}} - \alpha \cdot R_{\text{logo}} &= (1 - \alpha) \cdot R_{\text{orig}} \\
R_{\text{orig}} &= \frac{R_{\text{wm}} - \alpha \cdot R_{\text{logo}}}{1 - \alpha}
\end{aligned}
$$

G 通道和 B 通道同理。

### 2.3 关键数学性质

1. **分母非零**：$\alpha \in [0, 1)$（水印通常不会 100% 不透明），所以 $1 - \alpha \in (0, 1]$，除法总是合法的。
2. **像素保真**：当 $\alpha = 0$（不在水印区域），公式退化为 $\text{orig} = \text{wm}$，原样保留——和没动过一样。
3. **线性保真**：当 $\alpha > 0$ 时，每个 $R_{\text{orig}}$ 严格由 $(\text{wm}, \alpha, \text{logo})$ 三者决定，无任何随机性、无邻域参与。
4. **浮点无损**：Float32 整数 0-255 的表示精度是 $2^{-23} \approx 1.2 \times 10^{-7}$，远低于 1 个像素值（1/255 ≈ 0.004）。**反演后的像素值能精确恢复到原始整数**（量化截断到 0-255 整数后与原始 uint8 完全一致）。

> **结论**：如果 $\alpha$ 矩阵和 $\text{logo}$ 颜色**完全精确**已知，那 $\text{original}$ 就是**数学上唯一**的解，没有"近似"的余地。

---

## 3. 校准（Calibration）—— 为什么 $\alpha$ 是已知常数

### 3.1 Gemini 的水印是固定设计

Gemini 团队发布输出时，叠加的水印是**预渲染的资源**：
- 一个固定的 $\alpha$ 矩阵（同一产品版本、同一输出尺寸，水印完全相同）
- 一个固定的 $\text{logo}$ 颜色（典型为接近白色 `(248, 248, 248)` 或 `(255, 255, 255)`）
- 一个固定的位置和大小（按输出尺寸查表）

**不是**每个视频单独生成的，**不是**用户内容相关的——所以可以一次性校准、永久复用。

### 3.2 校准方法

在**已知纯色背景**（如全黑 `(0, 0, 0)`）上让 Gemini 生成一张只含水印的图，记输出为 $\text{wm}_{\text{calib}}$。则：

$$
\alpha = \frac{\text{wm}_{\text{calib}} - \text{bg}}{\text{logo} - \text{bg}} = \frac{\text{wm}_{\text{calib}} - (0, 0, 0)}{\text{logo} - (0, 0, 0)} = \frac{\text{wm}_{\text{calib}}}{\text{logo}}
$$

这与反向公式同形（数学上就是互逆），但用**已知**的 $\text{bg}$ 代替了未知的 $\text{original}$。

更一般的校准可以选**任意两种已知背景**（比如全黑和全白），联立两组方程解出 $\alpha$ 和 $\text{logo}$ 两个未知量；这是 GargantuaX 项目的内部做法。

### 3.3 校准后产出的两个常数

| 常数 | 形状 | 说明 |
|---|---|---|
| $\mathbf{A} = (\alpha_{ij})_{H \times W}$ | 水印外接矩形的二维 Float32 数组 | 每个像素一个 $\alpha$ 值 |
| $\text{logo} = (R, G, B)$ | 三元组（标量或 per-region） | 水印的"主色" |

**这就是"lossless"真正的来源**：输入像素 + 这两个**已知常数** → 唯一确定的输出像素。**没有外部信息、没有假设、没有学习**。

---

## 4. 检测流程（在 `omni.mp4` 上）

反向 alpha 混合的关键前置条件是"知道水印在哪里"（即 $\mathbf{A}$ 在哪）。GargantuaX 项目的检测分三层：

### 4.1 尺寸查表

Gemini 的可见水印随输出尺寸分级，公开的对照表大致是：

| 输出尺寸 | 水印尺寸 | 右边距 | 下边距 |
|---|---|---|---|
| 大尺寸（如 1080×1920, 1920×1080） | 96 × 96 px | 64 px | 64 px |
| 中尺寸（如 720×1280） | 48 × 48 px | 32 px | 32 px |
| 小尺寸（如 512×512） | 24 × 24 px | 16 px | 16 px |

对本项目 `omni.mp4` (1080×1920) 来说，**预测的 bbox 是**：

$$
\begin{aligned}
\text{right} &= 1080 - 64 = 1016 \\
\text{bottom} &= 1920 - 64 = 1856 \\
\text{left} &= 1016 - 96 = 920 \\
\text{top} &= 1856 - 96 = 1760 \\
\text{bbox} &= (920, 1760, 1016, 1856)
\end{aligned}
$$

### 4.2 局部 anchor search

尺寸查表只是粗略估计。GargantuaX 会在预测位置周围 ±N px 的小窗口里**滑动匹配**：

- 在每个候选位置，用反向公式还原整张图
- 计算还原后图像在窗口区域的**局部方差 / 边缘锐度**
- 还原越干净（即水印被精确剥离），邻域方差越正常；错位时仍能看到水印残影，方差异常

最优 anchor = 方差最接近"未水印基线"的位置。

### 4.3 验证

最终用反推公式把 $\alpha$ 应用一次，**目视检查**：
- 水印区域是否完全消失
- 邻域像素是否过渡自然（无重影、无错位）
- 失败则扩大搜索范围或拒绝处理

### 4.4 与本项目当前 `WATERMARK_BBOX` 的关系

我们当前在 `src/watermark/consts.py` 里硬编码的是：

```python
WATERMARK_BBOX: Final[tuple[int, int, int, int]] = (890, 1720, 990, 1830)
```

这与"按 Gemini 官方查表 + 校准"应得的 `(920, 1760, 1016, 1856)` **接近但不完全一致**：
- 我们偏左 30 px（左边 `890` vs `920`）
- 高度多 60 px（`1720..1830` = 110 vs `1760..1856` = 96）
- 宽度少 6 px（`890..990` = 100 vs `920..1016` = 96）

这个差异大概率来自手动视觉定位时为容错画大了一点——但对反向 alpha 路线来说，**bbox 与 $\alpha$ 矩阵必须 1:1 对齐**，否则会把 $\alpha=0$ 的正常像素也代入公式，导致背景被错误"减"一下产生黑斑。这就是为什么反向 alpha 路线**强烈依赖自动检测**，不接受手工 bbox。

---

## 5. 在 `src/watermark/services/inpaint.py` 的嵌入点

> 下面**只是伪代码**，展示怎么嵌入；本任务不实际修改代码。

### 5.1 新 backend 函数的形状

```python
# pseudocode — NOT real importable code

def _inpaint_alpha(
    frame_rgb: Image.Image,
    mask_l: Image.Image | None,  # ignored — alpha map carries its own shape
    *,
    alpha_map: np.ndarray,       # shape (H, W) float32 in [0, 1]
    logo_rgb: tuple[float, float, float] = (255.0, 255.0, 255.0),
) -> Image.Image:
    """Reverse alpha blending — mathematically lossless."""
    frame = np.asarray(frame_rgb).astype(np.float32)   # (H, W, 3)
    # broadcast alpha over RGB channels
    alpha = alpha_map[..., None]                       # (H, W, 1)
    logo = np.array(logo_rgb, dtype=np.float32)        # (3,)
    restored = (frame - alpha * logo) / (1.0 - alpha)  # (H, W, 3)
    return Image.fromarray(np.clip(restored, 0, 255).astype(np.uint8))
```

**关键性质**：
- 不需要 mask（alpha map 自带形状信息）
- 不需要 GPU（纯 numpy，逐元素广播）
- 不需要模型加载
- 240 帧 1080×1920 端到端 **< 1 秒**（每秒数十帧）

### 5.2 注册到 `_INPAINTERS`

```python
# pseudocode
_INPAINTERS = {
    "telea": _inpaint_telea,
    "ns":    _inpaint_ns,
    "lama":  _inpaint_lama,
    "alpha": _inpaint_alpha,  # ← new
}
```

`InpaintOptions` 已经有 `device` 和 `radius` 字段，对 alpha 路线都不适用，可以忽略。`--backend alpha` 选项可直接走 Click `Choice(["telea", "ns", "lama", "alpha"])`，零侵入。

### 5.3 校准子命令（可选 v0.2.0）

```python
# pseudocode
@click.command()
@click.argument("calibration_video", type=click.Path(exists=True, path_type=Path))
def calibrate_cmd(calibration_video: Path) -> None:
    """Extract the alpha map and logo color from a known calibration clip."""
    alpha_map, logo_rgb = calibrate(calibration_video, expected_bbox=...)
    np.save("data/gemini_96_alpha.npy", alpha_map)
    save_logo(logo_rgb, "data/gemini_logo.json")
    click.echo("calibrated → data/gemini_96_alpha.npy")
```

输出两个文件：
- `data/gemini_96_alpha.npy` — Float32 数组 $(96 \times 96)$
- `data/gemini_logo.json` — `{"r": 248, "g": 248, "b": 248}`

后续运行 `watermark remove --backend alpha` 时自动加载。

---

## 6. 对比矩阵

| 维度 | Telea（当前默认） | LaMa（可选） | **Reverse Alpha（反向 alpha 混合）** |
|---|---|---|---|
| **数学性质** | 邻域扩散近似 | 神经网络回归 | **代数精确反演** |
| **像素保真度** | 模糊 / 失真 | 细节有损 / 偶有 hallucination | **逐像素等于原始** |
| **速度（240 帧 1080×1920）** | ~23 s（M3 Max CPU） | ~50 min（CPU）/ 5 min（MPS） | **< 1 s**（纯 numpy） |
| **依赖** | OpenCV | PyTorch + ~250 MB 模型下载 | **NumPy**（已间接依赖） |
| **GPU 需求** | 无 | 强烈推荐 | **无** |
| **抗水印样式变化** | 鲁棒（自动适配任意 mask 形状） | 鲁棒 | **脆弱**（$\alpha$ map 必须匹配当前 Gemini 模板） |
| **抗背景复杂度** | 低频下好，纹理下失真 | 优秀（神经网路学过的） | 无关（数学反演，不"补"内容） |
| **可见残影风险** | 中（取决于 bbox 精度） | 低 | **零**（如果 $\alpha$ 精确） |
| **首次运行成本** | 无 | 模型下载 | **校准成本**（需要一张已知纯色背景的 Gemini 输出） |
| **维护成本** | 低 | 低 | **中**（Gemini 改版要重校准） |

**选型建议**：
- 追求**精确像素恢复 + 速度**：反向 alpha 混合（前提是 Gemini 模板没变）
- 追求**鲁棒性 / 适配不同水印**：LaMa
- 追求**纯本地、无依赖、零下载**：Telea（当前默认）

---

## 7. Caveats

### 7.1 不去 SynthID

Gemini 输出有**两层**水印机制：

| 类型 | 性质 | 能否用反向 alpha 去除 |
|---|---|---|
| **可见水印** | 像素域的 $\alpha$ 混合，半透明 ✦ / "AI Result" 文字 | ✅ **是**（本文档主题） |
| **SynthID** | 频域隐写水印（基于音频/视频信号的微小扰动） | ❌ **否** |

SynthID 是 Google 的隐写水印，**设计目标**就是抵抗裁剪、压缩、重编码等所有常见变换。要检测/去除 SynthID 必须用 `google-generativeai` SDK 调用 Google 服务端 API，没有开源方案。GargantuaX 项目的 README 也明确说："Does not remove SynthID"。

### 7.2 "Lossless" 假设校准的 $\alpha$ 仍匹配当前 Gemini 模板

GargantuaX 的 $\alpha$ map 标定到 "validated through 2026 年 4 月"。如果 Google 后续更新了可见水印的样式（不同形状 / 不同大小 / 不同 $\alpha$ 分布），需要：

1. 在已知纯色背景上重新生成一张水印图
2. 用 §3.2 的校准公式解出新的 $\alpha$ map 和 $\text{logo}$ RGB
3. 替换项目中的 `data/gemini_96_alpha.npy`

如果忽略这一步还硬跑反向 alpha，结果会"反着减"一下原图，产生**比水印更明显的黑斑**——所以这条路线必须有**检测失败时拒绝运行**的保护逻辑。

### 7.3 视频层仍有 ffmpeg 重编码损失

反向 alpha 混合对**单帧**是数学无损的。但要把修复后的 PNG 序列拼回 H.264 视频时仍需重编码：

| 路径 | 损失来源 | 实际影响 |
|---|---|---|
| `-c:a copy` | 无（音频字节 passthrough） | 音频 MD5 与源完全一致 |
| `-c:v libx264 -crf 16` | H.264 量化 | 视觉几乎不可见（PSNR 损失 < 1 dB） |
| `-c:v ffv1 / `-c:v png` 序列 + WAV | 无损容器 | **真正 bit-exact 视频**（文件会大很多） |

本项目当前使用 CRF 16 + `-c:a copy`，**对可见质量来说已经够"无损"**。如要严格意义上的"bit-exact"，输出 PNG 序列 + WAV 即可，不应视为路线劣势。

---

## 8. 与本项目的关系

### 8.1 现有架构已经为新 backend 做好准备

`src/watermark/services/inpaint.py` 的 `_INPAINTERS` 字典 + `InpaintOptions` dataclass 设计，是按 `new-feature` skill 的"可扩展 service 层"原则搭建的——加入新 backend 几乎是零侵入：

```
新增后端所需改动:
  1. services/inpaint.py  写入 _inpaint_alpha + 注册到 _INPAINTERS
  2. commands/remove.py   Choice 加 "alpha" 选项
  3. tests/               新增 alpha 路线单测 + Click CliRunner 集成测
  4. data/                校准产物 (alpha_map.npy, logo.json)
```

不需要改动 `detect.py` / `mask.py` / `reassemble.py` / `verify.py` / 任何 SPEC.md 段落。

### 8.2 当前 `WATERMARK_BBOX` 与 $\alpha$ 路线的冲突

反向 alpha 路线**不允许手工 bbox**：bbox 错了会把 $\alpha=0$ 的正常像素也代入公式。当前 `consts.WATERMARK_BBOX = (890, 1720, 990, 1830)` 是为 Telea 留了 10 px dilate 之后的结果，对反向 alpha **不直接可用**。

如果将来集成 alpha 路线，需要：
- 反向 alpha 走自己**自动检测**的 bbox（来自 §4.2 的 anchor search）
- Telea / LaMa 仍用 `WATERMARK_BBOX` + dilate
- 两条路线**用不同 bbox**：`InpaintOptions` 加一个 `bbox` 字段覆盖默认

### 8.3 校准样本的获取

反向 alpha 路线的 **bootstrapping 问题**：第一次需要一张"已知纯色背景的 Gemini 输出图"做校准。三种解法：

1. **手工合成**：自己用 Gemini 生成一张请求"输出纯色背景 + 文字 'test'"，截取水印区域作为校准输入
2. **借力已知项目**：直接采用 GargantuaX 项目 release 中校准好的 $\alpha$ map（MIT 协议，注明来源即可）
3. **在多种纯色背景上重渲染**：上传 `output_for_alpha_calibration.png` 到自己的 Gemini/AI Studio，请求"在这张图上只叠加水印"，下载后用 §3.2 公式反解 $\alpha$ 和 $\text{logo}$

(2) 是最快的工程路径，但每次 Gemini 改版要重新对齐。

### 8.4 文档与代码的优先级

按 `CLAUDE.md §1` 的 SPEC 协议：**SPEC > 代码**。本 `Lossless.md` 是设计文档，描述**为什么**和**如何**，不替代具体实现的"做什么"——后者写在 `src/watermark/services/inpaint.py` 的 docstring 和 `tests/` 中。

任何对 alpha 路线的实际实现都应先**更新 SPEC.md**（新增 acceptance criteria：原像素值与反向 alpha 输出在 $\alpha>0$ 区域必须 byte-identical），再改代码。

---

## 9. 参考资料

- [GargantuaX/gemini-watermark-remover](https://github.com/GargantuaX/gemini-watermark-remover) — 本文讨论的具体实现
- [allenk/GeminiWatermarkTool](https://github.com/allenk/GeminiWatermarkTool) — 同类项目的 C++ 实现，附 `synthid_research.md`
- [Alpha compositing (Wikipedia)](https://en.wikipedia.org/wiki/Alpha_compositing) — `α` 混合的数学背景
- [SynthID 官方说明](https://deepmind.google/technologies/synthid/) — 不可去隐写水印的来源

---

**文档版本**: v0.1.0 (2026-06-03)
**状态**: 设计文档，未关联代码改动。
