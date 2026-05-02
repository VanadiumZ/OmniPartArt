# 视频标准化与 Mask 帧对齐说明

本文整理 SegAnyMo 相关流程中：**分辨率、帧数、尺寸**如何变化，以及**最终 mask 的每一帧**与**最初输入视频**在时间与空间上的对应关系。依据 `core/utils/run_inference.py`、`sam2/run_sam2.py` 及 SAM2 `init_state` / 输出分辨率逻辑。

---

## 1. 总体结论（先看这里）

| 阶段 | 分辨率 / 尺寸 | 帧数 |
|------|----------------|------|
| 原始输入视频 | 由编码与解码器决定（容器内常见为固定像素宽高） | 由视频时长与 fps 决定 |
| 仅 `--video_path` 抽帧（未开 `-e`） | 与 OpenCV 读出的帧一致，脚本内**不显式缩放** | **全部帧**均导出 |
| `--video_path` + `-e` 抽帧 | 同上（抽帧阶段仍不缩放） | **最多 100 帧**，按间隔均匀抽取 |
| `-e` 下的 `resize_images` | **长边 ≤ 1000**，短边同比缩放（`INTER_AREA`） | **不变**（帧索引与子集与抽帧结果一致） |
| 深度 / 轨迹 / `inference.py` | 各子脚本**可能**另有内部缩放；`run_inference.py` 不再统一改尺寸 | 通常与输入图序列一致（以各脚本为准） |
| SAM2 与保存的 mask | 推理在模型 `image_size` 上完成；**输出的 logits / 保存的 PNG mask** 插值回 **与 `video_dir` 中帧图相同的 H×W** | 与 `video_dir` 中**按文件名排序后的帧列表**一一对应 |
| `run_sam2.py` 生成的视频 | 与 `video_dir` 中逐帧 `cv2.imread` 的 **H×W 一致** | 与上述帧列表长度一致 |
| 若对比「原始 mp4」而非「抽帧目录」 | 若抽帧/缩放改过，则与 mp4 **像素级**可能不一致；播放器 SAR 也会造成「显示比例」与像素网格不一致 | 均匀抽帧时 **时间采样**与原始视频不同 |

---

## 2. 阶段一：`run_inference.py` — 视频转图像（`--video_path`）

**代码位置：** `video_to_images()`，`__main__` 中 `args.video_path is not None` 分支。

### 2.1 输出路径与命名

- 序列名：`seq_name = basename(视频路径)` 去掉扩展名。
- 输出目录：`<视频所在目录>/images/<seq_name>/`。
- 若该目录**已存在**，**不会重新抽帧**（需注意增量/脏数据）。

### 2.2 分辨率与尺寸

- 使用 `cv2.VideoCapture` + `cap.read()`，将当前帧 **原样** 写入 `00000.png`, `00001.png`, …。
- **本脚本不对帧做 resize**；像素宽高 = 解码器给出的该帧数组形状（通常与视频「编码分辨率」一致，除非源文件或解码有特殊行为）。

### 2.3 帧数

- **未开启 `-e`（`args.e == False`）**：`target_frames = total_frames`，逐帧保存，**帧数 = 视频总帧数**（以 `CAP_PROP_FRAME_COUNT` 为准，可能与部分容器有轻微偏差）。
- **开启 `-e`（efficiency）**：`target_frames = min(total_frames, 100)`，按 `frame_interval` **均匀间隔**取帧，**最多 100 张图**；**时间轴被稀疏化**，与原始视频「每一物理帧」不再一一对应。

### 2.3 与 `args.data_dir` 的关系

- 抽帧后：`args.data_dir = <视频父目录>/images`。
- 后续每个序列的实际图像目录：`img_dir = <args.data_dir>/<seq_name>/`。

---

## 3. 阶段二：`run_inference.py` — 高效模式下的图像缩放（`-e` / `--e`）

**代码位置：** `resize_images()`，`__main__` 中 `if args.e:`。

### 3.1 何时触发

- 仅在命令行传入 **`-e`** 时执行。
- 若未从 `video_path` 抽帧而是直接指向已有 `images` 结构，同样会对 **`args.data_dir` 下各序列子文件夹** 做一遍缩放输出。

### 3.2 分辨率与尺寸

- 对每个序列子目录中的每张图：若 `max(h, w) > 1000`，则按比例缩放使 **长边 = 1000**（短边等比）；否则 **保持原尺寸**。
- 插值：`cv2.INTER_AREA`。

### 3.3 输出与后续数据根目录

- 输出根目录：`<dirname(args.data_dir)>/resize_images/`，内部仍按 **序列名子目录** 镜像原结构。
- 随后 **`args.data_dir` 被替换为 `resize_images`**，后续深度、轨迹、推理、SAM2 若由此入口调度，则统一使用 **缩放后的图序列**。

### 3.4 帧数

- **不删帧、不增帧**；每个序列内 **PNG 数量与抽帧/原图数量一致**，仅改变每张图的像素尺寸。

---

## 4. 阶段三：`main()` — 深度、轨迹、运动分割、SAM2（对「图序列」的共用约定）

**代码位置：** `core/utils/run_inference.py` 中 `main()`。

- 各任务通过 **`img_dir`** 读取图像；**不再读取原始 mp4**。
- **分辨率**：除阶段二的 `-e` 缩放外，`run_inference.py` **不再**统一改宽高；`run_depth.py`、`run_tapir.py`、`inference.py` 等 **可能**在各自内部有 resize（需查对应脚本，本文以 orchestrator 为准）。
- **帧数**：各模块通常按 `img_dir` 内帧数与排序一致消费；若某子脚本按 `step` 下采样，则其内部有效帧可能变少（以该脚本参数为准）。

---

## 5. SAM2 与 Mask：`run_sam2.py` + SAM2 预测器

### 5.1 输入帧来源

- `--video_dir` 指向与上游一致的 **帧图目录**（例如 `.../images/<seq>` 或 `.../resize_images/<seq>`）。
- 帧顺序：`frame_names` 为目录下 `.jpg/.jpeg/.png` 等扩展名文件 **按名字字符串排序** 后的基名列表（**不是**按整数帧号排序，除非文件名本身零填充且排序与时间一致）。

### 5.2 模型内部 vs 输出 mask 尺寸

- `init_state` 内 `load_video_frames` 会将帧 **resize 到模型 `image_size`** 做特征与跟踪。
- `inference_state` 中保存 **`video_height`, `video_width`**（来自加载流程记录的**原始帧图**宽高）。
- `propagate_in_video` 等返回给用户侧的结果经 **`_get_orig_video_res_output`**：将 mask **双线性插值**到 **`(video_height, video_width)`**，与 **`video_dir` 中对应帧图的像素网格一致**。

### 5.3 保存的 mask PNG 与索引

- mask 按 `frame_names[i]` 等与 **排序后的第 i 张图** 对齐。
- **空间上**：第 `i` 帧 mask 的 **高×宽** = 该帧在 `video_dir` 中对应图像的 **高×宽**（与 SAM2 记录的 `video_height`×`video_width` 一致，逐序列通常各帧同分辨率）。
- **时间上**：第 `i` 个 mask 对应排序后第 `i` 个文件名；若上游曾用 `-e` 抽帧，则 **i 不对应原始 mp4 的第 i 个解码帧**，而是对应 **稀疏采样后的第 i 张图**。

### 5.4 `run_sam2.py` 输出视频（如 `combined_mask_rgb_color.mp4`）

- RGB：`cv2.imread(video_dir 下各帧)`，**无额外缩放**。
- 与 `all_dyn_mask` / `all_static_mask` 等逐帧 zip；**输出视频分辨率 = `rgbs[0].shape` 的 H×W**，即与 **`video_dir` 首帧（排序后）图像一致**。
- **帧数**：等于 `frame_names` 长度，与 mask 序列长度一致。

---

## 6. 「Mask 第 i 帧」vs「最初输入视频」— 对照清单

以下「最初输入视频」指你作为 `--video_path` 传入的 **mp4 等文件**；若你从未用该入口、而是直接提供帧文件夹，则把「视频」理解为**你指定的原始帧序列**即可。

| 维度 | 无 `-e` 抽帧 | 有 `-e` 抽帧 | 有 `-e` 且经过 `resize_images` |
|------|----------------|----------------|--------------------------------|
| **与原始视频的帧索引** | 可视为 **0…N-1 一一对应**（以 OpenCV 帧计数与抽帧循环为准） | **不一一对应**；仅 **最多 100 个采样时刻** | 同左列；再在空间上缩小 |
| **Mask 帧索引 i** | 对应 **排序后第 i 张** `video_dir` 图像 | 对应 **稀疏采样后**排序第 i 张 | 同左；且图像为 **长边≤1000** 版本 |
| **Mask 与 RGB 的 H×W** | 与 **当前 `video_dir`** 中该帧图像 **相同** | 同左 | 与 **resize 后** 图 **相同**；通常 **小于或等于** 抽帧后未缩放尺寸 |
| **与原始视频 H×W** | 与解码帧一致（未再缩放） | 解码帧一致（抽帧阶段未缩放） | **可能小于**原始解码帧（长边限制 1000） |
| **与原始视频「显示尺寸」** | 若容器含 SAR，播放器显示可能与像素网格比例不同；pipeline 全程按 **像素** 处理 | 同左 | 同左 |

---

## 7. 实践建议

1. **要求 mask 与某段 mp4 逐帧、同分辨率对齐**：避免使用 `-e` 抽帧；或自行抽帧并保证 `video_dir` 与 mp4 帧序、分辨率一致后再跑 SAM2。
2. **需要与原始 4K 等完全同尺寸 mask**：不要使用会触发 `resize_images` 的 `-e`；并确认 `video_dir` 即为全分辨率序列。
3. **帧序依赖文件名排序**：若使用 `1.jpg, 2.jpg, …, 10.jpg` 等，应用零填充（`00001.jpg`）以免排序错乱。
4. **输出 MP4 与原始 MP4**：空间分辨率以 **`video_dir` 图像** 为准；时间轴以 **帧图数量与排序** 为准；**fps**（如脚本中默认 30）可能与源视频不同，仅影响播放快慢观感，不改变每帧像素内容对应关系。

---

## 8. 相关文件索引

| 文件 | 与本文相关的逻辑 |
|------|------------------|
| `core/utils/run_inference.py` | `video_to_images`、`resize_images`、`args.data_dir` 切换、`main()` 中各子命令的 `img_dir` |
| `sam2/run_sam2.py` | `cv2.imread` 组 `rgbs`、mask 与 `frame_names` 对齐、写视频 |
| `sam2/sam2/sam2_video_predictor.py` | `init_state` 中 `video_height`/`video_width`；`_get_orig_video_res_output` 将 mask 还原到原视频分辨率 |
| `sam2/sam2/utils/misc.py` | `load_video_frames`：内部 resize 到 `image_size`，并记录原始尺寸 |

---

*文档根据当前仓库代码整理；若子脚本（如 `run_depth.py`、`run_tapir.py`）内部另有缩放或跳帧，需结合各自参数单独核对。*
