# run_inference 全参数速查

在 `OmniPart` 目录执行，入口为：

```bash
python run_inference.py [参数]
```

## 参数总览

必选（二选一）：
- `--image <path>`：普通图片输入模式（RMBG + SAM，支持手动 merge）
- `--frame_dir <dir>`：预分割输入模式（目录内需 `frame.npy/mask0.npy/mask1.npy`，自动 merge）

可选（两种模式都可用）：
- `--output_dir <dir>`：输出目录，默认 `output/<输入名>`
- `--threshold <int>`：SAM 最小分割面积阈值（默认脚本内 `DEFAULT_SIZE_TH`）
- `--seed <int>`：随机种子（默认 `42`）
- `--cfg <float>`：part synthesis 的 CFG（默认 `7.5`）

仅 `--image` 模式常用：
- `--merge "0,1;3,4"`：手动 merge 组；分号分组，逗号组内 ID
- `--interactive_merge`：先看 SAM 预分割图，再在终端输入 merge 字符串

仅 `--frame_dir` 模式常用：
- `--contain_threshold <float>`：自动 merge 的包含率阈值（默认 `0.3`）

## 两种 mask merge 方法

1) 手动 merge（`--image`）
- 你自己指定 `--merge`，例如 `"0,1;3,4"`。
- 含义：把 `0` 和 `1` 合并一组，把 `3` 和 `4` 合并一组。

2) 自动 merge（`--frame_dir`）
- 脚本会根据 `mask0.npy` / `mask1.npy` 与 SAM 分割结果的包含率自动分组。
- 使用 `--contain_threshold` 控制“是否归入 static/dynamic 组”。

## 命令模板

### A. image + 手动 merge

```bash
python run_inference.py \
  --image assets/example_data/snake.png \
  --output_dir output/snake_manual \
  --threshold 50 \
  --merge "0,1;3,4" \
  --seed 42 \
  --cfg 7.5
```

### A-2. image + 先看图再交互输入 merge（推荐你的场景）

```bash
python run_inference.py \
  --image assets/example_data/snake.png \
  --interactive_merge \
  --output_dir output/snake_interactive \
  --threshold 50 \
  --seed 42 \
  --cfg 7.5
```

运行到中途会提示：
- 预分割图路径：`output/.../*_mask_pre_merge.png`
- 可用 SAM ID 列表
- 终端输入 merge（例：`0,1;3,4`），回车继续；直接回车表示不合并

### B. frame_dir + 自动 merge

```bash
python run_inference.py \
  --frame_dir /path/to/frame_dir \
  --output_dir output/frame_auto \
  --threshold 50 \
  --contain_threshold 0.3 \
  --seed 42 \
  --cfg 7.5
```
