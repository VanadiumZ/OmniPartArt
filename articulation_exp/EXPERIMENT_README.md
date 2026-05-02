# Articulated Joint Estimation: 实验框架文档

## Video Generator + 3D Part Prior for Articulated Object Joint Estimation

---

## 一、项目概览

本实验框架以 **hook 机制** 扩展 OmniPart，结合 PAct 的关节回归思路，实现了一套完整的铰接关节估计实验系统。核心不修改 OmniPart 主仓库代码，所有扩展通过外部模块和 PyTorch `register_forward_hook` 实现。

### 核心思路

```
OmniPart / TRELLIS → 3D part geometry anchor (SLAT tokens, bbox, mesh)
Wan / Video-DiT    → motion prior & temporal correspondence
Trainable Fusion   → fast joint initialization
Training-Free Fit  → physical consistency refinement
```

### 三条实验路线

| 路线 | 名称 | 目标 | 训练? |
|------|------|------|-------|
| Route A | Trainable Latent Fusion | 验证 Wan + 3D fusion 能否读出 articulation | 是 |
| Route B | Training-Free Fitting | frozen prior + per-instance optimization | 否 |
| Route C | Video-Generator Trajectory | 从 DiT 内部提取 tracks/flow | 否 |

---

## 二、目录结构

```
articulation_exp/
├── __init__.py
├── EXPERIMENT_README.md         # 本文档
├── run_experiments.sh           # 一键实验运行脚本
│
├── configs/                     # 所有配置文件
│   ├── dataset/
│   │   └── partnet_mobility.yaml
│   └── model/
│       ├── baseline_3d.yaml         # E1-A
│       ├── baseline_video.yaml      # E1-B
│       ├── late_fusion.yaml         # E1-C
│       ├── cross_fusion.yaml        # E2
│       ├── projective_fusion.yaml   # E3
│       └── training_free.yaml       # TF0/TF1/TF2
│
├── data/                        # 数据集定义
│   ├── __init__.py
│   ├── dataset_articulation.py  # ArticulationDataset
│   └── build_dataset.py         # DataLoader 构建
│
├── cache/                       # 特征缓存（hook-based）
│   ├── __init__.py
│   ├── cache_omnipart.py        # OmniPart SLAT/bbox 特征提取
│   └── cache_wan_features.py    # Wan VAE/DiT 特征提取
│
├── models/                      # 所有模型架构
│   ├── __init__.py              # build_model() 工厂函数
│   ├── joint_decoder.py         # 通用关节参数解码头
│   ├── part_slat_pool.py        # SLAT token 按 part 池化
│   ├── video_pooler.py          # 视频特征池化（mean/Perceiver）
│   ├── baseline_3d.py           # E1-A: 仅3D先验
│   ├── baseline_video.py        # E1-B: 仅视频先验
│   ├── fusion_late.py           # E1-C: 晚期融合
│   ├── fusion_cross_attn.py     # E2: Cross-Attention 融合（主线）
│   └── fusion_projective.py     # E3: 投影式融合
│
├── losses/                      # 损失函数
│   ├── __init__.py
│   ├── joint_losses.py          # 关节参数损失（axis/pivot/state/type）
│   ├── render_losses.py         # 渲染mask IoU / silhouette / collision
│   └── track_losses.py          # Track reprojection / smoothness
│
├── train/                       # 训练入口
│   ├── __init__.py
│   ├── trainer.py               # ArticulationTrainer 统一训练器
│   └── train_all.py             # CLI 入口
│
├── training_free/               # Training-Free 分支
│   ├── __init__.py
│   ├── moving_part_proposal.py  # TF0-1: 运动部件识别
│   ├── joint_candidate_generator.py  # TF0-2: 候选关节生成
│   ├── candidate_scorer.py      # TF0-3/TF1: 候选评分
│   ├── kinematic_refinement.py  # TF2: 运动学优化
│   └── run_training_free.py     # CLI 入口
│
├── video_generator_motion/      # Video Generator Trajectory 分支
│   ├── __init__.py
│   ├── dit_correspondence.py    # VGT0: DiT 时间对应提取
│   ├── kl_tracing_flow.py       # VGT1: KL-tracing flow
│   └── track_filtering.py       # Track 过滤与质量评估
│
└── eval/                        # 评估模块
    ├── __init__.py
    └── eval_joint_metrics.py    # 关节估计指标计算
```

---

## 三、Hook 机制详解

本框架的核心设计原则是 **不修改 OmniPart 主仓库代码**，通过 PyTorch 的 `register_forward_hook` 提取中间特征。

### 3.1 OmniPart SLAT 特征提取

```python
from articulation_exp.cache.cache_omnipart import OmniPartHookManager

hook_mgr = OmniPartHookManager()

# 注册 hook 到 SLAT flow model
model = pipeline.models["slat_flow_model"]
hook_mgr.register(model, "slat_flow_out")

# 推理后，从 cache 获取特征
slat_feats = hook_mgr.cache["slat_flow_out"]["feats"]  # [N, C]
slat_coords = hook_mgr.cache["slat_flow_out"]["coords"]  # [N, 4]

# 清理
hook_mgr.remove_all()
```

### 3.2 Wan DiT 特征提取

```python
from articulation_exp.cache.cache_wan_features import WanFeatureExtractor

extractor = WanFeatureExtractor(wan_model=model, vae=vae)
extractor.register_hooks(
    layer_preset="middle",      # 中间层（12/16层）
    timestep_preset="mid_noise" # 中噪声时间步（400-500）
)

vae_latent = extractor.extract_vae_latent(video_frames)
dit_features = extractor.extract_dit_features(vae_latent)
extractor.save_features(dit_features, save_dir)
```

### 3.3 Hook 设计参考

Hook 机制参考了 OmniPart 官方文档 `docs/intermediate_latents_and_hooks.md` 中的指引，主要 hook 点包括：

| Hook 目标 | 获取内容 | 用途 |
|-----------|----------|------|
| `slat_flow_model` | SLAT tokens ([N, C]) | 3D part 特征 |
| `sparse_structure_decoder` | 体素占据场 | 稀疏结构验证 |
| `image_cond_model` (DINOv2) | patch tokens | 图像条件特征 |
| Wan DiT blocks[8/16/24] | 隐藏状态 | 视频 motion evidence |
| Wan VAE encoder | latent | 低级视频表征 |

---

## 四、各实验详细说明

### 4.1 E0: 数据准备与特征缓存

**目标**：将 OmniPart/TRELLIS/Wan 的输出全部缓存，避免训练时反复调用大模型。

**缓存目录结构**：
```
data_cache/partnet_mobility/
  {category}/{object_id}/
    gt_joint.json           # GT 关节标签
    omnipart/
      slat_feats.pt         # [N3, C3] SLAT token features
      slat_xyz.pt           # [N3, 3] token 3D positions
      slat_part_ids.pt      # [N3] part assignment
      part_bboxes.npy       # [P, 6] part bounding boxes
    wan/
      vae_latent.pt         # VAE latent
      dit_l16_mid.pt        # DiT middle layer features
    motion/
      seganymotion_masks.npy
      seganymotion_tracks.npy
```

**GT JSON 格式**：
```json
{
  "joint_type": "revolute",
  "joint_axis": [0, 1, 0],
  "joint_pivot": [0.1, 0.2, 0],
  "joint_state": [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
  "moving_part_id": 1,
  "num_parts": 3
}
```

### 4.2 E1: 基础 Baselines

#### E1-A: 3D-only baseline
- **输入**: OmniPart SLAT tokens + part bbox
- **模型**: SLAT pooling → part Transformer → JointDecoder
- **目的**: 回答 "3D prior alone 能提供多少 articulation 信息？"

#### E1-B: Video-only baseline
- **输入**: Wan VAE latent 或 DiT hidden states
- **模型**: temporal-spatial pooling → MLP → JointDecoder
- **目的**: 回答 "video prior alone 是否编码了 motion/joint 信息？"

#### E1-C: Late fusion baseline
- **输入**: global video token + global 3D token
- **模型**: concat/add/gated fusion → MLP → JointDecoder
- **目的**: 验证简单融合是否优于单模态

**成功标准**：
```
late fusion > 3D-only
late fusion > video-only
DiT hidden state > VAE latent
```

### 4.3 E2: Video-to-SLAT Cross-Attention Fusion（主线）

**核心思想**: 让每个 3D part token 主动从 video tokens 中读取 motion evidence。

```
SLAT tokens (Q) × Video tokens (K, V) → Gated Cross-Attention → Fused Part Tokens → JointDecoder
```

**架构**:
- SLAT tokens → 3D positional encoding → part embedding → Q
- Video tokens → temporal pos enc → 2D pos enc → K, V
- Cross-Attention → gated residual → fused tokens
- Part pooling → Joint decoder

**为什么是主线**: Articulation 是 part-level 问题，global concat 太粗糙。Part-level 3D token + video motion evidence → part-level joint 更合理。

### 4.4 E3: Projective Video-to-SLAT Fusion

**核心思想**: 有 camera pose 时，用 3D→2D 投影显式建立 video feature 与 SLAT token 的对应。

```
For each 3D point x_i:
  u_i^t = Π(K, T_cam^t, x_i)           # 投影到帧 t
  f_video = GridSample(F_t, u_i^t)      # 采样视频特征
  f_fused = MLP([f_slat, f_video])      # 融合
```

**优势**: 比 cross-attention 的隐式 alignment 更精确，理论上应在 pivot/state 估计上更优。

### 4.5 TF0: Training-Free Geometry+Motion Candidate Fitting

**无需训练**。用 frozen OmniPart geometry + motion cues 搜索最佳 joint。

**流程**:
1. **Moving part proposal**: IoU(projected_part_mask, motion_mask)
2. **Candidate generation**: 枚举 revolute/prismatic 的 axis + pivot 候选
3. **Candidate scoring**: E_mask + E_track + E_collision + E_range + E_smooth
4. **Top-k selection**

**候选来源**:
- Revolute axis: 坐标轴、PCA轴、接触边界切线
- Revolute pivot: 接触区域中心、bbox 边缘中心
- Prismatic axis: track 平均位移方向、PCA 方向

### 4.6 TF1: Wan Feature Candidate Scoring

在 TF0 基础上增加 **Wan feature score**: 比较真实视频和候选动画视频的 Wan 特征相似度。

```
E_total = λ_geom × E_geom + λ_track × E_track + λ_wan × E_wan + ...
```

### 4.7 TF2: FreeArt3D-Lite Kinematic Refinement

固定 OmniPart geometry，仅优化 joint 参数：

```python
params = {
    "raw_axis": Tensor[3],         # 可优化
    "pivot": Tensor[3],            # 可优化
    "joint_state": Tensor[T],      # 可优化
}

L = λ_mask × L_mask + λ_track × L_track + λ_collision × L_collision + λ_smooth × L_smooth
```

### 4.8 VGT0: Video-DiT Correspondence

从 Wan/Video-DiT 内部特征中提取 temporal correspondence：

```
Query point p_0 in frame 0
→ find corresponding DiT token
→ feature similarity across frames
→ soft-argmax → trajectory p_t
```

可替代 SegAnyMotion 的外部 track 输出。

### 4.9 E4: Fusion Initialization + Refinement

**最终系统**：trainable fusion (E2/E3) 提供快速初始化 → training-free (TF2) 提供物理一致性修正。

---

## 五、损失函数

### 总损失
```
L = λ_part × L_part + λ_type × L_type + λ_axis × L_axis + λ_pivot × L_pivot + λ_state × L_state
```

### 各项损失

| 名称 | 公式 | 说明 |
|------|------|------|
| L_part | CE(pred, gt) | Moving part 分类 |
| L_type | CE(pred, gt) | Joint type 分类（fixed/revolute/prismatic） |
| L_axis | min(angle(u, u_gt), angle(-u, u_gt)) | 方向无关的轴角误差 |
| L_pivot | \|\|p_pred - p_gt\|\|_1 | 可 bbox 归一化 |
| L_state_rev | \|\|sin(q)-sin(q_gt)\|\|_1 + \|\|cos(q)-cos(q_gt)\|\|_1 | 旋转关节状态 |
| L_state_pri | \|\|d-d_gt\|\|_1 | 平移关节位移 |
| L_render | 1 - IoU(rendered_mask, gt_mask) | 渲染 mask (TF2/E4) |
| L_track | mean \|\|proj(T(q_t)x_k) - track_{k,t}\|\|_1 | Track 重投影 (TF2) |
| L_collision | relu(threshold - min_dist) | 碰撞惩罚 |
| L_smooth | (q_{t+1} - q_t)^2 | 状态平滑 |

---

## 六、评估指标

| 指标 | 计算方式 | 方向 |
|------|----------|------|
| Moving Part Acc | Top-1 accuracy | ↑ |
| Joint Type Acc | Classification accuracy | ↑ |
| Axis Err (°) | min(angle(u, u_gt), angle(-u, u_gt)) | ↓ |
| Pivot Err | L2 distance | ↓ |
| State Err | sin/cos L1 (revolute) or displacement L1 (prismatic) | ↓ |
| Render IoU | IoU(rendered_mask, gt_mask) | ↑ |

---

## 七、运行指南

### 7.1 环境准备

本框架不引入新的外部依赖，使用 OmniPart 原有环境即可。额外需要：

```bash
pip install pyyaml scipy
```

### 7.2 数据准备

1. 准备 PartNet-Mobility 渲染视频，建议先选 2-3 类（laptop, microwave, drawer），每类 10-20 个 objects
2. 运行 OmniPart 推理获取 part mesh / bbox / SLAT tokens
3. 提取 Wan/Video-DiT 特征
4. 将所有结果按上述目录结构缓存

### 7.3 运行全部实验

```bash
# 运行所有阶段
bash articulation_exp/run_experiments.sh --all

# 只运行某个阶段
bash articulation_exp/run_experiments.sh --stage e1    # baselines
bash articulation_exp/run_experiments.sh --stage e2    # cross-attention
bash articulation_exp/run_experiments.sh --stage tf0   # training-free
bash articulation_exp/run_experiments.sh --stage tf2   # refinement
bash articulation_exp/run_experiments.sh --stage eval  # 评估
```

### 7.4 单独训练某个模型

```bash
python -m articulation_exp.train.train_all \
    --model_config articulation_exp/configs/model/cross_fusion.yaml \
    --dataset_config articulation_exp/configs/dataset/partnet_mobility.yaml \
    --output_dir outputs/articulation/e2_cross_fusion \
    --device cuda
```

### 7.5 单独运行 Training-Free

```bash
python -m articulation_exp.training_free.run_training_free \
    --config articulation_exp/configs/model/training_free.yaml \
    --data_dir data_cache/partnet_mobility \
    --output_dir outputs/training_free \
    --stage tf2
```

### 7.6 评估已训练模型

```bash
python -m articulation_exp.train.train_all \
    --model_config articulation_exp/configs/model/cross_fusion.yaml \
    --output_dir outputs/articulation/e2_cross_fusion \
    --eval_only \
    --checkpoint outputs/articulation/e2_cross_fusion/best.pt
```

---

## 八、两天冲刺方案

### Day 1

| 任务 | 操作 |
|------|------|
| 选类 | laptop (revolute), drawer (prismatic), microwave (revolute) |
| 数据 | 每类 10-20 objects, 8 frames |
| 缓存 | OmniPart part mesh/bbox, simple SLAT features, Wan VAE latent |
| 训练 | E1-A (3D-only), E1-B (video-only), E1-C (late fusion) |

### Day 2

| 任务 | 操作 |
|------|------|
| E2 | 实现并训练 simplified cross-attention |
| TF0 | 实现 moving part + candidate enumeration + scoring |
| 出表 | 5 个方法的完整对比结果表 |

### 需要回答的 4 个关键问题

1. **video + 3D 是否超过 3D-only？** → 验证 motion evidence 的价值
2. **cross-attention 是否超过 late fusion？** → 验证 part-level query 的优越性
3. **TF0 candidate fitting 是否能给出合理 axis？** → 验证 geometry prior 的直接可用性
4. **Wan DiT feature 是否比 VAE latent 更有用？** → 验证 video generator hidden states 的信息量

---

## 九、预期结果表模板

| Method | Motion Source | 3D Prior | Train? | Refine? | Part Acc↑ | Type Acc↑ | Axis Err↓ | Pivot Err↓ | State Err↓ |
|--------|-------------|----------|--------|---------|-----------|-----------|-----------|------------|------------|
| 3D-only | none | SLAT | ✓ | ✗ | | | | | |
| video-only | Wan latent | none | ✓ | ✗ | | | | | |
| late fusion | Wan latent | SLAT | ✓ | ✗ | | | | | |
| cross-attn | Wan tokens | SLAT | ✓ | ✗ | | | | | |
| projective | proj. Wan | SLAT | ✓ | ✗ | | | | | |
| TF0 | motion mask | mesh | ✗ | cand. | | | | | |
| TF1 | mask+Wan | mesh | ✗ | cand. | | | | | |
| TF2 | tracks+Wan | mesh | ✗ | optim. | | | | | |
| E4 | fusion+ref | SLAT+mesh | ✓ | optim. | | | | | |

---

## 十、与原仓库的关系

### 不修改的部分
- `modules/part_synthesis/` - 完整的 OmniPart 推理栈
- `modules/bbox_gen/` - BBox 生成模块
- `modules/inference_utils.py` - 推理工具
- `scripts/inference_omnipart.py` - 推理脚本
- `training/` - OmniPart 训练代码

### 通过 Hook 扩展的部分
- `slat_flow_model` → 提取 SLAT tokens
- `sparse_structure_decoder` → 提取体素占据
- `image_cond_model` → 提取 DINO patch features

### PAct 参考的部分
- `ArticulationRegressionHead` 的关节回归设计
- `articulation_utils.py` 的关节变换与动画逻辑
- `convert_data_range` 的输出后处理方式
- 关节类型定义 (fixed/revolute/prismatic/continuous)

---

## 十一、扩展方向

1. **TF3 (PAct-style denoising descriptor)**: Hook OmniPart denoising 过程，缓存多步 part token features
2. **VGT1 (KL-tracing flow)**: 扰动式 flow 提取
3. **E5 (Video-SDS)**: 用 Wan denoising score 优化 articulated mesh
4. **Multi-part**: 扩展到多运动部件的场景
5. **Real video**: 在真实视频上的定性评估

---

*文档最后更新: 2026-04-26*
