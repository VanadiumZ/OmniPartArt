# Video Generator + 3D Part Prior for Articulated Object Joint Estimation  
# 完整实验设计文档

## 0. 一句话主线

我们要做的不是简单把 **video latent** 和 **3D latent** 拼接，而是构建一个以 **3D part latent 为 geometry anchor**、以 **video generator latent / trajectory 为 motion evidence** 的 articulated joint estimation 框架。

最终系统分成三条互相支撑的路线：

```text
Route A: Trainable Latent Fusion
Wan / Video-DiT feature + OmniPart / TRELLIS SLAT
→ cross-modal fusion
→ fast joint initialization

Route B: Training-Free Video-Prior-Guided Fitting
OmniPart part mesh + video motion cues + Wan feature score
→ joint candidate search
→ per-instance kinematic optimization

Route C: Video-Generator Trajectory Branch
Wan / Video-DiT internal correspondence / KL-tracing flow
→ replace or reduce SegAnyMotion
→ simplify final backbone
```

最终最理想的系统是：

```text
OmniPart / TRELLIS gives part-level 3D geometry.
Video generator gives motion prior and temporal correspondence.
Trainable fusion gives fast initialization.
Training-free fitting gives physical consistency and refinement.
```

---

## 1. 研究问题定义

### 1.1 输入

```text
Input:
  - monocular RGB video V = {I_t}_{t=1}^{T}
  - optional first-frame object crop
  - optional camera pose / intrinsics for synthetic data
  - optional segmentation / tracking results
```

### 1.2 输出

```text
Output:
  - moving part id
  - static parent part id
  - joint type: fixed / revolute / prismatic
  - joint axis u ∈ R^3
  - joint pivot / origin p ∈ R^3
  - per-frame joint state q_t
  - optional articulated part mesh sequence
```

### 1.3 核心假设

```text
H1: 3D generator prior provides geometry anchor.
    OmniPart / TRELLIS gives part decomposition, part mesh, part bbox, SLAT tokens.

H2: Video generator prior provides motion evidence.
    Wan / Video-DiT hidden states may encode temporal correspondence, motion layout, part dynamics.

H3: Joint estimation should be part-centric, not global.
    Each 3D part token should query video motion evidence.

H4: Training-free refinement is necessary.
    Feed-forward prediction gives initialization; explicit kinematic optimization improves pivot, axis, and state consistency.

H5: Video generator may replace external tracking.
    DiffTrack / KL-tracing style methods suggest video generators can expose correspondence or flow-like signals.
```

---

## 2. 总体方法框架

## 2.1 Main System Overview

```text
RGB video
  │
  ├── Wan / Video-DiT feature extractor
  │       ├── VAE latent
  │       ├── DiT hidden states
  │       ├── attention correspondence
  │       └── optional generator-derived flow / tracks
  │
  ├── OmniPart / TRELLIS 3D part prior
  │       ├── part mesh
  │       ├── part bbox
  │       ├── part id
  │       ├── SLAT tokens
  │       └── optional denoising part-token features
  │
  ├── Trainable Fusion Branch
  │       ├── late fusion baseline
  │       ├── SLAT-query-video cross-attention
  │       └── projective video-to-SLAT fusion
  │
  ├── Training-Free Fitting Branch
  │       ├── moving part proposal
  │       ├── joint candidate generation
  │       ├── video / mask / track / Wan feature scoring
  │       └── continuous kinematic refinement
  │
  └── Output
          ├── moving part
          ├── joint type
          ├── axis
          ├── pivot
          └── joint state sequence
```

---

## 3. 实验路线总览

## 3.1 三条实验路线

| Route | Name | Purpose | Output |
|---|---|---|---|
| A | Trainable Latent Fusion | 验证 Wan latent / hidden states 是否能被轻量模型读出 articulation 信息 | fast joint initialization |
| B | Training-Free Fitting | 不训练新模型，用 frozen prior + per-instance optimization 估计 joint | zero-shot joint estimation |
| C | Video-Generator Trajectory | 从 video generator 内部提取 tracks / flow，替代 SegAnyMotion | simplified motion backbone |

---

## 3.2 最终优先级

```text
P0: 必做，决定方向是否成立
  E0: data and feature cache
  E1: 3D-only / video-only / late-fusion baselines
  E2: video-to-SLAT cross-attention
  TF0: geometry + motion candidate fitting

P1: 主线增强，决定是否有论文亮点
  TF1: Wan feature candidate scoring
  TF2: continuous kinematic refinement
  VGT0: Video-DiT correspondence extraction
  E3: projective video-to-SLAT fusion

P2: 高级增强，作为最终系统或 ablation
  E4: fusion initialization + training-free refinement
  VGT1: KL-tracing / counterfactual flow
  TF3: PAct-style denoising feature proposal

P3: 风险较高，放到最后
  E5: video-SDS / Wan denoising score optimization
  TF4: geometry residual optimization
```

---

# Part I. Data Preparation

## 4. 数据准备

## 4.1 首选数据

| Priority | Dataset | Usage |
|---|---|---|
| P0 | PartNet-Mobility rendered videos | controlled synthetic training / evaluation |
| P0 | video2articulation / iTACO-like synthetic data | video-to-joint validation |
| P1 | ACD / articulated object datasets | cross-dataset generalization |
| P2 | real captured monocular videos | qualitative demo |

---

## 4.2 推荐初始类别

第一阶段不要铺太大，先选 5-6 类最容易体现 articulation 的类别：

```text
1. laptop
2. microwave
3. refrigerator
4. cabinet / drawer
5. dishwasher
6. safe / oven
```

优先选：

```text
- single movable part
- clear revolute or prismatic motion
- clean object mask
- camera pose available
- GT joint label available
```

---

## 4.3 视频渲染设置

### Stage 1: fixed camera

```text
frames: 8 or 16
resolution: 256 or 512
camera: fixed
motion: one full open / close trajectory
background: clean
lighting: fixed
```

### Stage 2: mild camera motion

```text
frames: 16
camera: small orbit / small translation
motion: same articulation
background: clean or simple
```

### Stage 3: real-video-like

```text
frames: 16-32
camera: handheld / unknown
background: cluttered
motion: partial opening
occlusion: optional
```

---

## 4.4 每个样本需要保存的内容

```python
sample = {
    "object_id": str,
    "category": str,

    # video
    "frames": Tensor[T, H, W, 3],
    "object_masks": Tensor[T, H, W],
    "camera_intrinsics": Tensor[3, 3],
    "camera_extrinsics": Tensor[T, 4, 4],

    # GT articulation labels
    "joint_type": str,             # fixed / revolute / prismatic
    "joint_axis": Tensor[3],
    "joint_pivot": Tensor[3],
    "joint_state": Tensor[T],
    "moving_part_id": int,
    "static_parent_id": int,

    # OmniPart / TRELLIS outputs
    "part_meshes": List[Mesh],
    "part_bboxes": Tensor[P, 6],
    "part_masks_2d": Tensor[P, H, W],
    "slat_feats": Tensor[N3, C3],
    "slat_xyz": Tensor[N3, 3],
    "slat_part_ids": Tensor[N3],

    # Wan / video generator features
    "wan_vae_latent": Tensor,
    "wan_dit_features": Dict[str, Tensor],

    # optional motion cues
    "seganymotion_masks": Tensor[T, H, W],
    "seganymotion_tracks": Tensor[N, T, 2],
    "generator_tracks": Tensor[N, T, 2],
    "generator_flow": Tensor[T-1, H, W, 2],
}
```

---

## 4.5 Cache 目录设计

```text
data_cache/
  partnet_mobility/
    category/
      object_id/
        frames.mp4
        frames/
        gt_joint.json
        cameras.npz
        omnipart/
          part_meshes/
          part_bboxes.npy
          part_masks.npy
          slat_feats.pt
          slat_xyz.pt
          slat_part_ids.pt
        wan/
          vae_latent.pt
          dit_l8_high.pt
          dit_l16_high.pt
          dit_l24_high.pt
          dit_l16_mid.pt
        motion/
          seganymotion_masks.npy
          seganymotion_tracks.npy
          generator_tracks.npy
          generator_flow.npy
```

---

# Part II. Trainable Latent Fusion Branch

## 5. E0: Feature Cache

### Goal

先把 OmniPart / TRELLIS / Wan 全部 cache 下来，避免训练时反复调用大模型。

### Steps

```text
1. Render PartNet-Mobility videos.
2. Run OmniPart to obtain part meshes, part boxes, and part masks.
3. Extract TRELLIS / OmniPart SLAT tokens if accessible.
4. Run Wan / Video-DiT feature extractor.
5. Hook selected DiT layers and denoising timesteps.
6. Save everything to cache.
```

### Wan feature variants

```text
V0: Wan VAE latent only
V1: Wan DiT shallow hidden states
V2: Wan DiT middle hidden states
V3: Wan DiT deep hidden states
V4: high-noise timestep features
V5: mid-noise timestep features
V6: high + mid concatenation
```

### Expected output

```text
A stable cached dataset where each sample has:
  - 3D part tokens
  - video tokens
  - GT joint labels
  - optional motion tracks / masks
```

---

## 6. E1: Basic Baselines

### Goal

先回答最基础的问题：

```text
3D prior alone works?
Video prior alone works?
Simple fusion works?
```

### E1-A: 3D-only baseline

```text
Input:
  OmniPart / TRELLIS SLAT tokens
  part bbox
  part id

Model:
  SLAT pooling
  part-level Transformer / MLP
  joint decoder

Output:
  moving part
  joint type
  axis
  pivot
  q_t
```

### E1-B: video-only baseline

```text
Input:
  Wan VAE latent or DiT hidden states

Model:
  temporal-spatial pooling
  Perceiver pooling or Transformer
  joint decoder

Output:
  joint type
  axis
  pivot
  q_t
```

### E1-C: late-fusion baseline

```text
Input:
  global video token
  global 3D token

Model:
  concat / add / gated fusion
  MLP decoder

Output:
  moving part
  joint type
  axis
  pivot
  q_t
```

### Success criterion

```text
late fusion > 3D-only
late fusion > video-only
DiT hidden state > VAE latent
```

如果 E1-C 没有超过 E1-A / E1-B，优先检查：

```text
1. label normalization
2. coordinate system alignment
3. part id mapping
4. camera pose correctness
5. video feature extraction layer
6. train / val category split
```

---

## 7. E2: Video-to-SLAT Cross-Attention Fusion

### Goal

这是 trainable branch 的主线。让每个 3D part / SLAT token 主动从 video token 里读取 motion evidence。

### Core idea

```text
SLAT tokens as queries.
Video tokens as keys and values.

3D token asks:
  - am I moving?
  - how do I move?
  - which video region corresponds to me?
  - what is my relative motion to the static body?
```

### Architecture

```text
OmniPart / TRELLIS SLAT tokens
  → 3D positional encoding
  → part embedding
  → Q

Wan / Video-DiT tokens
  → temporal positional encoding
  → 2D positional encoding
  → optional moving-mask embedding
  → K, V

Cross Attention
  → gated residual update
  → fused SLAT tokens
  → part pooling
  → joint decoder
```

### Pseudocode

```python
class VideoToSLATFusion(nn.Module):
    def __init__(self, c_slat, c_video, d=768, num_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(c_slat, d)
        self.k_proj = nn.Linear(c_video, d)
        self.v_proj = nn.Linear(c_video, d)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=num_heads,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(d, d),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(d, c_slat)

    def forward(self, slat_tokens, video_tokens, slat_pos, video_pos, video_mask=None):
        q = self.q_proj(slat_tokens + slat_pos)
        k = self.k_proj(video_tokens + video_pos)
        v = self.v_proj(video_tokens)

        delta, attn = self.cross_attn(
            q, k, v,
            key_padding_mask=video_mask,
        )

        fused = slat_tokens + self.out_proj(self.gate(delta) * delta)
        return fused, attn
```

### Joint decoder

```python
part_tokens = pool_by_part(fused_slat_tokens, slat_part_ids)

pred = {
    "moving_logits": MLP_part(part_tokens),
    "joint_type_logits": MLP_type(part_tokens),
    "axis": normalize(MLP_axis(part_tokens)),
    "pivot": MLP_pivot(part_tokens),
    "state_seq": MLP_state(part_tokens),
}
```

### Loss

```text
L =
  λ_part  L_moving_part
+ λ_type  L_joint_type
+ λ_axis  L_axis
+ λ_pivot L_pivot
+ λ_state L_state
```

### Axis loss

```text
L_axis = min(
  angle(u_pred, u_gt),
  angle(-u_pred, u_gt)
)
```

### Pivot loss

建议先用 bbox-normalized coordinate：

```text
p_world = bbox_center + p_norm * bbox_size / 2
L_pivot = ||p_pred_norm - p_gt_norm||_1
```

### Why this is main branch

相比 global concat，这条线更合理，因为 articulation 是 part-level 问题：

```text
global video + global 3D → joint
```

太粗。

更好的形式是：

```text
part-level 3D token + video motion evidence → part-level joint
```

---

## 8. E3: Projective Video-to-SLAT Fusion

### Goal

在有 camera pose 的 synthetic 数据上，用 3D-to-2D projection 显式建立 video feature 和 SLAT token 的对应关系。

### Core idea

```text
For each 3D SLAT voxel or sampled part point:
  x_i ∈ R^3
  project x_i to frame t
  sample video feature F_t(u_i^t)
  aggregate over time
  fuse into the 3D token
```

### Formula

```text
u_i^t = Π(K, T_cam^t, x_i)

f_i^video = TemporalPool({
  GridSample(F_t, u_i^t)
}_{t=1}^{T})

f_i^fused = MLP([f_i^slat, f_i^video])
```

### Variants

| ID | Projected feature | Purpose |
|---|---|---|
| E3-A | Wan VAE feature map | low-cost baseline |
| E3-B | Wan DiT middle-layer feature | motion-aware feature |
| E3-C | generator trajectory feature | explicit correspondence |
| E3-D | SegAnyMotion mask-gated feature | upper-bound motion cue |

### Expected advantage

```text
cross-attention learns 2D-3D alignment implicitly.
projective fusion provides explicit 2D-3D alignment.
```

所以 E3 理论上应该比 E2 更擅长：

```text
- pivot estimation
- q_t estimation
- track reprojection consistency
```

---

## 9. E4: Fusion Initialization + Kinematic Refinement

### Goal

把 trainable branch 定位成 fast initializer，然后用 explicit kinematic optimization 修正 axis / pivot / q_t。

### Pipeline

```text
E2 / E3 predicts:
  moving part
  joint type
  axis
  pivot
  state sequence

Then:
  transform moving part mesh by predicted joint
  render articulated sequence
  compare with observed mask / tracks / silhouette
  optimize joint parameters
```

### Optimization variables

```python
raw_axis
pivot
joint_state[t]
optional_part_SE3_correction
```

### Revolute transform

```text
x_t = R(u, q_t)(x - p) + p
```

### Prismatic transform

```text
x_t = x + q_t u
```

### Refinement loss

```text
L_refine =
  λ_mask L_rendered_mask
+ λ_track L_track_reprojection
+ λ_sil L_object_silhouette
+ λ_col L_collision
+ λ_smooth L_state_smooth
+ λ_prior L_joint_prior
```

### Expected improvement

```text
E2 / E3:
  good moving part, joint type, coarse axis

E4:
  better pivot, better q_t, better rendered mask IoU
```

---

# Part III. Training-Free Branch

## 10. TF0: Geometry + Motion Candidate Fitting

### Goal

建立最小 training-free baseline，不训练任何 joint decoder。

### Input

```text
OmniPart part mesh
OmniPart part mask
part bbox
SegAnyMotion mask / tracks or GT motion mask
```

### Steps

```text
1. Determine moving part.
2. Generate revolute / prismatic candidate joints.
3. Render candidate articulated sequence.
4. Score candidate using mask / track / collision.
5. Select top-k.
6. Optionally optimize axis / pivot / q_t.
```

### Moving part proposal

```text
score_move(i) =
  IoU(projected_part_mask_i, moving_mask)
  or
  average track displacement inside part mask
```

### Revolute candidate generation

```text
axis candidates:
  - x / y / z axes of object bbox
  - x / y / z axes of part bbox
  - PCA axes of moving part points
  - contact-boundary tangent directions
  - hinge-like bbox edges

pivot candidates:
  - contact region centroid
  - bbox edge centers
  - closest static-moving boundary points
  - side face centers
```

### Prismatic candidate generation

```text
axis candidates:
  - average track displacement direction
  - PCA direction of 2D / 3D motion
  - bbox principal axes
```

### Candidate score

```text
E_total =
  λ_mask E_mask
+ λ_track E_track
+ λ_collision E_collision
+ λ_range E_range
+ λ_smooth E_state_smooth
```

### Success criterion

TF0 应该至少能在简单类别上给出合理结果：

```text
laptop: revolute hinge
drawer: prismatic axis
microwave: revolute door
refrigerator: revolute door
```

---

## 11. TF1: Wan Feature Candidate Scoring

### Goal

验证 video generator prior 是否可以帮助筛掉不自然的 joint candidate。

### Core idea

```text
candidate joint θ
  → animate OmniPart part mesh
  → render candidate video
  → extract Wan / Video-DiT feature
  → compare with real video feature
  → rank candidates
```

### Rendered candidate representations

不要只用 RGB，因为 RGB 会引入 texture mismatch。优先使用：

```text
1. moving mask video
2. part-color video
3. silhouette video
4. normal video
5. depth-like video
```

### Wan feature score

```text
E_wan =
  || Φ_wan(V_real_representation)
   - Φ_wan(V_candidate_representation) ||
```

### Total candidate score

```text
E_total =
  λ_geom E_geom
+ λ_track E_track
+ λ_wan E_wan
+ λ_collision E_collision
+ λ_smooth E_smooth
```

### Ablation

| Variant | Description |
|---|---|
| TF1-A | no Wan score |
| TF1-B | Wan VAE feature score |
| TF1-C | Wan DiT middle-layer score |
| TF1-D | high-noise feature score |
| TF1-E | high + mid feature score |

### Success criterion

```text
TF1 should improve candidate ranking over TF0.
Especially:
  - joint type accuracy
  - axis angular error
  - q_t error
```

---

## 12. TF2: FreeArt3D-Lite Joint Optimization

### Goal

借鉴 FreeArt3D 的 per-instance optimization 思路，但先固定 OmniPart geometry，只优化 joint。

### Difference from FreeArt3D

```text
FreeArt3D:
  optimize static geometry
  optimize moving geometry
  optimize articulation
  use 3D diffusion prior

Ours-lite:
  fix OmniPart geometry
  optimize joint axis
  optimize joint pivot
  optimize q_t
  optionally optimize small part SE(3)
  use video motion prior and mask / track loss
```

### Initialization

```text
from TF0 / TF1:
  moving part id
  joint type
  axis u_0
  pivot p_0
  q_t^0
```

### Optimization variables

```python
params = {
    "raw_axis": Tensor[3],
    "pivot": Tensor[3],
    "joint_state": Tensor[T],
    "delta_SE3_moving": Optional[Tensor[6]],
}
```

### Loss

```text
L =
  λ_mask L_mask
+ λ_track L_track
+ λ_silhouette L_silhouette
+ λ_collision L_collision
+ λ_state_smooth L_state_smooth
+ λ_wan L_wan_feature
```

### Expected improvement

TF2 应该主要提升：

```text
- pivot error
- q_t error
- rendered mask IoU
- track reprojection error
```

---

## 13. TF3: PAct-Style Denoising Feature Proposal

### Goal

借鉴 PAct 的发现：articulation 信息可能隐藏在 part-centric denoising tokens 中，但我们不训练 PAct 的 MLP head。

### PAct-like idea

```text
part-denoising token features
  → aggregate over denoising steps
  → part descriptor
  → proposal / ranking / candidate prior
```

### Our adaptation

```text
Instead of:
  part token feature → trained articulation MLP → joint

Use:
  part token feature → training-free part descriptor
  part descriptor + geometry + motion score → candidate proposal
```

### Steps

```text
1. Hook OmniPart / TRELLIS denoising process.
2. Cache multi-step part token features.
3. Aggregate selected denoising steps.
4. Mean-pool + max-pool per part.
5. Use descriptor for:
   - movable part proposal
   - root / parent part proposal
   - candidate type prior
   - hard negative filtering
```

### Priority

TF3 工程成本较高，因为要 hook denoising features，所以排在 TF0 / TF1 / TF2 后面。

---

# Part IV. Video-Generator Trajectory Branch

## 14. VGT0: DiffTrack-Style Video-DiT Correspondence

### Goal

从 video generator 内部特征中提取 temporal correspondence，减少对 SegAnyMotion 的依赖。

### Core idea

```text
For a query point p_0 in frame 0:
  find corresponding video token
  compute feature / attention similarity across frames
  select best matching token in each frame
  obtain trajectory p_t
```

### Steps

```text
1. Feed video into Wan / Video-DiT.
2. Hook selected layers and denoising timesteps.
3. For grid query points in frame 0:
   - map pixel to token
   - compute token similarity to later frames
   - use argmax or soft-argmax to get correspondence
4. Filter low-confidence tracks.
5. Save generator_tracks.
```

### Output

```python
generator_tracks = Tensor[N, T, 2]
track_confidence = Tensor[N, T]
```

### Ablation

| Variant | Feature source |
|---|---|
| VGT0-A | shallow DiT layer |
| VGT0-B | middle DiT layer |
| VGT0-C | deep DiT layer |
| VGT0-D | high-noise timestep |
| VGT0-E | mid-noise timestep |
| VGT0-F | high + mid ensemble |

### Use in downstream

```text
generator_tracks can replace SegAnyMotion tracks in:
  - moving part proposal
  - TF0 candidate fitting
  - TF2 refinement
  - E3 projective fusion
```

---

## 15. VGT1: KL-Tracing / Counterfactual Flow

### Goal

用 generative video model 作为 motion probe，提取 optical-flow-like signal。

### Core idea

```text
clean generation
perturbed generation
difference / KL divergence
  → propagated correspondence
  → sparse or dense flow
```

### Steps

```text
1. Select query points or sparse grid patches.
2. Add small tracer perturbation to first frame.
3. Run video generator or predictor twice:
   - clean
   - perturbed
4. Compare predictive distributions.
5. Locate where perturbation propagates.
6. Convert to flow / tracks.
```

### Output

```python
generator_flow = Tensor[T-1, H, W, 2]
generator_tracks = Tensor[N, T, 2]
```

### Risk

```text
- expensive because each perturbation may require extra forward passes
- may depend on model architecture
- sparse version should be implemented first
```

---

## 16. VGT2: Replace SegAnyMotion

### Goal

验证最终系统是否可以摆脱 SegAnyMotion，让 video generator 自己提供 motion cue。

### Comparison

| ID | Motion source | Usage |
|---|---|---|
| VGT2-A | SegAnyMotion | external baseline |
| VGT2-B | Video-DiT correspondence | generator-derived tracks |
| VGT2-C | KL-tracing flow | generator-derived flow |
| VGT2-D | DiT correspondence + KL flow | generator-only ensemble |
| VGT2-E | SegAnyMotion + generator tracks | upper bound |

### Evaluation

```text
Track quality:
  - track EPE
  - visibility accuracy
  - long-range consistency

Motion segmentation:
  - moving mask IoU
  - moving part accuracy

Downstream joint:
  - joint type accuracy
  - axis angular error
  - pivot error
  - q_t error
  - rendered mask IoU
```

### Success criterion

如果 VGT2-B / VGT2-C downstream joint accuracy 接近 SegAnyMotion，就可以在最终系统里减少或替代 SegAnyMotion。

---

# Part V. Model Heads and Losses

## 17. Joint Parameterization

## 17.1 Revolute joint

```text
axis u ∈ R^3, normalized
pivot p ∈ R^3
angle q_t ∈ R
```

### Transform

```text
x_t = R(u, q_t)(x - p) + p
```

### Loss

```text
L_revolute =
  L_axis
+ L_pivot
+ L_angle
+ L_render
+ L_track
```

### Angle representation

建议预测：

```text
sin(q_t), cos(q_t)
```

避免 angle wrap-around。

---

## 17.2 Prismatic joint

```text
axis u ∈ R^3, normalized
origin p optional
displacement d_t ∈ R
```

### Transform

```text
x_t = x + d_t u
```

### Loss

```text
L_prismatic =
  L_axis
+ L_displacement
+ L_render
+ L_track
```

---

## 17.3 Part-level prediction

不要只预测 object-level joint。建议每个 part 都输出：

```text
moving probability
parent probability
joint type
axis
pivot
state sequence
```

### Multi-part case

如果一个物体有多个 movable parts：

```text
Option A:
  supervised part id matching

Option B:
  Hungarian matching over predicted joints and GT joints

Option C:
  first stage only single dominant moving part
```

第一阶段建议先做 single dominant moving part，降低复杂度。

---

## 18. Loss Summary

```text
L_total =
  λ_part L_part
+ λ_type L_type
+ λ_axis L_axis
+ λ_pivot L_pivot
+ λ_state L_state
+ λ_render L_render
+ λ_track L_track
+ λ_collision L_collision
+ λ_smooth L_smooth
```

### Part classification loss

```text
L_part = CE(pred_moving_part, gt_moving_part)
```

### Joint type loss

```text
L_type = CE(pred_joint_type, gt_joint_type)
```

### Axis loss

```text
L_axis = min(
  arccos(|u_pred · u_gt|),
  arccos(|(-u_pred) · u_gt|)
)
```

### Pivot loss

```text
L_pivot = ||p_pred_norm - p_gt_norm||_1
```

### State loss

```text
Revolute:
  L_state = ||sin(q_pred) - sin(q_gt)||_1
          + ||cos(q_pred) - cos(q_gt)||_1

Prismatic:
  L_state = ||d_pred - d_gt||_1
```

### Rendered mask loss

```text
L_render = 1 - IoU(rendered_moving_mask, gt_or_motion_mask)
```

### Track reprojection loss

```text
L_track =
  mean_k,t || project(T(q_t) x_k) - track_k,t ||_1
```

---

# Part VI. Comparison Settings

## 19. 必须做的主对比

| Method | 3D Input | Motion Input | Training | Refinement |
|---|---|---|---|---|
| B0: 3D-only | OmniPart / SLAT | none | yes | no |
| B1: video-only | none | Wan latent | yes | no |
| B2: late fusion | OmniPart + Wan | Wan latent | yes | no |
| E2: cross-attention | SLAT | Wan tokens | yes | no |
| E3: projective fusion | SLAT + camera | projected Wan feature | yes | no |
| TF0 | OmniPart mesh | SegAnyMotion / GT motion | no | candidate search |
| TF1 | OmniPart mesh | SegAnyMotion + Wan score | no | candidate search |
| TF2 | OmniPart mesh | tracks + masks + Wan score | no | yes |
| VGT-TF2 | OmniPart mesh | generator tracks / flow | no | yes |
| E4 | SLAT + mesh | fusion init + motion cues | yes | yes |

---

## 20. Final Results Table Template

| Method | Motion Source | 3D Prior | Training? | Refinement? | Moving Part Acc ↑ | Joint Type Acc ↑ | Axis Err ↓ | Pivot Err ↓ | q_t Err ↓ | Render IoU ↑ |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 3D-only | none | OmniPart / SLAT | yes | no |  |  |  |  |  |  |
| video-only | Wan latent | none | yes | no |  |  |  |  |  |  |
| late fusion | Wan latent | OmniPart / SLAT | yes | no |  |  |  |  |  |  |
| cross-attn fusion | Wan tokens | SLAT | yes | no |  |  |  |  |  |  |
| projective fusion | projected Wan feature | SLAT | yes | no |  |  |  |  |  |  |
| TF0 | SegAnyMotion | OmniPart mesh | no | candidate |  |  |  |  |  |  |
| TF1 | SegAnyMotion + Wan score | OmniPart mesh | no | candidate |  |  |  |  |  |  |
| TF2 | SegAnyMotion + Wan score | OmniPart mesh | no | optimize |  |  |  |  |  |  |
| VGT-TF2 | generator tracks / flow | OmniPart mesh | no | optimize |  |  |  |  |  |  |
| E4 | fusion init + refinement | SLAT + mesh | yes | optimize |  |  |  |  |  |  |

---

# Part VII. Ablation Plan

## 21. Wan Feature Ablation

| Variant | Description | Expected |
|---|---|---|
| VAE latent | compressed low-level video latent | weak but cheap |
| DiT shallow | local appearance / low-level feature | maybe good for mask |
| DiT middle | motion / correspondence candidate | likely best |
| DiT deep | semantic / appearance-heavy | may overfit category |
| high-noise timestep | layout / global motion | good for joint type / axis |
| mid-noise timestep | balance motion and detail | good overall |
| low-noise timestep | detail / texture | may be less useful |

---

## 22. Fusion Ablation

| Variant | Description |
|---|---|
| concat + MLP | lowest baseline |
| gated sum | simple token fusion |
| Perceiver pooling | compress video tokens |
| SLAT-query-video cross-attention | main model |
| projective video-to-SLAT | geometry-aware model |
| trajectory-token fusion | use explicit generator tracks |

Expected order:

```text
concat + MLP
< Perceiver / gated fusion
< cross-attention
< projective fusion
< fusion + refinement
```

---

## 23. Motion Source Ablation

| Motion Source | Role |
|---|---|
| none | shape prior only |
| GT mask / flow | upper bound |
| SegAnyMotion | external motion baseline |
| Video-DiT correspondence | generator-derived tracks |
| KL-tracing flow | generator-derived flow |
| SegAnyMotion + generator tracks | ensemble upper bound |

---

## 24. 3D Prior Ablation

| 3D Prior | Purpose |
|---|---|
| bbox only | lowest geometry baseline |
| part mesh sampled points | geometry-only baseline |
| OmniPart part mesh | part-aware geometry |
| TRELLIS / OmniPart SLAT | structured latent |
| SLAT + bbox | geometry + spatial prior |
| SLAT + mesh points | strongest 3D prior |

---

## 25. Training-Free Ablation

| Variant | Purpose |
|---|---|
| candidate only | test geometric heuristic |
| + mask score | test motion mask |
| + track score | test trajectory constraint |
| + Wan feature score | test video prior |
| + continuous refinement | test optimization |
| + geometry residual | FreeArt3D-like stronger version |
| + PAct-style denoising descriptor | test denoising part feature |

---

# Part VIII. 实验执行顺序

## 26. 最小可行路径

如果时间紧，按这个顺序做：

```text
Step 1:
  build rendered dataset
  cache OmniPart outputs
  cache Wan features

Step 2:
  train 3D-only
  train video-only
  train late fusion

Step 3:
  train video-to-SLAT cross-attention

Step 4:
  implement TF0 candidate fitting

Step 5:
  add Wan feature score → TF1

Step 6:
  add continuous refinement → TF2

Step 7:
  try generator-derived tracks → VGT0

Step 8:
  combine fusion initialization + TF2 refinement → E4
```

---

## 27. 两天冲刺版本

如果只有两天，目标不是完整系统，而是拿到 40% 的关键证据。

### Day 1

```text
1. 选 2-3 类：
   - laptop
   - drawer
   - microwave

2. 每类 10-20 个 objects。

3. 每个 object 渲染 8 frames。

4. 准备 GT:
   - joint type
   - axis
   - pivot
   - q_t
   - moving part id

5. cache:
   - part mesh / bbox
   - simple part point feature
   - Wan VAE latent or existing video feature

6. 训练:
   - 3D-only
   - video-only
   - late fusion
```

### Day 2

```text
1. 实现 simplified cross-attention:
   part tokens as query
   video pooled tokens as key / value

2. 训练 E2-small。

3. 实现 TF0:
   - moving part from GT / mask overlap
   - enumerate axis candidates
   - score by rendered mask or simple track loss

4. 输出第一版结果表:
   - 3D-only
   - video-only
   - late fusion
   - cross-attention
   - TF0
```

### 两天内最重要的结论

```text
Q1: video+3D 是否超过 3D-only？
Q2: cross-attention 是否超过 late fusion？
Q3: training-free candidate fitting 是否能给出合理 axis？
Q4: Wan DiT feature 是否比 VAE latent 更有用？
```

这四个问题只要答出两个，就可以继续推进。

---

# Part IX. 代码结构建议

## 28. Repo Structure

```text
articulation_world_model/
  configs/
    dataset/
      partnet_mobility.yaml
    model/
      baseline_3d.yaml
      baseline_video.yaml
      late_fusion.yaml
      cross_fusion.yaml
      projective_fusion.yaml
      training_free.yaml

  data/
    render_partnet.py
    build_dataset.py
    dataset_articulation.py

  cache/
    cache_omnipart.py
    cache_wan_features.py
    cache_seganymotion.py
    cache_generator_tracks.py

  models/
    wan_feature_extractor.py
    omnipart_feature_extractor.py
    part_slat_pool.py
    video_pooler.py
    fusion_late.py
    fusion_cross_attn.py
    fusion_projective.py
    joint_decoder.py

  training_free/
    moving_part_proposal.py
    joint_candidate_generator.py
    candidate_renderer.py
    candidate_scorer.py
    kinematic_refinement.py

  video_generator_motion/
    dit_correspondence.py
    kl_tracing_flow.py
    track_filtering.py

  losses/
    joint_losses.py
    render_losses.py
    track_losses.py
    collision_losses.py

  train/
    train_3d_only.py
    train_video_only.py
    train_late_fusion.py
    train_cross_fusion.py
    train_projective_fusion.py

  eval/
    eval_joint_metrics.py
    eval_tracks.py
    eval_render_iou.py
    visualize_joint.py
    visualize_tracks.py
```

---

# Part X. 最终论文叙事

## 29. 不建议的表述

不要把工作讲成：

```text
We concatenate video latent and 3D latent for joint prediction.
```

这个听起来太弱，也容易被质疑：

```text
video latent has no depth
3D latent and video latent topology mismatch
joint regression may just learn category bias
```

---

## 30. 推荐表述

更好的表述是：

```text
We investigate whether pretrained video generators can serve as motion-prior engines for articulated joint estimation. Instead of directly regressing joints from RGB videos, we inject motion-aware video features into structured 3D part latents provided by OmniPart / TRELLIS. The 3D branch provides part-level geometry anchors, while the video branch provides temporal motion evidence. A lightweight fusion module initializes explicit kinematic parameters, which are further refined by a training-free motion and geometry consistency optimization.
```

中文版本：

```text
我们不是简单拼接 video latent 和 3D latent，而是把 video generator 中的 motion-aware features 注入到 OmniPart / TRELLIS 的结构化 3D part latent space 中。3D part latent 负责提供几何、部件和深度锚点，video generator 负责提供运动先验和时间一致性证据。模型先通过轻量 cross-modal fusion 预测显式 joint 参数，再通过 training-free kinematic refinement 保证物理一致性。
```

---

## 31. 最终 Contribution Draft

```text
1. We propose a part-centric video-to-3D latent fusion framework for articulated joint estimation, where structured 3D part tokens query motion-aware video generator features.

2. We introduce a training-free video-prior-guided articulation fitting branch that combines OmniPart geometry, motion cues, and Wan / Video-DiT feature scoring to optimize explicit kinematic parameters per instance.

3. We explore video-generator-derived trajectories and flow as an alternative to external motion segmentation / tracking modules, reducing dependency on SegAnyMotion.

4. We show that trainable fusion and training-free fitting are complementary: fusion provides fast initialization, while optimization improves geometric and physical consistency.
```

---

# Part XI. References

```text
OmniPart:
  Part-Aware 3D Generation with Semantic Decoupling and Structural Cohesion
  https://arxiv.org/abs/2507.06165

TRELLIS:
  Structured 3D Latents for Scalable and Versatile 3D Generation
  https://arxiv.org/abs/2412.01506

SegAnyMotion:
  Segment Any Motion in Videos
  https://arxiv.org/abs/2503.22268

PAct:
  Part-Decomposed Single-View Articulated Object Generation
  https://arxiv.org/abs/2602.14965

FreeArt3D:
  Training-Free Articulated Object Generation using 3D Diffusion
  https://arxiv.org/abs/2510.25765

DiffTrack:
  Emergent Temporal Correspondences from Video Diffusion Transformers
  https://arxiv.org/abs/2506.17220

Zero-shot optical flow from generative video models:
  Taming generative video models for zero-shot optical flow extraction
  https://arxiv.org/abs/2507.09082

Wan2.2:
  Wan-Video official repository
  https://github.com/Wan-Video/Wan2.2
```