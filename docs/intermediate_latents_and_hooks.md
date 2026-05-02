# OmniPart 推理中的中间 Latent 与 Hook 提取指南

本文说明在**单个 case** 的完整推理流程中，可通过 **forward hook** 或**少量改代码**取到的主要中间表征（latent / 张量），便于调试与分析。

适用路径示例：`scripts/inference_omnipart.py`、`app_utils.generate_parts`（WebUI 同源逻辑）。

---

## 一、Sparse Structure 阶段（`get_coords` / `sample_sparse_structure`）

| 名称 | 位置 | 典型形状 / 类型 | 获取方式 |
|------|------|-------------------|----------|
| 结构流初始噪声 `noise` | `OmniPartImageTo3DPipeline.sample_sparse_structure` | `[num_samples, C, reso, reso, reso]` dense | Hook `sparse_structure_flow_model` 输入；或在 sampler 内缓存 |
| 结构流采样输出 `z_s` | `sparse_structure_sampler.sample(...).samples` | 与 noise 同阶 dense latent | Hook sampler 的 `_inference_model` 或采样结束处缓存 |
| 占据场 `decoder(z_s)` | `sparse_structure_decoder` 前向 | 体素 logits / 概率场 | Hook `decoder` 的 `forward` 输出 |
| 稀疏坐标 `coords` | `torch.argwhere(decoder(z_s)>0)[:, [0,2,3,4]]` | `[N, 4]` int，`[batch, x, y, z]` | 已有 `voxel_coords.npy`；或在 `get_coords` return 前保存 |

---

## 二、BBox 生成阶段（`BboxGen.generate`）

| 名称 | 位置 | 说明 |
|------|------|------|
| 图像 token `image_latents` | `image_encoder(batch['images'])` 之后 | 图像编码中间表征 |
| 融合序列 `voxel_token` | `cat(image_latents, feat_volume, dim=1)` 之后 | 送入自回归 decoder 的序列 |
| 离散 token `output` | `decoder.generate(...)` 返回值 | `tokenizer.decode` 之前 |
| 连续 bbox `bounds` | `tokenizer.decode(...)` 之后 | 对应落盘 `bboxes.npy` |
| PartField 点特征 | `partfield_encoder.encode(rot_points)` | 如 `[B*M, 448]` 量级 |
| 体素特征体 `feat_volume` | scatter + `partfield_voxel_encoder` 前后 | 如 `[B, C, 64, 64, 64]` 量级 |

**推荐 Hook 子模块**：`image_encoder`、`partfield_encoder`、`partfield_voxel_encoder`、自回归 `decoder`（LM）。

---

## 三、SLAT 流阶段（`sample_slat` → `SLatFlowModel`）

| 名称 | 位置 | 说明 |
|------|------|------|
| SLAT 初始噪声 | `SparseTensor(randn feats, coords=coords)` | 与 `prepare_part_synthesis_input` 给出的 `coords` 点数一致 |
| 采样器各步 `x_t` | `slat_sampler` 内部 Euler/多步循环 | 需 hook sampler 的 `_inference_model` 或逐步缓存 |
| 采样结束 SLAT | `slat_sampler.sample(...).samples` | **归一化之前**的 SLAT（在 `sample_slat` 内 `* std + mean` 之前） |
| 归一化后 SLAT | `slat * std + mean` 之后 | `sample_slat` 返回给 `get_slat` 的值 |
| 切分并滤点后的 SLAT | `divide_slat` → `remove_noise` 之后 | **进入 `slat_decoder_*` 之前的最终 SLAT** |

**推荐 Hook**：`pipeline.models['slat_flow_model']` 整网 forward 的输入/输出；或 `SLatFlowModel` 内具体 block（需改库或子类）。

---

## 四、SLAT Decoder 阶段（`decode_slat`）

| 名称 | 位置 |
|------|------|
| Mesh 解码输出 | `slat_decoder_mesh(slat)` |
| Gaussian 解码输出 | `slat_decoder_gs(slat)` |
| Radiance Field 解码输出 | `slat_decoder_rf(slat)`（若 `formats` 包含） |

一般为**已解码几何/高斯对象列表**；若需「decoder 内部层 latent」，需对 `slat_decoder_*` 的子 module 注册 hook。

---

## 五、图像条件（与 SLAT 共用）

| 名称 | 位置 |
|------|------|
| DINO patch 特征 | `encode_image` / `get_cond` 中的 patch tokens |
| 融合 mask 后的 `cond` | `structured_latent_flow.forward` 内 `cond = cond + group_emb` 之后 |

**Hook**：`image_cond_model`（DINOv2）或 `SLatFlowModel.forward` 前半段。

---

## 六、优先推荐的 5 个检查点（性价比最高）

1. **`voxel_coords` 或 `decoder(z_s)`** — 检查第一阶段稀疏结构是否合理。  
2. **BBox 分支的 `output`（LM token）** — `tokenizer.decode` 之前的离散序列。  
3. **`slat_sampler.sample(...).samples`** — **归一化前**的 SLAT。  
4. **`divide_slat` + `remove_noise` 之后** — **decoder 输入侧**最终 SLAT。  
5. **`slat_flow_model` 单次 forward 输出** — 若只关心「一步去噪后」表征，可在采样循环内 hook。

---

## 七、Hook 注册示例（PyTorch）

```python
handles = []
cache = {}

def make_hook(name):
    def fn(module, inp, out):
        if hasattr(out, "detach"):
            cache[name] = out.detach().cpu()
        else:
            cache[name] = out  # 如 SparseTensor，按需 .feats / .coords 拷贝
    return fn

# 示例：SLAT flow 输出
model = pipeline.models["slat_flow_model"]
h = model.register_forward_hook(make_hook("slat_flow_out"))
handles.append(h)

# 推理结束后
# for h in handles: h.remove()
```

**注意**：

- `SparseTensor` 等自定义类型可能无 `.detach()`，需按实际类型保存 `feats` / `coords`。  
- 多步扩散/流匹配需在 **sampler 循环内**逐 step 缓存，仅 hook 一次 `forward` 只能看到其中一步。  
- 大 tensor 建议 `.cpu()` 再 `torch.save` / `np.save`，避免显存占用。

---

## 八、与「整体 / 部件」相关的解码输出

`get_slat` 末尾：`decode_slat(divide_slat(slat, part_layouts), formats)`。

- `divide_slat` 按 `part_layouts` 的 slice 将长序列 SLAT 切成 **整体 + 各 part**，再经 `remove_noise` 后交给 decoder。  
- `save_parts_outputs` 中列表下标 **`i=0` 通常为整体**，`i>=1` 为各部件（具体以 `process_utils.py` 循环为准）。

---

## 相关源码文件（便于对照）

| 阶段 | 文件 |
|------|------|
| 稀疏结构采样 | `modules/part_synthesis/pipelines/omnipart_image_to_parts.py`（`sample_sparse_structure`, `get_coords`） |
| SLAT 采样与解码 | 同上（`sample_slat`, `divide_slat`, `decode_slat`） |
| SLAT 条件与嵌入 | `modules/part_synthesis/models/structured_latent_flow.py` |
| BBox 生成 | `modules/bbox_gen/models/autogressive_bbox_gen.py` |
| 推理脚本 | `scripts/inference_omnipart.py` |
| 保存部件 | `modules/part_synthesis/process_utils.py`（`save_parts_outputs`） |

---
