"""
Debug script: 只跑 SAM 分割 + Mask Merge，跳过 bbox 生成和 3D 合成。
流程与 run_inference.py 手动 merge 完全一致：
  1. RMBG 去背景 → resize_and_pad_to_square → SAM 分割得到 n 个小分块
  2. 加载 mask1.npy (moving) 参考，映射到 518×518
  3. 计算每个 SAM 分块与 mask1 的重叠度，自动生成 merge 分组
  4. 用与手动 merge 一致的 apply_merge 执行合并
  所有输出 mask 来自 SAM 分割结果，不来自 mask reference。

用法:
    python debug_mask_merge.py \
        --frame_dir result/sam2_frames/final_res/frames/extracted_frames/00024 \
        --output_dir output/debug_merge/00024 \
        [--threshold 300] [--contain_threshold 0.3]
"""

import os
import sys
import argparse

os.environ['SPCONV_ALGO'] = 'native'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, build_sam
from transformers import AutoModelForImageSegmentation
from huggingface_hub import hf_hub_download

from modules.label_2d_mask.label_parts import (
    prepare_image,
    resize_and_pad_to_square,
    get_sam_mask,
    get_mask,
    clean_segment_edges,
    size_th as DEFAULT_SIZE_TH,
)
from modules.label_2d_mask.visualizer import Visualizer

from run_inference import (
    load_mask_reference,
    auto_merge_groups,
    apply_merge,
    CANVAS_SIZE,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 颜色配置 (RGB)
# ---------------------------------------------------------------------------
MASK1_COLOR   = ( 60, 160, 255)   # 蓝 -> mask1 (moving reference)
CONTOUR_ALPHA = 0.55
CONTOUR_THICK = 2

_SEG_PALETTE = [
    (255, 200,  60), ( 60, 220, 120), (200,  60, 255), ( 60, 230, 230),
    (255, 130,  30), (100, 180, 255), (220, 220,  60), (255,  80, 160),
    ( 80, 255, 180), (180, 100, 255), (255, 210, 130), (130, 255, 100),
    (255,  60, 100), ( 60, 100, 255), (200, 255,  60), (255, 160,  80),
    ( 60, 255, 220), (160,  60, 255), (255, 255, 130), (130, 200, 200),
]


def _colored_segments(group_ids: np.ndarray) -> np.ndarray:
    h, w = group_ids.shape
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    unique_ids = sorted(np.unique(group_ids[group_ids >= 0]).tolist())
    for i, sid in enumerate(unique_ids):
        out[group_ids == sid] = _SEG_PALETTE[i % len(_SEG_PALETTE)]
    return out


def _overlay_mask_region(canvas, mask, color, alpha):
    out = canvas.copy().astype(np.float32)
    fg = mask > 0
    for c in range(3):
        out[fg, c] = out[fg, c] * (1 - alpha) + color[c] * alpha
    return out.astype(np.uint8)


def _draw_contours(canvas, mask, color, thickness=2):
    out = canvas.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, color[::-1], thickness)
    return out


def _add_legend(canvas, labels):
    out = canvas.copy()
    x0, y0 = 10, canvas.shape[0] - 10 - len(labels) * 22
    for i, (text, color) in enumerate(labels):
        y = y0 + i * 22
        cv2.rectangle(out, (x0, y - 14), (x0 + 16, y + 2), color[::-1], -1)
        cv2.putText(out, text, (x0 + 22, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _add_title(canvas, text):
    out = canvas.copy()
    cv2.putText(out, text, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
    return out


def make_comparison(frame_rgb, pre_group_ids, post_group_ids, mask1, assignment):
    """
    4-panel comparison:
      [原图+mask1] | [pre-merge SAM segs] | [post-merge segs] | [post-merge+mask1 overlay]
    """
    # --- Panel 0: 原图 + mask1 (moving) ---
    p0 = frame_rgb.copy()
    p0 = _overlay_mask_region(p0, mask1, MASK1_COLOR, 0.40)
    p0 = _draw_contours(p0, mask1, MASK1_COLOR)
    p0 = _add_title(p0, "Reference mask1 (moving)")
    p0 = _add_legend(p0, [("mask1 (moving)", MASK1_COLOR)])

    # --- Panel 1: pre-merge SAM segments with assignment labels ---
    p1 = _colored_segments(pre_group_ids)
    for sid in sorted(np.unique(pre_group_ids[pre_group_ids >= 0]).tolist()):
        ys, xs = np.where(pre_group_ids == sid)
        cy, cx = int(ys.mean()), int(xs.mean())
        label = assignment.get(int(sid), "?")
        short = {"moving": "M", "static": "S"}.get(label, "?")
        cv2.putText(p1, f"{sid}({short})", (cx - 14, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(p1, f"{sid}({short})", (cx - 14, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    p1 = _add_title(p1, "Pre-merge SAM segs")
    p1_legend = [(f"seg{sid}({assignment.get(int(sid),'?')[0].upper()})",
                   _SEG_PALETTE[i % len(_SEG_PALETTE)])
                 for i, sid in enumerate(sorted(np.unique(pre_group_ids[pre_group_ids >= 0]).tolist()))]
    p1 = _add_legend(p1, p1_legend[:10])

    # --- Panel 2: post-merge segments ---
    p2 = _colored_segments(post_group_ids)
    for sid in sorted(np.unique(post_group_ids[post_group_ids >= 0]).tolist()):
        ys, xs = np.where(post_group_ids == sid)
        cy, cx = int(ys.mean()), int(xs.mean())
        cv2.putText(p2, str(sid), (cx - 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(p2, str(sid), (cx - 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    p2 = _add_title(p2, "Post-merge SAM segs")

    # --- Panel 3: post-merge + mask1 overlay ---
    p3 = _colored_segments(post_group_ids)
    p3 = _overlay_mask_region(p3, mask1, MASK1_COLOR, CONTOUR_ALPHA * 0.4)
    p3 = _draw_contours(p3, mask1, MASK1_COLOR, CONTOUR_THICK)
    p3 = _add_title(p3, "Post-merge + mask1 overlay")
    p3 = _add_legend(p3, [("mask1 (moving)", MASK1_COLOR)])

    return np.hstack([p0, p1, p2, p3])


def main():
    parser = argparse.ArgumentParser(
        description="Debug mask merge (no bbox/3D) — same SAM pipeline as run_inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frame_dir", required=True,
                        help="Dir with frame.png + mask1.npy")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--threshold", type=int, default=DEFAULT_SIZE_TH,
                        help="Min SAM segment size (pixels)")
    parser.add_argument("--contain_threshold", type=float, default=0.3,
                        help="Min containment ratio to assign a SAM seg to moving group")
    args = parser.parse_args()

    name = os.path.basename(os.path.normpath(args.frame_dir))
    output_dir = args.output_dir or os.path.join("output", "debug_merge", name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output -> {output_dir}")

    # ------------------------------------------------------------------
    # [1/4] Load SAM + RMBG
    # ------------------------------------------------------------------
    sam_ckpt_local = os.path.join("ckpt", "sam_vit_h_4b8939.pth")
    if not os.path.isfile(sam_ckpt_local):
        print("Downloading SAM checkpoint ...")
        sam_ckpt_local = hf_hub_download(
            repo_id="omnipart/OmniPart_modules",
            filename="sam_vit_h_4b8939.pth", local_dir="ckpt",
        )
    print("  Loading SAM ...")
    sam_model = build_sam(checkpoint=sam_ckpt_local).to(DEVICE)
    sam_gen = SamAutomaticMaskGenerator(sam_model)

    print("  Loading BriaRMBG 2.0 ...")
    rmbg_model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0", trust_remote_code=True
    )
    rmbg_model.to(DEVICE).eval()

    models = {"sam": sam_gen, "rmbg": rmbg_model}

    # ------------------------------------------------------------------
    # [2/4] RMBG + resize_and_pad_to_square + SAM (identical to --image path)
    # ------------------------------------------------------------------
    frame_png = os.path.join(args.frame_dir, "frame.png")
    if not os.path.isfile(frame_png):
        print(f"Error: frame.png not found in {args.frame_dir}")
        sys.exit(1)

    print(f"\n[2/4] Segmenting {frame_png} (RMBG + SAM) ...")
    img = Image.open(frame_png).convert("RGB")
    processed_image = prepare_image(img, rmbg_net=rmbg_model.to(DEVICE))
    processed_image = resize_and_pad_to_square(processed_image)

    white_bg = Image.new("RGBA", processed_image.size, (255, 255, 255, 255))
    image = np.array(Image.alpha_composite(white_bg, processed_image.convert("RGBA")).convert("RGB"))

    rgba_path = os.path.join(output_dir, f"{name}_processed.png")
    processed_image.save(rgba_path)

    visual = Visualizer(image)
    pre_group_ids, pre_merge_im = get_sam_mask(
        image,
        sam_gen,
        visual,
        merge_groups=None,
        rgba_image=processed_image,
        img_name=name,
        save_dir=output_dir,
        size_threshold=args.threshold,
    )
    Image.fromarray(pre_merge_im).save(
        os.path.join(output_dir, f"{name}_mask_pre_merge.png")
    )
    get_mask(pre_group_ids, image, ids=2, img_name=name, save_dir=output_dir)

    unique_init = sorted(np.unique(pre_group_ids[pre_group_ids >= 0]).tolist())
    print(f"  Initial SAM segments: {len(unique_init)}  IDs: {unique_init}")

    # ------------------------------------------------------------------
    # [3/4] Load mask1 reference + auto merge groups
    # ------------------------------------------------------------------
    mask1_path = os.path.join(args.frame_dir, "mask1.npy")
    if not os.path.isfile(mask1_path):
        print(f"Error: mask1.npy not found in {args.frame_dir}")
        sys.exit(1)

    mask1_518 = load_mask_reference(mask1_path, target_size=CANVAS_SIZE)

    print(f"\n[3/4] Auto-computing merge groups from mask1 ...")
    merge_groups, assignment = auto_merge_groups(
        pre_group_ids, mask1_518,
        contain_threshold=args.contain_threshold,
    )

    post_group_ids, mask_exr_path = apply_merge(
        pre_group_ids, image, merge_groups, models,
        output_dir, name, rgba_path, args.threshold,
    )

    # ------------------------------------------------------------------
    # [4/4] Visualization
    # ------------------------------------------------------------------
    print("\n[4/4] Generating comparison image ...")
    comp = make_comparison(image, pre_group_ids, post_group_ids,
                           mask1_518, assignment)

    vis_path = os.path.join(output_dir, f"{name}_merge_comparison.png")
    cv2.imwrite(vis_path, cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print(f"  Saved -> {vis_path}")

    p3 = _colored_segments(post_group_ids)
    p3 = _overlay_mask_region(p3, mask1_518, MASK1_COLOR, 0.3)
    p3 = _draw_contours(p3, mask1_518, MASK1_COLOR, 3)
    big_path = os.path.join(output_dir, f"{name}_merge_overlay.png")
    cv2.imwrite(big_path, cv2.cvtColor(p3, cv2.COLOR_RGB2BGR))
    print(f"  Saved -> {big_path}")

    print("\nDone.")
    print(f"  Segments: {len(np.unique(post_group_ids[post_group_ids >= 0]))}")
    print(f"  Assignment: {assignment}")


if __name__ == "__main__":
    main()
