"""
OmniPart Standalone Inference Script

Complete pipeline: Image -> Background Removal -> SAM Segmentation -> Mask Merge
                  -> Voxel Coords -> BBox Generation -> Part Synthesis -> 3D Export

No Gradio dependency required.

Two input modes:
  1. --image PATH            : standard path; RMBG removes background, user provides
                               --merge string to group SAM segments manually.
  2. --frame_dir DIR         : directory with frame.npy + mask0.npy + mask1.npy
                               (from a SAM2 pre-segmentation).  Background is derived
                               from the union of provided masks; merge groups are
                               computed automatically by IoU matching.
"""

import os
import sys
import argparse
import glob as glob_module

os.environ['SPCONV_ALGO'] = 'native'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import torch
import trimesh
from PIL import Image
from omegaconf import OmegaConf
from segment_anything import SamAutomaticMaskGenerator, build_sam
from transformers import AutoModelForImageSegmentation
from huggingface_hub import hf_hub_download

from modules.bbox_gen.models.autogressive_bbox_gen import BboxGen
from modules.part_synthesis.pipelines import OmniPartImageTo3DPipeline
from modules.part_synthesis.process_utils import save_parts_outputs
from modules.inference_utils import (
    load_img_mask,
    prepare_bbox_gen_input,
    prepare_part_synthesis_input,
    gen_mesh_from_bounds,
    vis_voxel_coords,
    merge_parts,
)
from modules.label_2d_mask.label_parts import (
    prepare_image,
    get_sam_mask,
    get_mask,
    clean_segment_edges,
    resize_and_pad_to_square,
    size_th as DEFAULT_SIZE_TH,
)
from modules.label_2d_mask.visualizer import Visualizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CANVAS_SIZE = 518  # OmniPart's fixed internal resolution


# ---------------------------------------------------------------------------
# 1. Model Loading
# ---------------------------------------------------------------------------

def load_models(sam_ckpt, partfield_ckpt, bbox_gen_ckpt):
    """Load all required models and return them as a dict."""
    print("[1/4] Loading models ...")

    print("  - SAM ...")
    sam_model = build_sam(checkpoint=sam_ckpt).to(device=DEVICE)
    sam_mask_generator = SamAutomaticMaskGenerator(sam_model)

    print("  - BriaRMBG 2.0 ...")
    rmbg_model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0", trust_remote_code=True
    )
    rmbg_model.to(DEVICE).eval()

    print("  - PartSynthesis pipeline ...")
    part_synthesis_pipeline = OmniPartImageTo3DPipeline.from_pretrained("omnipart/OmniPart")
    part_synthesis_pipeline.to(DEVICE)

    print("  - BboxGen ...")
    bbox_gen_config = OmegaConf.load("configs/bbox_gen.yaml").model.args
    bbox_gen_config.partfield_encoder_path = partfield_ckpt
    bbox_gen_model = BboxGen(bbox_gen_config)
    bbox_gen_model.load_state_dict(torch.load(bbox_gen_ckpt), strict=False)
    bbox_gen_model.to(DEVICE).eval().half()

    print("  All models loaded.\n")
    return {
        "sam": sam_mask_generator,
        "rmbg": rmbg_model,
        "part_synth": part_synthesis_pipeline,
        "bbox_gen": bbox_gen_model,
    }


# ---------------------------------------------------------------------------
# 2. Image Processing & SAM Segmentation  (original path)
# ---------------------------------------------------------------------------

def segment_image(image_path, models, output_dir, size_threshold):
    """
    Background removal -> resize & pad -> SAM segmentation -> split disconnected.

    Returns:
        group_ids  (np.ndarray): per-pixel segment IDs, -1 = background
        image      (np.ndarray): white-bg RGB image (518x518)
        rgba_path  (str): path to saved RGBA processed image
        img_name   (str): base name without extension
    """
    print("[2/4] Segmenting image ...")
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    img = Image.open(image_path).convert("RGB")
    processed_image = prepare_image(img, rmbg_net=models["rmbg"].to(DEVICE))
    processed_image = resize_and_pad_to_square(processed_image)

    white_bg = Image.new("RGBA", processed_image.size, (255, 255, 255, 255))
    white_bg_img = Image.alpha_composite(white_bg, processed_image.convert("RGBA"))
    image = np.array(white_bg_img.convert("RGB"))

    rgba_path = os.path.join(output_dir, f"{img_name}_processed.png")
    processed_image.save(rgba_path)

    visual = Visualizer(image)
    group_ids, pre_merge_im = get_sam_mask(
        image,
        models["sam"],
        visual,
        merge_groups=None,
        rgba_image=processed_image,
        img_name=img_name,
        save_dir=output_dir,
        size_threshold=size_threshold,
    )

    Image.fromarray(pre_merge_im).save(
        os.path.join(output_dir, f"{img_name}_mask_pre_merge.png")
    )
    get_mask(group_ids, image, ids=2, img_name=img_name, save_dir=output_dir)

    unique = np.unique(group_ids)
    unique = unique[unique >= 0]
    print(f"  Initial segments: {len(unique)}  IDs: {sorted(unique.tolist())}\n")

    return group_ids, image, rgba_path, img_name


# ---------------------------------------------------------------------------
# 2-B. Frame-dir path helpers
# ---------------------------------------------------------------------------

def center_pad_to_canvas(arr, bbox, canvas_size=CANVAS_SIZE, bg_value=0):
    """
    Crop the object region defined by *bbox* from *arr* and paste it centred
    onto a blank canvas of size (canvas_size, canvas_size[, C]).

    Args:
        arr        : numpy array with shape (H, W) or (H, W, C)
        bbox       : (row_min, row_max, col_min, col_max)  -- inclusive
        canvas_size: target square side length (default: 518)
        bg_value   : fill value for the canvas background (0 or 255)

    Returns:
        canvas     : ndarray of shape (canvas_size, canvas_size[, C])
        paste_row  : top-left row where the crop was pasted
        paste_col  : top-left col where the crop was pasted
    """
    row_min, row_max, col_min, col_max = bbox
    crop = arr[row_min:row_max + 1, col_min:col_max + 1]
    crop_h, crop_w = crop.shape[:2]

    if crop_h > canvas_size or crop_w > canvas_size:
        scale = canvas_size / max(crop_h, crop_w)
        new_h = max(1, int(round(crop_h * scale)))
        new_w = max(1, int(round(crop_w * scale)))
        print(f"  [scale-down] bbox ({crop_h}x{crop_w}) -> ({new_h}x{new_w}), scale={scale:.4f}")
        if arr.ndim == 2:
            interp = cv2.INTER_NEAREST if arr.dtype == np.uint8 and arr.max() <= 1 else cv2.INTER_AREA
            crop = cv2.resize(crop, (new_w, new_h), interpolation=interp)
        else:
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        crop_h, crop_w = new_h, new_w

    paste_row = (canvas_size - crop_h) // 2
    paste_col = (canvas_size - crop_w) // 2

    if arr.ndim == 2:
        canvas = np.full((canvas_size, canvas_size), bg_value, dtype=arr.dtype)
    else:
        canvas = np.full((canvas_size, canvas_size, arr.shape[2]), bg_value, dtype=arr.dtype)

    canvas[paste_row:paste_row + crop_h, paste_col:paste_col + crop_w] = crop
    return canvas, paste_row, paste_col


def build_canvas_from_frame(frame_dir, output_dir, img_name, canvas_size=CANVAS_SIZE):
    """
    Load frame.npy + mask*.npy from *frame_dir*, centre-pad them to
    *canvas_size* x *canvas_size* without rescaling the object, and produce
    the RGBA file that downstream SAM segmentation expects.

    Returns:
        image_518   : (H518, W518, 3) uint8 white-bg RGB ndarray
        rgba_path   : path to saved RGBA PNG  (alpha = foreground mask)
        mask0_518   : (H518, W518) uint8 binary mask for part 0 (static)
        mask1_518   : (H518, W518) uint8 binary mask for part 1 (dynamic)
        bbox        : (row_min, row_max, col_min, col_max) used for the crop
        paste_offset: (paste_row, paste_col)
    """
    # --- load raw arrays ---
    frame = np.load(os.path.join(frame_dir, "frame.npy"))   # (H, W, 3) uint8
    mask0 = np.load(os.path.join(frame_dir, "mask0.npy"))   # (H, W)    uint8
    mask1 = np.load(os.path.join(frame_dir, "mask1.npy"))   # (H, W)    uint8

    print(f"  Loaded frame {frame.shape}, mask0 {mask0.shape}, mask1 {mask1.shape}")

    # --- compute object bounding box from union of all masks ---
    union = ((mask0 | mask1) > 0)
    rows = np.where(union.any(axis=1))[0]
    cols = np.where(union.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("No foreground pixels found in mask0 or mask1.")

    bbox = (int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1]))
    bbox_h = bbox[1] - bbox[0] + 1
    bbox_w = bbox[3] - bbox[2] + 1
    print(f"  Object bbox: H={bbox_h}, W={bbox_w} -> fits in {canvas_size}x{canvas_size}: "
          f"{bbox_h <= canvas_size and bbox_w <= canvas_size}")

    # --- centre-pad each array with the same offsets ---
    frame_518, paste_row, paste_col = center_pad_to_canvas(
        frame, bbox, canvas_size, bg_value=255
    )
    mask0_518, _, _ = center_pad_to_canvas(mask0, bbox, canvas_size, bg_value=0)
    mask1_518, _, _ = center_pad_to_canvas(mask1, bbox, canvas_size, bg_value=0)
    paste_offset = (paste_row, paste_col)

    print(f"  Pasted at row={paste_row}, col={paste_col}")

    # --- build RGBA: alpha channel = union of masks (foreground = 255) ---
    alpha = ((mask0_518 | mask1_518) > 0).astype(np.uint8) * 255
    rgba_arr = np.dstack([frame_518, alpha])                 # (518, 518, 4)
    rgba_image = Image.fromarray(rgba_arr, mode="RGBA")
    rgba_path = os.path.join(output_dir, f"{img_name}_processed.png")
    rgba_image.save(rgba_path)

    # white-bg RGB for SAM
    white_bg = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
    image_518 = np.array(Image.alpha_composite(white_bg, rgba_image).convert("RGB"))

    return image_518, rgba_path, mask0_518, mask1_518, bbox, paste_offset


# ---------------------------------------------------------------------------
# 3. Mask Merge
# ---------------------------------------------------------------------------

def parse_merge_groups(merge_str, valid_ids):
    """Parse '0,1;3,4' into [[0,1],[3,4]], validating against valid_ids."""
    if not merge_str or not merge_str.strip():
        return None
    groups = []
    for part in merge_str.strip().split(";"):
        ids = [int(x) for x in part.split(",") if x.strip()]
        existing = [i for i in ids if i in valid_ids]
        if existing:
            groups.append(ids)
    return groups if groups else None


def get_image_merge_string(initial_merge, group_ids, output_dir, img_name, interactive_merge):
    """
    Resolve merge string for image mode.

    If interactive_merge is enabled, pause after SAM pre-merge visualization is
    generated and let the user type merge groups in terminal.
    """
    merge_value = initial_merge.strip() if isinstance(initial_merge, str) else ""
    if not interactive_merge:
        return merge_value

    preview_path = os.path.join(output_dir, f"{img_name}_mask_pre_merge.png")
    unique_ids = np.unique(group_ids)
    unique_ids = unique_ids[unique_ids >= 0]
    id_list = sorted(unique_ids.tolist())

    print("[3/4] Interactive merge input ...")
    print(f"  Preview image: {preview_path}")
    print(f"  Available SAM segment IDs: {id_list}")
    print("  Format example: 0,1;3,4")
    if merge_value:
        print(f"  Current --merge value: '{merge_value}' (press Enter to keep it)")
    else:
        print("  Current --merge value: <empty> (press Enter to keep no-merge)")

    try:
        typed = input("  Enter merge groups (or Enter to continue): ").strip()
    except EOFError:
        print("  No interactive stdin detected, keep existing --merge value.")
        return merge_value

    return typed if typed else merge_value


def load_mask_reference(mask_path, target_size=CANVAS_SIZE):
    """
    Load a mask .npy file and apply the same resize-and-pad-to-square
    transform that segment_image uses, so the mask aligns with SAM results.

    Returns:
        mask_518 : (target_size, target_size) uint8 binary mask
    """
    mask = np.load(mask_path)  # (H, W) uint8
    h, w = mask.shape[:2]

    if w > h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))

    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    paste_y = (target_size - new_h) // 2
    paste_x = (target_size - new_w) // 2
    canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized

    return canvas


def auto_merge_groups(sam_group_ids, mask1_518, contain_threshold=0.3):
    """
    Automatically compute merge groups from SAM segments using mask1 (moving
    part reference).  Every segment is assigned to exactly one of two groups:

      - "moving": segment overlaps mask1 above *contain_threshold*
      - "static": everything else

    The output format is identical to what manual merge produces — a list of
    lists of SAM segment IDs, ready to be passed to apply_merge.

    Returns:
        merge_groups : List[List[int]] — always two groups (moving, static).
                       None only if there are no valid segments at all.
        assignment   : dict  {sam_id: 'moving'|'static'}
    """
    unique_ids = np.unique(sam_group_ids)
    unique_ids = unique_ids[unique_ids >= 0]

    moving_group = []
    static_group = []
    assignment = {}

    fg1 = mask1_518 > 0

    for sid in unique_ids:
        seg_mask = sam_group_ids == sid
        seg_area = seg_mask.sum()
        if seg_area == 0:
            continue

        inter1 = (seg_mask & fg1).sum()
        contain1 = inter1 / seg_area

        if contain1 >= contain_threshold:
            moving_group.append(int(sid))
            assignment[int(sid)] = "moving"
            print(f"  SAM seg {sid}: contain_mask1={contain1:.3f} -> moving")
        else:
            static_group.append(int(sid))
            assignment[int(sid)] = "static"
            print(f"  SAM seg {sid}: contain_mask1={contain1:.3f} -> static")

    merge_groups = []
    if len(moving_group) >= 1:
        merge_groups.append(moving_group)
    if len(static_group) >= 1:
        merge_groups.append(static_group)

    print(f"  Moving group  ({len(moving_group)} segs): {moving_group}")
    print(f"  Static group  ({len(static_group)} segs): {static_group}")

    return merge_groups if merge_groups else None, assignment


def apply_merge(group_ids, image, merge_str_or_groups, models, output_dir, img_name,
                rgba_path, size_threshold):
    """
    Merge segments and save .exr mask.

    *merge_str_or_groups* can be:
        - a string like '0,1;3,4'  (original manual mode)
        - a list of lists          (auto mode from auto_merge_groups)
        - None                     (no merging)

    Returns:
        group_ids   (np.ndarray): updated per-pixel IDs
        mask_path   (str): path to saved .exr mask for downstream 3D generation
    """
    print("[3/4] Applying merge ...")
    original_ids = group_ids.copy()
    unique = np.unique(original_ids)
    valid_ids = set(unique[unique >= 0].tolist())

    if isinstance(merge_str_or_groups, str):
        merge_groups = parse_merge_groups(merge_str_or_groups, valid_ids)
    else:
        merge_groups = merge_str_or_groups  # already a list or None

    if merge_groups:
        print(f"  Merge groups: {merge_groups}")
    else:
        print("  No merge groups; keeping original segmentation.")

    processed_image = Image.open(rgba_path)
    visual = Visualizer(image)

    new_group_ids, merged_im = get_sam_mask(
        image,
        models["sam"],
        visual,
        merge_groups=merge_groups,
        existing_group_ids=original_ids,
        skip_split=True,
        rgba_image=processed_image,
        img_name=img_name,
        save_dir=output_dir,
        size_threshold=size_threshold,
    )
    new_group_ids = clean_segment_edges(new_group_ids)
    get_mask(new_group_ids, image, ids=3, img_name=img_name, save_dir=output_dir)

    save_mask = (new_group_ids + 1).reshape(CANVAS_SIZE, CANVAS_SIZE, 1).repeat(3, axis=-1)
    mask_exr_path = os.path.join(output_dir, f"{img_name}_mask.exr")
    cv2.imwrite(mask_exr_path, save_mask.astype(np.float32))

    unique_new = np.unique(new_group_ids)
    unique_new = unique_new[unique_new >= 0]
    print(f"  Segments after merge: {len(unique_new)}  IDs: {sorted(unique_new.tolist())}\n")

    return new_group_ids, mask_exr_path


# ---------------------------------------------------------------------------
# 4. 3D Generation
# ---------------------------------------------------------------------------

def explode_mesh(mesh, explosion_scale=0.4):
    """Push each sub-mesh away from the global center for exploded view."""
    if isinstance(mesh, trimesh.Trimesh):
        return trimesh.Scene(mesh)
    scene = mesh if isinstance(mesh, trimesh.Scene) else mesh
    if len(scene.geometry) <= 1:
        return scene

    exploded = trimesh.Scene()
    centers = []
    for name, geom in scene.geometry.items():
        if hasattr(geom, "vertices"):
            tf = scene.graph[name][0]
            c = np.mean(trimesh.transformations.transform_points(geom.vertices, tf), axis=0)
            centers.append(c)
    centers = np.array(centers)
    global_c = centers.mean(axis=0)

    for i, (name, geom) in enumerate(scene.geometry.items()):
        if not hasattr(geom, "vertices") or i >= len(centers):
            continue
        d = centers[i] - global_c
        n = np.linalg.norm(d)
        d = d / n if n > 1e-6 else np.random.randn(3) / np.linalg.norm(np.random.randn(3))
        tf = scene.graph[name][0].copy()
        tf[:3, 3] += d * explosion_scale
        exploded.add_geometry(geom, transform=tf, geom_name=name)
    return exploded


def filter_degenerate_bboxes(bboxes, min_thickness=0.01):
    """
    Remove degenerate bounding boxes where any axis has near-zero thickness.

    Args:
        bboxes: ndarray of shape (N, 2, 3) — each row is [min_corner, max_corner]
        min_thickness: minimum allowed thickness along each axis

    Returns:
        filtered bboxes with the same shape convention
    """
    keep = []
    for i, bbox in enumerate(bboxes):
        dims = bbox[1] - bbox[0]  # (3,) — thickness along x, y, z
        if np.all(dims >= min_thickness):
            keep.append(i)
        else:
            degenerate_axes = [ax for ax, d in zip("xyz", dims) if d < min_thickness]
            print(f"  Filtering bbox {i}: degenerate on axis {degenerate_axes}, "
                  f"dims={dims.tolist()}")
    filtered = bboxes[keep]
    if len(keep) < len(bboxes):
        print(f"  Kept {len(keep)}/{len(bboxes)} bboxes after filtering")
    return filtered


def generate_3d(rgba_path, mask_exr_path, models, output_dir, seed, cfg_strength):
    """
    Full 3D generation: voxel coords -> bbox prediction -> part synthesis
    -> mesh & gaussian export.
    """
    print("[4/4] Generating 3D parts ...")
    pipeline = models["part_synth"]
    bbox_gen = models["bbox_gen"]

    img_white_bg, img_black_bg, ordered_mask_input, img_mask_vis = load_img_mask(
        rgba_path, mask_exr_path
    )
    img_mask_vis.save(os.path.join(output_dir, "img_mask_vis.png"))

    # --- voxel structure ---
    print("  Sampling voxel structure ...")
    voxel_coords = pipeline.get_coords(
        img_black_bg, num_samples=1, seed=seed,
        sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5},
    )
    voxel_coords = voxel_coords.cpu().numpy()
    np.save(os.path.join(output_dir, "voxel_coords.npy"), voxel_coords)
    vis_voxel_coords(voxel_coords).export(os.path.join(output_dir, "voxel_coords_vis.ply"))

    # --- bbox generation ---
    print("  Generating bounding boxes ...")
    bbox_input = prepare_bbox_gen_input(
        os.path.join(output_dir, "voxel_coords.npy"), img_white_bg, ordered_mask_input
    )
    bbox_output = bbox_gen.generate(bbox_input)
    raw_bboxes = bbox_output["bboxes"][0]
    filtered_bboxes = filter_degenerate_bboxes(raw_bboxes, min_thickness=0.01)
    np.save(os.path.join(output_dir, "bboxes.npy"), filtered_bboxes)
    gen_mesh_from_bounds(filtered_bboxes).export(
        os.path.join(output_dir, "bboxes_vis.glb")
    )

    # --- part synthesis ---
    print("  Running part synthesis ...")
    ps_input = prepare_part_synthesis_input(
        os.path.join(output_dir, "voxel_coords.npy"),
        os.path.join(output_dir, "bboxes.npy"),
        ordered_mask_input,
    )
    torch.cuda.empty_cache()

    ps_output = pipeline.get_slat(
        img_black_bg,
        ps_input["coords"],
        [ps_input["part_layouts"]],
        ps_input["masks"],
        seed=seed,
        slat_sampler_params={"steps": 25, "cfg_strength": cfg_strength},
        formats=["mesh", "gaussian"],
        preprocess_image=False,
    )

    # --- export ---
    print("  Exporting results ...")
    save_parts_outputs(
        ps_output,
        output_dir=output_dir,
        simplify_ratio=0.0,
        save_video=False,
        save_glb=True,
        textured=False,
    )
    merge_parts(output_dir)

    combined_mesh = trimesh.load(os.path.join(output_dir, "mesh_segment.glb"))
    explode_mesh(combined_mesh, explosion_scale=0.3).export(
        os.path.join(output_dir, "exploded_parts.glb")
    )

    outputs = {
        "bboxes_vis":    os.path.join(output_dir, "bboxes_vis.glb"),
        "mesh_segment":  os.path.join(output_dir, "mesh_segment.glb"),
        "exploded_mesh": os.path.join(output_dir, "exploded_parts.glb"),
        "merged_gs":     os.path.join(output_dir, "merged_gs.ply"),
        "exploded_gs":   os.path.join(output_dir, "exploded_gs.ply"),
    }
    print("  3D generation done.\n")
    return outputs


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _download_ckpts():
    os.makedirs("ckpt", exist_ok=True)
    sam_ckpt = hf_hub_download(
        repo_id="omnipart/OmniPart_modules",
        filename="sam_vit_h_4b8939.pth", local_dir="ckpt",
    )
    partfield_ckpt = hf_hub_download(
        repo_id="omnipart/OmniPart_modules",
        filename="partfield_encoder.ckpt", local_dir="ckpt",
    )
    bbox_gen_ckpt = hf_hub_download(
        repo_id="omnipart/OmniPart_modules",
        filename="bbox_gen.ckpt", local_dir="ckpt",
    )
    return sam_ckpt, partfield_ckpt, bbox_gen_ckpt


def main():
    parser = argparse.ArgumentParser(
        description="OmniPart: single-image part-aware 3D generation (standalone)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- input: mutually exclusive modes ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", type=str,
        help="Input image path (standard mode: RMBG + manual --merge string)",
    )
    input_group.add_argument(
        "--frame_dir", type=str,
        help=(
            "Directory containing frame.npy + mask0.npy + mask1.npy "
            "(SAM2 pre-segmentation mode: no RMBG, auto merge groups). "
            "mask0 = static part, mask1 = dynamic part."
        ),
    )

    # --- shared options ---
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: output/<name>)")
    parser.add_argument("--threshold", type=int, default=DEFAULT_SIZE_TH,
                        help="Min SAM segment size in pixels")
    parser.add_argument("--contain_threshold", type=float, default=0.3,
                        help="[frame_dir mode] Min containment ratio (intersection/seg_area) "
                             "to assign a SAM seg to a reference mask group")
    parser.add_argument("--merge", type=str, default="",
                        help="[image mode] Merge groups string, e.g. '0,1;3,4'")
    parser.add_argument(
        "--interactive_merge",
        action="store_true",
        help="[image mode] Pause after SAM preview and type merge groups in terminal",
    )
    parser.add_argument("--seed", type=int, default=42, help="Generation seed")
    parser.add_argument("--cfg", type=float, default=7.5, help="CFG strength")
    args = parser.parse_args()

    # --- resolve output dir ---
    if args.frame_dir:
        name = os.path.basename(os.path.normpath(args.frame_dir))
    else:
        name = os.path.splitext(os.path.basename(args.image))[0]
    output_dir = args.output_dir or os.path.join("output", name)
    os.makedirs(output_dir, exist_ok=True)

    # --- download checkpoints ---
    sam_ckpt, partfield_ckpt, bbox_gen_ckpt = _download_ckpts()

    # --- load models ---
    models = load_models(sam_ckpt, partfield_ckpt, bbox_gen_ckpt)

    # =========================================================================
    # PATH A: standard image + manual merge string
    # =========================================================================
    if args.image:
        if not os.path.isfile(args.image):
            print(f"Error: image not found: {args.image}")
            sys.exit(1)

        img_name = os.path.splitext(os.path.basename(args.image))[0]

        group_ids, image, rgba_path, img_name = segment_image(
            args.image, models, output_dir, args.threshold,
        )
        merge_string = get_image_merge_string(
            args.merge, group_ids, output_dir, img_name, args.interactive_merge
        )
        group_ids, mask_exr_path = apply_merge(
            group_ids, image, merge_string, models, output_dir, img_name,
            rgba_path, args.threshold,
        )

    # =========================================================================
    # PATH B: frame_dir with mask1 reference -> same SAM pipeline + auto merge
    # =========================================================================
    else:
        if not os.path.isdir(args.frame_dir):
            print(f"Error: frame_dir not found: {args.frame_dir}")
            sys.exit(1)

        # Use frame.png as input — same pipeline as PATH A (RMBG + SAM)
        frame_png = os.path.join(args.frame_dir, "frame.png")
        if not os.path.isfile(frame_png):
            print(f"Error: frame.png not found in {args.frame_dir}")
            sys.exit(1)

        group_ids, image, rgba_path, img_name = segment_image(
            frame_png, models, output_dir, args.threshold,
        )

        # Load mask1 reference and map to 518x518 (same transform as resize_and_pad_to_square)
        mask1_path = os.path.join(args.frame_dir, "mask1.npy")
        if not os.path.isfile(mask1_path):
            print(f"Error: mask1.npy not found in {args.frame_dir}")
            sys.exit(1)
        mask1_518 = load_mask_reference(mask1_path, target_size=CANVAS_SIZE)

        print("[3/4] Auto-computing merge groups from mask1 reference ...")
        merge_groups, assignment = auto_merge_groups(
            group_ids, mask1_518, contain_threshold=args.contain_threshold,
        )

        group_ids, mask_exr_path = apply_merge(
            group_ids, image, merge_groups, models, output_dir, img_name,
            rgba_path, args.threshold,
        )

    # --- 3D generation (same for both paths) ---
    outputs = generate_3d(
        rgba_path, mask_exr_path, models, output_dir, args.seed, args.cfg,
    )

    print("=" * 60)
    print("All done!  Output files:")
    for tag, path in outputs.items():
        status = "OK" if os.path.isfile(path) else "MISSING"
        print(f"  [{status}] {tag:20s} -> {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
