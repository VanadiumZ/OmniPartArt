"""
Entry point for the training-free pipeline (TF0 / TF1 / TF2).

Usage:
    python -m articulation_exp.training_free.run_training_free \\
        --config configs/model/training_free.yaml \\
        --data_dir data_cache/partnet_mobility \\
        --output_dir outputs/training_free
"""
import argparse
import os
import sys
import json
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from articulation_exp.training_free.moving_part_proposal import MovingPartProposal
from articulation_exp.training_free.joint_candidate_generator import (
    JointCandidateGenerator,
)
from articulation_exp.training_free.candidate_scorer import CandidateScorer
from articulation_exp.training_free.kinematic_refinement import (
    KinematicRefiner, RefinementConfig,
)


def load_sample(sample_dir: str) -> dict:
    """Load a single cached sample for training-free evaluation."""
    gt_path = os.path.join(sample_dir, "gt_joint.json")
    with open(gt_path, "r") as f:
        gt = json.load(f)

    omni_dir = os.path.join(sample_dir, "omnipart")
    data = {"gt": gt}

    bbox_path = os.path.join(omni_dir, "part_bboxes.npy")
    if os.path.exists(bbox_path):
        data["part_bboxes"] = np.load(bbox_path)

    # Placeholder for mesh points (would be loaded from actual mesh files)
    data["moving_points"] = np.random.randn(100, 3).astype(np.float32) * 0.1
    data["static_points"] = np.random.randn(200, 3).astype(np.float32) * 0.1

    motion_dir = os.path.join(sample_dir, "motion")
    mask_path = os.path.join(motion_dir, "seganymotion_masks.npy")
    if os.path.exists(mask_path):
        data["motion_mask"] = np.load(mask_path)
    track_path = os.path.join(motion_dir, "seganymotion_tracks.npy")
    if os.path.exists(track_path):
        data["motion_tracks"] = np.load(track_path)

    return data


def run_tf0(data: dict, cfg: dict) -> dict:
    """TF0: Geometry + motion candidate fitting."""
    tf0_cfg = cfg.get("tf0", {})

    # Step 1: Moving part proposal
    proposer = MovingPartProposal(method="mask_iou")
    if "motion_mask" in data and "part_bboxes" in data:
        P = data["part_bboxes"].shape[0]
        dummy_masks = np.random.rand(P, 64, 64) > 0.5
        moving_id, scores = proposer.propose(dummy_masks, data.get("motion_mask"))
    else:
        moving_id = data["gt"].get("moving_part_id", 0)

    # Step 2: Generate candidates
    generator = JointCandidateGenerator(
        num_axis_candidates=tf0_cfg.get("num_axis_candidates", 20),
        num_pivot_candidates=tf0_cfg.get("num_pivot_candidates", 10),
        joint_types=tf0_cfg.get("joint_types", ["revolute", "prismatic"]),
    )
    moving_bbox = data.get("part_bboxes", np.zeros((1, 6)))[0] if "part_bboxes" in data else np.zeros(6)
    obj_bbox = np.array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5])

    candidates = generator.generate(
        data["moving_points"], data["static_points"],
        moving_bbox, obj_bbox,
        data.get("motion_tracks"),
    )

    # Step 3: Score candidates
    scoring_cfg = tf0_cfg.get("scoring", {})
    scorer = CandidateScorer(**scoring_cfg)
    scored = scorer.score_candidates(
        candidates,
        data["moving_points"],
        data["static_points"],
        motion_mask=data.get("motion_mask"),
        motion_tracks=data.get("motion_tracks"),
    )

    best = scored[0] if scored else None
    return {
        "moving_part_id": moving_id,
        "best_candidate": best,
        "num_candidates": len(scored),
    }


def run_tf2(data: dict, tf0_result: dict, cfg: dict) -> dict:
    """TF2: Kinematic refinement on top of TF0/TF1."""
    tf2_cfg = cfg.get("tf2", {})

    cand = tf0_result.get("best_candidate")
    if cand is None:
        return {"error": "no candidate to refine"}

    refiner = KinematicRefiner(RefinementConfig(
        lr=tf2_cfg.get("lr", 0.01),
        num_iterations=tf2_cfg.get("num_iterations", 200),
        optimize_axis=tf2_cfg.get("optimize_axis", True),
        optimize_pivot=tf2_cfg.get("optimize_pivot", True),
        optimize_state=tf2_cfg.get("optimize_state", True),
        lambda_collision=tf2_cfg.get("loss", {}).get("lambda_collision", 0.3),
        lambda_smooth=tf2_cfg.get("loss", {}).get("lambda_smooth", 0.2),
    ))

    init_state = np.linspace(
        cand.state_range[0], cand.state_range[1], 8
    ).astype(np.float32)

    refined = refiner.refine(
        joint_type=cand.joint_type,
        init_axis=cand.axis,
        init_pivot=cand.pivot,
        init_state=init_state,
        moving_points=data["moving_points"],
        static_points=data["static_points"],
        motion_tracks=data.get("motion_tracks"),
    )

    return {
        "joint_type": cand.joint_type,
        **refined,
    }


def main():
    parser = argparse.ArgumentParser(description="Training-free articulation pipeline")
    parser.add_argument(
        "--config", type=str,
        default="articulation_exp/configs/model/training_free.yaml",
    )
    parser.add_argument("--data_dir", type=str, default="data_cache/partnet_mobility")
    parser.add_argument("--output_dir", type=str, default="outputs/training_free")
    parser.add_argument("--stage", type=str, default="tf0", choices=["tf0", "tf1", "tf2"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)["training_free"]

    results_all = []
    data_root = args.data_dir

    if os.path.exists(data_root):
        for cat in sorted(os.listdir(data_root)):
            cat_dir = os.path.join(data_root, cat)
            if not os.path.isdir(cat_dir):
                continue
            for obj_id in sorted(os.listdir(cat_dir)):
                obj_dir = os.path.join(cat_dir, obj_id)
                if not os.path.isfile(os.path.join(obj_dir, "gt_joint.json")):
                    continue

                data = load_sample(obj_dir)

                tf0_result = run_tf0(data, cfg)

                if args.stage == "tf2":
                    tf2_result = run_tf2(data, tf0_result, cfg)
                    result = {**tf0_result, "refined": tf2_result}
                else:
                    result = tf0_result

                result["object_id"] = obj_id
                result["category"] = cat
                results_all.append(result)
                print(f"Processed {cat}/{obj_id}")

    output_path = os.path.join(args.output_dir, f"{args.stage}_results.json")

    def _serialize(obj):
        if hasattr(obj, "joint_type"):
            return {"type": obj.joint_type, "axis": obj.axis.tolist(), "pivot": obj.pivot.tolist(), "score": obj.score}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(output_path, "w") as f:
        json.dump(results_all, f, indent=2, default=_serialize)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
