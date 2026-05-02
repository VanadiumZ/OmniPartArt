"""
Evaluation metrics for articulated joint estimation.

Metrics:
  - Moving Part Accuracy (top-1)
  - Joint Type Accuracy
  - Axis Angular Error (degrees)
  - Pivot Error (L2 distance)
  - State Error (angle or displacement)
  - Rendered Mask IoU (if available)
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class JointMetrics:
    """Accumulate and compute joint estimation metrics over a dataset."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._moving_correct = 0
        self._moving_total = 0
        self._type_correct = 0
        self._type_total = 0
        self._axis_errors = []
        self._pivot_errors = []
        self._state_errors = []
        self._render_ious = []

    @torch.no_grad()
    def update(self, pred: Dict[str, torch.Tensor], batch: Dict):
        """Update metrics with a batch of predictions."""
        # Moving part accuracy
        if "moving_logits" in pred and "moving_part_id" in batch:
            logits = pred["moving_logits"]
            for i in range(logits.shape[0]):
                pred_id = logits[i].argmax().item()
                gt_id = batch["moving_part_id"][i]
                if isinstance(gt_id, torch.Tensor):
                    gt_id = gt_id.item()
                self._moving_correct += int(pred_id == gt_id)
                self._moving_total += 1

        # Joint type accuracy
        if "joint_type_logits" in pred and "joint_type" in batch:
            type_logits = pred["joint_type_logits"]
            gt_types = batch["joint_type"]
            if type_logits.dim() == 3:
                type_logits = type_logits[:, 0, :]
            pred_types = type_logits.argmax(dim=-1)
            if gt_types.dim() == 0:
                gt_types = gt_types.unsqueeze(0)
            correct = (pred_types == gt_types).sum().item()
            self._type_correct += correct
            self._type_total += gt_types.shape[0]

        # Axis angular error
        if "axis" in pred and "joint_axis" in batch:
            pred_axis = pred["axis"]
            gt_axis = batch["joint_axis"]
            if pred_axis.dim() == 3:
                pred_axis = pred_axis[:, 0, :]
            pred_norm = F.normalize(pred_axis, dim=-1)
            gt_norm = F.normalize(gt_axis, dim=-1)
            cos_sim = (pred_norm * gt_norm).sum(dim=-1).clamp(-1, 1).abs()
            angles = torch.acos(cos_sim) * 180.0 / np.pi
            self._axis_errors.extend(angles.cpu().numpy().tolist())

        # Pivot error
        if "pivot" in pred and "joint_pivot" in batch:
            pred_pivot = pred["pivot"]
            gt_pivot = batch["joint_pivot"]
            if pred_pivot.dim() == 3:
                pred_pivot = pred_pivot[:, 0, :]
            errors = torch.norm(pred_pivot - gt_pivot, dim=-1)
            self._pivot_errors.extend(errors.cpu().numpy().tolist())

        # State error
        if "state_sincos" in pred and "joint_state" in batch:
            pred_state = pred["state_sincos"]
            gt_state = batch["joint_state"]
            if pred_state.dim() == 4:
                pred_state = pred_state[:, 0, :, :]
            gt_sin = torch.sin(gt_state)
            gt_cos = torch.cos(gt_state)
            err = (
                (pred_state[:, :, 0] - gt_sin).abs()
                + (pred_state[:, :, 1] - gt_cos).abs()
            ).mean(dim=-1)
            self._state_errors.extend(err.cpu().numpy().tolist())

    def compute(self) -> Dict[str, float]:
        results = {}

        if self._moving_total > 0:
            results["moving_part_acc"] = self._moving_correct / self._moving_total

        if self._type_total > 0:
            results["joint_type_acc"] = self._type_correct / self._type_total

        if self._axis_errors:
            results["axis_err_deg"] = float(np.mean(self._axis_errors))
            results["axis_err_deg_median"] = float(np.median(self._axis_errors))

        if self._pivot_errors:
            results["pivot_err"] = float(np.mean(self._pivot_errors))
            results["pivot_err_median"] = float(np.median(self._pivot_errors))

        if self._state_errors:
            results["state_err"] = float(np.mean(self._state_errors))

        if self._render_ious:
            results["render_iou"] = float(np.mean(self._render_ious))

        return results

    def summary_table(self) -> str:
        """Return a formatted table string of all metrics."""
        results = self.compute()
        lines = ["=" * 50, "Joint Estimation Metrics", "=" * 50]
        for key, val in results.items():
            if "acc" in key:
                lines.append(f"  {key:30s}: {val:.2%}")
            elif "err" in key:
                lines.append(f"  {key:30s}: {val:.4f}")
            elif "iou" in key:
                lines.append(f"  {key:30s}: {val:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)


def evaluate_training_free_results(
    results: List[Dict], gt_data: List[Dict]
) -> Dict[str, float]:
    """
    Evaluate training-free pipeline results against ground truth.
    """
    metrics = {
        "moving_part_acc": 0,
        "joint_type_acc": 0,
        "axis_err_deg": [],
        "pivot_err": [],
    }
    total = 0

    for pred, gt in zip(results, gt_data):
        total += 1

        if pred.get("moving_part_id") == gt.get("moving_part_id"):
            metrics["moving_part_acc"] += 1

        cand = pred.get("best_candidate") or pred.get("refined", {})
        gt_type = gt.get("joint_type", "fixed")
        pred_type = cand.get("joint_type", cand.get("type", "fixed")) if isinstance(cand, dict) else getattr(cand, "joint_type", "fixed")
        if pred_type == gt_type:
            metrics["joint_type_acc"] += 1

        pred_axis = np.array(cand.get("axis", cand.axis if hasattr(cand, "axis") else [0, 0, 1]))
        gt_axis = np.array(gt.get("joint_axis", [0, 0, 1]))
        pred_axis = pred_axis / (np.linalg.norm(pred_axis) + 1e-8)
        gt_axis = gt_axis / (np.linalg.norm(gt_axis) + 1e-8)
        cos_sim = np.abs(np.dot(pred_axis, gt_axis)).clip(-1, 1)
        angle_deg = np.degrees(np.arccos(cos_sim))
        metrics["axis_err_deg"].append(angle_deg)

        if "pivot" in (cand if isinstance(cand, dict) else {}):
            pred_pivot = np.array(cand["pivot"])
            gt_pivot = np.array(gt.get("joint_pivot", [0, 0, 0]))
            metrics["pivot_err"].append(np.linalg.norm(pred_pivot - gt_pivot))

    if total > 0:
        metrics["moving_part_acc"] /= total
        metrics["joint_type_acc"] /= total
        metrics["axis_err_deg"] = float(np.mean(metrics["axis_err_deg"])) if metrics["axis_err_deg"] else 0
        metrics["pivot_err"] = float(np.mean(metrics["pivot_err"])) if metrics["pivot_err"] else 0

    return metrics
