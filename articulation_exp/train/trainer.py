"""
Unified trainer for all trainable experiment branches.
Supports: baseline_3d, baseline_video, late_fusion, cross_fusion, projective_fusion.
"""
import os
import time
import json
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Optional
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from articulation_exp.models import build_model
from articulation_exp.losses.joint_losses import ArticulationLoss
from articulation_exp.data.build_dataset import build_dataloaders
from articulation_exp.eval.eval_joint_metrics import JointMetrics


class ArticulationTrainer:
    def __init__(
        self,
        model_config_path: str,
        dataset_config_path: str,
        output_dir: str = "outputs/articulation",
        device: str = "cuda",
    ):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        with open(model_config_path, "r") as f:
            self.model_config = yaml.safe_load(f)
        with open(dataset_config_path, "r") as f:
            self.dataset_config = yaml.safe_load(f)

        train_cfg = self.model_config.get("training", {})

        self.model = build_model(self.model_config).to(device)
        self.criterion = ArticulationLoss(
            **self.model_config.get("loss", {})
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.get("lr", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )

        self.epochs = train_cfg.get("epochs", 100)
        self.batch_size = train_cfg.get("batch_size", 16)
        warmup_epochs = train_cfg.get("warmup_epochs", 5)

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.01, total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs - warmup_epochs,
        )
        self.scheduler = SequentialLR(
            self.optimizer, [warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            dataset_config_path, batch_size=self.batch_size,
        )

        self.metrics = JointMetrics()
        self.best_val_loss = float("inf")
        self.history = {"train": [], "val": []}

    def _to_device(self, batch: Dict) -> Dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                out[k] = [t.to(self.device) for t in v]
            else:
                out[k] = v
        return out

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        epoch_losses = {}
        count = 0

        for batch in self.train_loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()

            pred = self.model(batch)
            losses = self.criterion(pred, batch)

            losses["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v.item()
            count += 1

        return {k: v / max(count, 1) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict:
        self.model.eval()
        epoch_losses = {}
        count = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)
            pred = self.model(batch)
            losses = self.criterion(pred, batch)

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v.item()
            count += 1

        return {k: v / max(count, 1) for k, v in epoch_losses.items()}

    def train(self):
        print(f"Training {self.model_config['model']['name']} for {self.epochs} epochs")
        print(f"Train: {len(self.train_loader.dataset)} | Val: {len(self.val_loader.dataset)}")

        for epoch in range(self.epochs):
            t0 = time.time()
            train_losses = self.train_epoch(epoch)
            val_losses = self.validate()
            self.scheduler.step()

            self.history["train"].append(train_losses)
            self.history["val"].append(val_losses)

            if val_losses.get("total", float("inf")) < self.best_val_loss:
                self.best_val_loss = val_losses["total"]
                self.save_checkpoint("best.pt")

            elapsed = time.time() - t0
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:3d}/{self.epochs} | "
                    f"train_loss={train_losses.get('total', 0):.4f} | "
                    f"val_loss={val_losses.get('total', 0):.4f} | "
                    f"time={elapsed:.1f}s"
                )

        self.save_checkpoint("last.pt")
        self._save_history()
        print(f"Best val loss: {self.best_val_loss:.4f}")

    @torch.no_grad()
    def evaluate(self) -> Dict:
        self.model.eval()
        self.metrics.reset()

        for batch in self.test_loader:
            batch = self._to_device(batch)
            pred = self.model(batch)
            self.metrics.update(pred, batch)

        results = self.metrics.compute()
        results_path = os.path.join(self.output_dir, "test_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Test results saved to {results_path}")
        return results

    def save_checkpoint(self, name: str):
        path = os.path.join(self.output_dir, name)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.model_config,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])

    def _save_history(self):
        path = os.path.join(self.output_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
