"""
Entry point for training all trainable baselines and fusion models.

Usage:
    python -m articulation_exp.train.train_all \\
        --model_config configs/model/baseline_3d.yaml \\
        --dataset_config configs/dataset/partnet_mobility.yaml \\
        --output_dir outputs/baseline_3d
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from articulation_exp.train.trainer import ArticulationTrainer


def main():
    parser = argparse.ArgumentParser(description="Train articulation models")
    parser.add_argument(
        "--model_config", type=str, required=True,
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--dataset_config", type=str,
        default="articulation_exp/configs/dataset/partnet_mobility.yaml",
        help="Path to dataset config YAML",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/articulation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    trainer = ArticulationTrainer(
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        output_dir=args.output_dir,
        device=args.device,
    )

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    if args.eval_only:
        results = trainer.evaluate()
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
    else:
        trainer.train()
        results = trainer.evaluate()
        print("\nFinal Test Results:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
