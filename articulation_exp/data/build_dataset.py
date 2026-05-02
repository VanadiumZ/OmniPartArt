"""
Build data loaders from configuration.
"""
import yaml
from torch.utils.data import DataLoader
from .dataset_articulation import ArticulationDataset


def build_dataloaders(config_path: str, batch_size: int = 16, num_workers: int = 4):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["dataset"]

    common_kwargs = dict(
        root_dir=cfg["root_dir"],
        categories=cfg.get("categories"),
        max_parts=cfg.get("max_parts", 10),
        single_moving_part=cfg.get("single_moving_part", True),
        num_frames=cfg["video"].get("num_frames", 8),
    )

    train_ds = ArticulationDataset(split="train", **common_kwargs)
    val_ds = ArticulationDataset(split="val", **common_kwargs)
    test_ds = ArticulationDataset(split="test", **common_kwargs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=ArticulationDataset.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=ArticulationDataset.collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=ArticulationDataset.collate_fn,
    )

    return train_loader, val_loader, test_loader
