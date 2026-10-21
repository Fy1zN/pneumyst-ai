"""Datasets over the preprocessed 320 px images."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ml.training.augment import AugConfig, augment

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def load_processed(root: Path, source: str) -> pd.DataFrame:
    """Rows of <source>_processed.csv that preprocessed successfully."""
    df = pd.read_csv(root / "processed" / "manifests" / f"{source}_processed.csv", low_memory=False)
    return df[df["proc_status"] == "ok"].reset_index(drop=True)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HxW gray -> float 3xHxW with fixed ImageNet normalization."""
    x = np.repeat(img.astype(np.float32)[None] / 255.0, 3, axis=0)
    return torch.from_numpy((x - IMAGENET_MEAN) / IMAGENET_STD)


class XrayDataset(Dataset):
    """Returns (image tensor, label vector). Unknown labels are NaN and are masked out of the loss."""

    def __init__(self, df: pd.DataFrame, root: Path, label_cols: list[str], aug: AugConfig | None = None):
        self.paths = [str(Path(root) / p) for p in df["processed_path"]]
        self.labels = df[label_cols].to_numpy(dtype=np.float32)
        self.aug = aug

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[i]) as im:
            img = np.asarray(im.convert("L"))
        if self.aug is not None:
            # Drawn from torch's RNG so each worker and each epoch gets different augmentations.
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
            img = augment(img, rng, self.aug)
        return to_tensor(img), torch.tensor(self.labels[i])