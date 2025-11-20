"""Dataset loading utilities for EMNIST."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .. import config


class DatasetSplit:
    """Container for a dataset split."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images  # (n_samples, height, width)
        self.labels = labels  # (n_samples,)


class EmnistLoader:
    """Load and cache EMNIST letters data."""

    def __init__(
        self,
        split: str = "letters",
        source_dir: Path | None = None,
    ) -> None:
        if split != "letters":
            raise ValueError("Currently only the 'letters' split is supported.")

        self.split = split
        self.source_dir = source_dir or config.ARCHIVE_DIR

        config.ensure_directories()

    def _source_file(self, filename: str) -> Path:
        path = self.source_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected {filename} in {self.source_dir}. "
                "Please download and place the EMNIST dataset there."
            )
        return path

    def _ensure_cached(self) -> None:
        """Copy CSV files to the canonical raw data directory if needed."""
        mapping = {
            "emnist-letters-train.csv": config.EMNIST_LETTERS_TRAIN,
            "emnist-letters-test.csv": config.EMNIST_LETTERS_TEST,
            "emnist-letters-mapping.txt": config.EMNIST_LETTERS_MAPPING,
        }
        for filename, target_path in mapping.items():
            if not target_path.exists():
                source_path = self._source_file(filename)
                shutil.copy2(source_path, target_path)

    def load_split(self, split: str = "train", limit: int | None = None) -> DatasetSplit:
        """Load a specific split of the dataset."""
        self._ensure_cached()

        if split == "train":
            csv_path = config.EMNIST_LETTERS_TRAIN
        elif split == "test":
            csv_path = config.EMNIST_LETTERS_TEST
        else:
            raise ValueError("Split must be either 'train' or 'test'.")

        df = pd.read_csv(csv_path, header=None)
        if limit is not None:
            df = df.iloc[:limit]

        labels = df.iloc[:, 0].to_numpy(dtype=np.int8)
        pixels = df.iloc[:, 1:].to_numpy(dtype=np.uint8)

        images = pixels.reshape((-1, config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

        # EMNIST images are transposed and upside down; fix orientation.
        images = np.transpose(images, (0, 2, 1))
        images = np.flip(images, axis=2)

        return DatasetSplit(images=images, labels=labels)

    def load_train_test(
        self,
        limit_train: int | None = None,
        limit_test: int | None = None,
    ) -> Tuple[DatasetSplit, DatasetSplit]:
        """Convenience method to load both splits."""
        train_split = self.load_split("train", limit=limit_train)
        test_split = self.load_split("test", limit=limit_test)
        return train_split, test_split


