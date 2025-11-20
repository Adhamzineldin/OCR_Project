"""Plotting helpers for evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .. import config


def save_confusion_matrix(
    matrix: np.ndarray,
    labels: Iterable[str],
    *,
    title: str,
    output_path: Path,
) -> Path:
    """Save a confusion matrix heatmap."""
    config.ensure_directories()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def save_feature_importances(
    importances: np.ndarray,
    *,
    top_k: int = 20,
    title: str = "Feature Importances",
    output_path: Path,
) -> Optional[Path]:
    """Save a bar chart of the most important features."""
    if importances is None:
        return None

    config.ensure_directories()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    indices = np.argsort(importances)[::-1][:top_k]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=np.arange(len(indices)), y=importances[indices])
    plt.title(title)
    plt.xlabel("Feature Index (sorted)")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


