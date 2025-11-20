"""Evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from .. import config
from ..models.base_model import Classifier


class CrossValidationResult:
    """Container for cross-validation statistics."""

    def __init__(self, scores: np.ndarray):
        self.scores = scores
        self.mean = float(np.mean(scores))
        self.std = float(np.std(scores))


def cross_validate_classifier(
    classifier: Classifier,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    folds: int = config.CV_FOLDS,
) -> CrossValidationResult:
    """Run cross-validation and return score statistics."""
    sk_model = getattr(classifier, "classifier", None)
    if sk_model is None:
        raise ValueError("Classifier must wrap an sklearn estimator for cross-validation.")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(sk_model, features, labels, cv=cv, n_jobs=config.N_JOBS)
    return CrossValidationResult(scores=scores)


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[int, float]:
    """Compute accuracy per label."""
    classes = np.unique(y_true)
    metrics: Dict[int, float] = {}
    for cls in classes:
        mask = y_true == cls
        metrics[int(cls)] = accuracy_score(y_true[mask], y_pred[mask])
    return metrics


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Confusion matrix helper."""
    return confusion_matrix(y_true, y_pred)


