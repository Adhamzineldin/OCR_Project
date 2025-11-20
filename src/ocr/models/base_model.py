"""Base classes for classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass
class EvaluationResult:
    """Stores evaluation metrics for a trained classifier."""

    accuracy: float
    report: str
    confusion_matrix: np.ndarray


class Classifier:
    """Base class for classifiers."""

    name: str = ""

    def fit(self, features: np.ndarray, labels: np.ndarray):
        """Train the classifier."""
        raise NotImplementedError("Subclasses must implement fit")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError("Subclasses must implement predict")

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> EvaluationResult:
        """Evaluate the classifier."""
        predictions = self.predict(features)
        acc = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, zero_division=0)
        matrix = confusion_matrix(labels, predictions)
        return EvaluationResult(accuracy=acc, report=report, confusion_matrix=matrix)

    def feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importances if available, else None."""
        return None

    def get_params(self) -> Dict[str, Any]:
        """Return model parameters for logging."""
        return {}


