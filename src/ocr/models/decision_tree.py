"""Decision tree classifier wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier

from .base_model import Classifier
from .. import config


class DecisionTreeModel(Classifier):
    """Thin wrapper around scikit-learn's DecisionTreeClassifier."""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        criterion: str = "gini",
        random_state: int = config.RANDOM_STATE,
    ):
        self.name = "decision_tree"
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self.classifier = SkDecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_state,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.classifier.fit(features, labels)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.predict(features)

    def feature_importances(self) -> np.ndarray:
        return getattr(self.classifier, "feature_importances_", None)

    def get_params(self) -> Dict[str, Any]:
        return self.classifier.get_params()


