"""Random forest classifier wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier

from .base_model import Classifier
from .. import config


class RandomForestModel(Classifier):
    """Thin wrapper around scikit-learn's RandomForestClassifier."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        n_jobs: int = config.N_JOBS,
        random_state: int = config.RANDOM_STATE,
    ):
        self.name = "random_forest"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.classifier = SkRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.classifier.fit(features, labels)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.predict(features)

    def feature_importances(self) -> np.ndarray:
        return self.classifier.feature_importances_

    def get_params(self) -> Dict[str, Any]:
        return self.classifier.get_params()


