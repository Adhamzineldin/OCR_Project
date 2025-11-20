"""High-level OCR training pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np

from . import config
from .data.loader import EmnistLoader, DatasetSplit
from .evaluation.metrics import (
    cross_validate_classifier,
    CrossValidationResult,
    per_class_accuracy,
)
from .models.base_model import Classifier, EvaluationResult
from .models.decision_tree import DecisionTreeModel
from .models.random_forest import RandomForestModel
from .preprocessing.feature_engineering import FeatureComposer, FeatureExtractor
from .preprocessing.image_processor import ImagePreprocessor
from .visualization.plots import save_confusion_matrix, save_feature_importances


MODEL_REGISTRY: Dict[str, type[Classifier]] = {
    "decision_tree": DecisionTreeModel,
    "random_forest": RandomForestModel,
}


class TrainingArtifacts:
    """Paths to generated artifacts."""

    def __init__(
        self,
        model_path: Path,
        confusion_matrix_path: Path,
        feature_importances_path: Path | None,
        metrics_path: Path,
    ):
        self.model_path = model_path
        self.confusion_matrix_path = confusion_matrix_path
        self.feature_importances_path = feature_importances_path
        self.metrics_path = metrics_path


class OcrPipeline:
    """Coordinate data loading, preprocessing, and model training."""

    def __init__(
        self,
        feature_extractors: Iterable[FeatureExtractor],
        loader: EmnistLoader = None,
        image_preprocessor: ImagePreprocessor = None,
    ):
        self.feature_extractors = feature_extractors
        self.loader = loader if loader is not None else EmnistLoader()
        self.image_preprocessor = image_preprocessor if image_preprocessor is not None else ImagePreprocessor()
        self.composer = FeatureComposer(tuple(self.feature_extractors))

    def _prepare_features(
        self,
        dataset_split: DatasetSplit,
        *,
        fit_extractors: bool,
    ) -> np.ndarray:
        images = self.image_preprocessor.transform(dataset_split.images)
        if fit_extractors:
            features = self.composer.fit_transform(images, dataset_split.labels)
        else:
            features = self.composer.transform(images)
        return features

    def prepare_datasets(
        self,
        *,
        limit_train: int | None = None,
        limit_test: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and transform the dataset into features."""
        train_split, test_split = self.loader.load_train_test(
            limit_train=limit_train,
            limit_test=limit_test,
        )
        X_train = self._prepare_features(train_split, fit_extractors=True)
        X_test = self._prepare_features(test_split, fit_extractors=False)
        y_train = train_split.labels
        y_test = test_split.labels
        return X_train, X_test, y_train, y_test

    def build_model(self, model_name: str, **model_kwargs) -> Classifier:
        model_cls = MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")
        return model_cls(**model_kwargs)

    def train_and_evaluate(
        self,
        model_name: str,
        *,
        limit_train: int | None = None,
        limit_test: int | None = None,
        run_cross_validation: bool = True,
        model_kwargs: dict | None = None,
    ) -> Tuple[Classifier, EvaluationResult, CrossValidationResult | None, TrainingArtifacts]:
        """Train the requested model and persist artifacts."""
        features_train, features_test, labels_train, labels_test = self.prepare_datasets(
            limit_train=limit_train,
            limit_test=limit_test,
        )

        model = self.build_model(model_name, **(model_kwargs or {}))
        model.fit(features_train, labels_train)
        evaluation = model.evaluate(features_test, labels_test)

        cv_result = None
        if run_cross_validation:
            cv_result = cross_validate_classifier(model, features_train, labels_train)

        artifacts = self._persist_artifacts(
            model=model,
            evaluation=evaluation,
            cv_result=cv_result,
            labels=labels_test,
            predictions=model.predict(features_test),
        )
        return model, evaluation, cv_result, artifacts

    def _persist_artifacts(
        self,
        *,
        model: Classifier,
        evaluation: EvaluationResult,
        cv_result: CrossValidationResult | None,
        labels: np.ndarray,
        predictions: np.ndarray,
    ) -> TrainingArtifacts:
        config.ensure_directories()

        model_filename = f"{model.name}.joblib"
        metrics_filename = f"{model.name}_metrics.json"
        confusion_filename = f"{model.name}_confusion.png"
        feature_importance_filename = f"{model.name}_feature_importances.png"

        model_path = config.ARTIFACTS_DIR / model_filename
        metrics_path = config.ARTIFACTS_DIR / metrics_filename
        confusion_path = config.ARTIFACTS_DIR / confusion_filename
        feature_path = config.ARTIFACTS_DIR / feature_importance_filename

        joblib.dump(
            {
                "model": model,
                "composer": self.composer,
                "image_preprocessor": self.image_preprocessor,
            },
            model_path,
        )

        mapping_path = config.EMNIST_LETTERS_MAPPING
        if mapping_path.exists():
            label_map = self._load_label_mapping(mapping_path)
            labels_text = [label_map[int(lbl)] for lbl in sorted(set(labels))]
        else:
            labels_text = [str(i) for i in sorted(set(labels))]

        save_confusion_matrix(
            matrix=evaluation.confusion_matrix,
            labels=labels_text,
            title=f"{model.name} Confusion Matrix",
            output_path=confusion_path,
        )

        feature_path_saved = save_feature_importances(
            importances=model.feature_importances(),
            title=f"{model.name} Feature Importances",
            output_path=feature_path,
        )

        metrics_payload = {
            "accuracy": evaluation.accuracy,
            "classification_report": evaluation.report,
            "cross_validation": {
                "mean": cv_result.mean if cv_result else None,
                "std": cv_result.std if cv_result else None,
                "scores": cv_result.scores.tolist() if cv_result else None,
            },
            "per_class_accuracy": per_class_accuracy(labels, predictions),
            "model_params": model.get_params(),
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        return TrainingArtifacts(
            model_path=model_path,
            confusion_matrix_path=confusion_path,
            feature_importances_path=feature_path_saved,
            metrics_path=metrics_path,
        )

    @staticmethod
    def _load_label_mapping(path: Path) -> Dict[int, str]:
        """Load EMNIST mapping file."""
        mapping: Dict[int, str] = {}
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    idx = int(parts[0])
                    code = int(parts[-1])
                except ValueError:
                    continue
                mapping[idx] = chr(code)
        return mapping


