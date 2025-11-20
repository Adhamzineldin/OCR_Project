"""Command-line interface for running the OCR pipeline."""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, List

from . import config
from .pipeline import OcrPipeline
from .preprocessing.feature_engineering import HogFeatureExtractor, PcaFeatureExtractor, FeatureExtractor
from .data.loader import EmnistLoader


FEATURE_REGISTRY = {
    "hog": HogFeatureExtractor,
    "pca": PcaFeatureExtractor,
}


def build_feature_extractors(names: Iterable[str]) -> List[FeatureExtractor]:
    extractors: List[FeatureExtractor] = []
    for name in names:
        cls = FEATURE_REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"Unknown feature extractor '{name}'. Available: {list(FEATURE_REGISTRY)}")
        extractors.append(cls())
    return extractors


def command_train(args: argparse.Namespace) -> None:
    extractors = build_feature_extractors(args.features)
    pipeline = OcrPipeline(feature_extractors=extractors, loader=EmnistLoader())
    
    # Build model kwargs from CLI arguments
    model_kwargs = {}
    if hasattr(args, 'n_estimators') and args.n_estimators is not None:
        model_kwargs['n_estimators'] = args.n_estimators
    
    model, evaluation, cv_result, artifacts = pipeline.train_and_evaluate(
        model_name=args.model,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        run_cross_validation=not args.skip_cv,
        model_kwargs=model_kwargs if model_kwargs else None,
    )

    print(f"Model trained: {model.name}")
    print(f"Accuracy: {evaluation.accuracy:.4f}")
    if cv_result:
        print(f"Cross-validation mean: {cv_result.mean:.4f} (std: {cv_result.std:.4f})")
    print(f"Artifacts saved to: {config.ARTIFACTS_DIR}")
    print(f" - Model: {artifacts.model_path}")
    print(f" - Metrics: {artifacts.metrics_path}")
    print(f" - Confusion matrix: {artifacts.confusion_matrix_path}")
    if artifacts.feature_importances_path:
        print(f" - Feature importances: {artifacts.feature_importances_path}")


def command_prepare_data(_: argparse.Namespace) -> None:
    loader = EmnistLoader()
    loader.load_split("train", limit=1)  # triggers caching
    print("EMNIST files cached to raw data directory.")


def command_list_features(_: argparse.Namespace) -> None:
    print("Available feature extractors:")
    for key in FEATURE_REGISTRY:
        print(f" - {key}")


def command_list_models(_: argparse.Namespace) -> None:
    from .pipeline import MODEL_REGISTRY

    print("Available models:")
    for key in MODEL_REGISTRY:
        print(f" - {key}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCR pipeline CLI")
    subparsers = parser.add_subparsers(dest="command")

    prepare_parser = subparsers.add_parser("prepare-data", help="Cache EMNIST dataset to raw directory")
    prepare_parser.set_defaults(func=command_prepare_data)

    train_parser = subparsers.add_parser("train", help="Train and evaluate a model")
    train_parser.add_argument("--model", default="random_forest", help="Model name")
    train_parser.add_argument(
        "--features",
        nargs="+",
        default=["hog"],
        help="Feature extractor names",
    )
    train_parser.add_argument("--limit-train", type=int, default=None, help="Limit number of training samples")
    train_parser.add_argument("--limit-test", type=int, default=None, help="Limit number of test samples")
    train_parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation step")
    train_parser.add_argument("--n-estimators", type=int, default=None, help="Number of trees for random forest (default: 200)")
    train_parser.set_defaults(func=command_train)

    list_features_parser = subparsers.add_parser("list-features", help="List available feature extractors")
    list_features_parser.set_defaults(func=command_list_features)

    list_models_parser = subparsers.add_parser("list-models", help="List available models")
    list_models_parser.set_defaults(func=command_list_models)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()


