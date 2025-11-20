"""Centralised configuration for the OCR project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

# EMNIST files (assuming CSV versions saved under RAW_DATA_DIR)
EMNIST_LETTERS_TRAIN = RAW_DATA_DIR / "emnist-letters-train.csv"
EMNIST_LETTERS_TEST = RAW_DATA_DIR / "emnist-letters-test.csv"
EMNIST_LETTERS_MAPPING = RAW_DATA_DIR / "emnist-letters-mapping.txt"

# Default image parameters
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

# Random seeds and cross-validation defaults
RANDOM_STATE = 42
N_JOBS = -1
CV_FOLDS = 5


def ensure_directories() -> None:
    """Ensure that required directories exist."""
    for directory in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


