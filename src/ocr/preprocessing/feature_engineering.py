"""Feature extraction utilities for OCR."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

from .. import config


class FeatureExtractor:
    """Base class for feature extractors."""

    def fit(self, images: np.ndarray, labels: np.ndarray | None = None) -> "FeatureExtractor":
        """Fit the extractor to the data (optional for some extractors)."""
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        """Transform images to features."""
        raise NotImplementedError("Subclasses must implement transform")

    def fit_transform(
        self, images: np.ndarray, labels: np.ndarray | None = None
    ) -> np.ndarray:
        return self.fit(images, labels).transform(images)


class HogFeatureExtractor(FeatureExtractor):
    """Histogram of Oriented Gradients features."""

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
        block_norm: str = "L2-Hys",
    ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, images: np.ndarray, labels: np.ndarray | None = None) -> "HogFeatureExtractor":
        # HOG is stateless; nothing to fit.
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        if images.ndim != 3:
            raise ValueError("HOG extractor expects images with shape (n_samples, height, width)")

        hog_features = [
            hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                visualize=False,
                feature_vector=True,
            )
            for image in images
        ]
        return np.asarray(hog_features, dtype=np.float32)


class PcaFeatureExtractor(FeatureExtractor):
    """Principal component features extracted from flattened pixels."""

    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)

    def fit(self, images: np.ndarray, labels: np.ndarray | None = None) -> "PcaFeatureExtractor":
        if images.ndim != 3:
            raise ValueError("PCA extractor expects images with shape (n_samples, height, width)")

        flattened = images.reshape(images.shape[0], -1)
        scaled = self.scaler.fit_transform(flattened)
        self.pca.fit(scaled)
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        if images.ndim != 3:
            raise ValueError("PCA extractor expects images with shape (n_samples, height, width)")

        flattened = images.reshape(images.shape[0], -1)
        scaled = self.scaler.transform(flattened)
        components = self.pca.transform(scaled)
        return components.astype(np.float32)


class FeatureComposer:
    """Combine multiple feature extractors by concatenating their outputs."""

    def __init__(self, extractors: Sequence[FeatureExtractor]):
        self.extractors = extractors

    def fit(
        self,
        images: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> "FeatureComposer":
        for extractor in self.extractors:
            extractor.fit(images, labels)
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        features = [extractor.transform(images) for extractor in self.extractors]
        return np.concatenate(features, axis=1)

    def fit_transform(
        self,
        images: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> np.ndarray:
        self.fit(images, labels)
        return self.transform(images)


