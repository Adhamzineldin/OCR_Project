"""Image preprocessing utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from skimage.transform import resize

from .. import config


class ImagePreprocessor:
    """Resize and normalize EMNIST images."""

    def __init__(
        self,
        target_size: tuple[int, int] = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
        normalize: bool = True,
    ):
        self.target_size = target_size
        self.normalize = normalize

    def transform(self, images: np.ndarray | Iterable[np.ndarray]) -> np.ndarray:
        """Resize (if needed) and normalize images to floats."""
        images_arr = np.asarray(list(images)) if not isinstance(images, np.ndarray) else images

        if images_arr.ndim != 3:
            raise ValueError(
                f"Expected input of shape (n_samples, height, width); got {images_arr.shape}"
            )

        # Resize only if shape differs
        if images_arr.shape[1:3] != self.target_size:
            resized = np.empty((images_arr.shape[0], *self.target_size), dtype=np.float32)
            for idx, img in enumerate(images_arr):
                resized[idx] = resize(img, self.target_size, anti_aliasing=True)
            images_arr = resized
        else:
            images_arr = images_arr.astype(np.float32)

        # Normalize to 0-1 range
        if self.normalize:
            images_arr /= 255.0

        return images_arr

    def flatten(self, images: np.ndarray) -> np.ndarray:
        """Flatten images to vectors."""
        processed = self.transform(images)
        return processed.reshape(processed.shape[0], -1)


