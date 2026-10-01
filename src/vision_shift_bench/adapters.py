"""Small, framework-independent classifier interfaces and reference adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class LogitPredictor(Protocol):
    """Interface required by the robustness evaluator."""

    def predict_logits(self, images: ArrayLike) -> NDArray[np.float64]:
        """Return one row of class logits for each input image."""


def _as_image_batch(images: ArrayLike) -> NDArray[np.float64]:
    batch = np.asarray(images, dtype=np.float64)
    if batch.ndim != 4 or batch.shape[0] == 0:
        raise ValueError("images must have shape (examples, height, width, channels)")
    if batch.shape[1] < 2 or batch.shape[2] < 2 or batch.shape[3] not in (1, 3):
        raise ValueError(
            "images must have spatial dimensions >= 2 and one or three channels"
        )
    if not np.all(np.isfinite(batch)):
        raise ValueError("images must contain only finite values")
    if np.any(batch < 0.0) or np.any(batch > 1.0):
        raise ValueError("image values must lie in [0, 1]")
    return batch


@dataclass(frozen=True)
class NearestCentroidClassifier:
    """Transparent reference classifier used to validate the evaluation pipeline.

    This is intentionally modest: it makes the benchmark executable without a
    deep-learning framework while preserving the same logit adapter boundary.
    """

    centroids: NDArray[np.float64]
    image_shape: tuple[int, int, int]
    logit_scale: float = 100.0

    @classmethod
    def fit(
        cls,
        images: ArrayLike,
        labels: ArrayLike,
        *,
        logit_scale: float = 100.0,
    ) -> "NearestCentroidClassifier":
        batch = _as_image_batch(images)
        targets = np.asarray(labels)
        if targets.ndim != 1 or targets.shape[0] != batch.shape[0]:
            raise ValueError("labels must contain one value per image")
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("labels must be integers")
        classes = np.unique(targets)
        if not np.array_equal(classes, np.arange(classes.size)) or classes.size < 2:
            raise ValueError("labels must include consecutive classes starting at zero")
        if not np.isfinite(logit_scale) or logit_scale <= 0.0:
            raise ValueError("logit_scale must be finite and positive")

        flat = batch.reshape(batch.shape[0], -1)
        centroids = np.stack([flat[targets == label].mean(axis=0) for label in classes])
        return cls(centroids=centroids, image_shape=batch.shape[1:], logit_scale=logit_scale)

    def predict_logits(self, images: ArrayLike) -> NDArray[np.float64]:
        batch = _as_image_batch(images)
        if batch.shape[1:] != self.image_shape:
            raise ValueError(
                f"expected image shape {self.image_shape}, received {batch.shape[1:]}"
            )
        flat = batch.reshape(batch.shape[0], -1)
        mean_squared_distance = np.mean(
            (flat[:, None, :] - self.centroids[None, :, :]) ** 2,
            axis=2,
        )
        return -self.logit_scale * mean_squared_distance


@dataclass(frozen=True)
class BlockMeanCentroidClassifier:
    """Nearest-centroid reference model using non-overlapping mean pooling."""

    centroids: NDArray[np.float64]
    image_shape: tuple[int, int, int]
    block_size: int
    logit_scale: float = 100.0

    @staticmethod
    def _features(
        batch: NDArray[np.float64], block_size: int
    ) -> NDArray[np.float64]:
        height, width = batch.shape[1:3]
        if height % block_size or width % block_size:
            raise ValueError("image dimensions must be divisible by block_size")
        pooled = batch.reshape(
            batch.shape[0],
            height // block_size,
            block_size,
            width // block_size,
            block_size,
            batch.shape[3],
        ).mean(axis=(2, 4))
        return pooled.reshape(batch.shape[0], -1)

    @classmethod
    def fit(
        cls,
        images: ArrayLike,
        labels: ArrayLike,
        *,
        block_size: int = 2,
        logit_scale: float = 100.0,
    ) -> "BlockMeanCentroidClassifier":
        batch = _as_image_batch(images)
        targets = np.asarray(labels)
        if targets.ndim != 1 or targets.shape[0] != batch.shape[0]:
            raise ValueError("labels must contain one value per image")
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("labels must be integers")
        classes = np.unique(targets)
        if not np.array_equal(classes, np.arange(classes.size)) or classes.size < 2:
            raise ValueError("labels must include consecutive classes starting at zero")
        if (
            not isinstance(block_size, int)
            or isinstance(block_size, bool)
            or block_size < 1
        ):
            raise ValueError("block_size must be a positive integer")
        if not np.isfinite(logit_scale) or logit_scale <= 0.0:
            raise ValueError("logit_scale must be finite and positive")

        features = cls._features(batch, block_size)
        centroids = np.stack(
            [features[targets == label].mean(axis=0) for label in classes]
        )
        return cls(
            centroids=centroids,
            image_shape=batch.shape[1:],
            block_size=block_size,
            logit_scale=logit_scale,
        )

    def predict_logits(self, images: ArrayLike) -> NDArray[np.float64]:
        batch = _as_image_batch(images)
        if batch.shape[1:] != self.image_shape:
            raise ValueError(
                f"expected image shape {self.image_shape}, received {batch.shape[1:]}"
            )
        features = self._features(batch, self.block_size)
        mean_squared_distance = np.mean(
            (features[:, None, :] - self.centroids[None, :, :]) ** 2,
            axis=2,
        )
        return -self.logit_scale * mean_squared_distance
