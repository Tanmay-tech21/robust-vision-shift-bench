"""Synthetic reference images used only to validate the benchmark machinery."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_reference_image(*, size: int = 96) -> NDArray[np.float64]:
    """Create a structured RGB fixture containing gradients, edges, and a disc."""
    if not isinstance(size, int) or isinstance(size, bool) or size < 16:
        raise ValueError("size must be an integer of at least 16")

    axis = np.linspace(0.0, 1.0, size)
    x, y = np.meshgrid(axis, axis)
    checker = ((np.floor(x * 8) + np.floor(y * 8)) % 2).astype(np.float64)
    disc = ((x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.18**2).astype(np.float64)
    return np.stack((x, y, 0.25 + 0.35 * checker + 0.35 * disc), axis=-1).clip(0.0, 1.0)


def make_pattern_dataset(
    *, samples_per_class: int = 80, size: int = 24, seed: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Create a three-class texture fixture for end-to-end pipeline checks."""
    if not isinstance(samples_per_class, int) or isinstance(samples_per_class, bool):
        raise ValueError("samples_per_class must be an integer")
    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least two")
    if not isinstance(size, int) or isinstance(size, bool) or size < 16:
        raise ValueError("size must be an integer of at least 16")

    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, size, endpoint=False)
    x, y = np.meshgrid(axis, axis)
    prototypes = (
        0.5 + 0.32 * np.sin(2.0 * np.pi * 4.0 * x),
        0.5 + 0.32 * np.sin(2.0 * np.pi * 4.0 * y),
        0.5 + 0.32 * np.sin(2.0 * np.pi * 3.0 * (x + y)),
    )

    images: list[NDArray[np.float64]] = []
    labels: list[int] = []
    for label, prototype in enumerate(prototypes):
        for _ in range(samples_per_class):
            shifted = np.roll(
                prototype,
                shift=(int(rng.integers(-1, 2)), int(rng.integers(-1, 2))),
                axis=(0, 1),
            )
            contrast = float(rng.uniform(0.88, 1.12))
            offset = float(rng.uniform(-0.025, 0.025))
            noisy = 0.5 + contrast * (shifted - 0.5) + offset
            noisy += rng.normal(0.0, 0.02, size=noisy.shape)
            images.append(np.repeat(np.clip(noisy, 0.0, 1.0)[..., None], 3, axis=2))
            labels.append(label)

    order = rng.permutation(len(labels))
    return np.stack(images)[order], np.asarray(labels, dtype=np.int64)[order]
