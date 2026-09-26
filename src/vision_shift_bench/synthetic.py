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
