"""Image-space diagnostics for corruption validation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _paired_images(reference: ArrayLike, candidate: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    expected = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(candidate, dtype=np.float64)
    if expected.shape != observed.shape or expected.size == 0:
        raise ValueError("reference and candidate must have the same non-empty shape")
    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(observed)):
        raise ValueError("images must contain only finite values")
    return expected, observed


def root_mean_square_error(reference: ArrayLike, candidate: ArrayLike) -> float:
    """Return pixel-wise RMSE in normalised intensity units."""
    expected, observed = _paired_images(reference, candidate)
    return float(np.sqrt(np.mean(np.square(expected - observed))))


def peak_signal_to_noise_ratio(reference: ArrayLike, candidate: ArrayLike) -> float:
    """Return PSNR in decibels for images with a unit intensity range."""
    rmse = root_mean_square_error(reference, candidate)
    if rmse == 0.0:
        return float("inf")
    return float(-20.0 * np.log10(rmse))


def distortion_metrics(reference: ArrayLike, candidate: ArrayLike) -> dict[str, float]:
    """Return a compact, JSON-compatible distortion summary."""
    return {
        "rmse": root_mean_square_error(reference, candidate),
        "psnr_db": peak_signal_to_noise_ratio(reference, candidate),
    }
