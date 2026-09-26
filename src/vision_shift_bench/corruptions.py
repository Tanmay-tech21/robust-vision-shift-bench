"""Deterministic, severity-controlled image corruptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageFilter

CorruptionKind = Literal["brightness", "contrast", "gaussian_noise", "gaussian_blur"]

_BRIGHTNESS_FACTORS = (0.85, 0.70, 0.55, 0.40, 0.25)
_CONTRAST_FACTORS = (0.85, 0.70, 0.55, 0.40, 0.25)
_NOISE_STANDARD_DEVIATIONS = (0.03, 0.06, 0.10, 0.16, 0.24)
_BLUR_RADII = (0.5, 1.0, 1.5, 2.5, 4.0)


@dataclass(frozen=True)
class CorruptionSpec:
    """Fully specified corruption for reproducible evaluation."""

    kind: CorruptionKind
    severity: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {
            "brightness",
            "contrast",
            "gaussian_noise",
            "gaussian_blur",
        }:
            raise ValueError("unsupported corruption kind")
        if not isinstance(self.severity, int) or isinstance(self.severity, bool):
            raise ValueError("severity must be an integer from 1 to 5")
        if not 1 <= self.severity <= 5:
            raise ValueError("severity must be an integer from 1 to 5")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")


def _as_image(image: ArrayLike) -> NDArray[np.float64]:
    pixels = np.asarray(image, dtype=np.float64)
    if pixels.ndim not in (2, 3) or pixels.shape[0] < 2 or pixels.shape[1] < 2:
        raise ValueError("image must have shape (height, width) or (height, width, channels)")
    if pixels.ndim == 3 and pixels.shape[2] not in (1, 3):
        raise ValueError("image must contain one or three channels")
    if not np.all(np.isfinite(pixels)):
        raise ValueError("image must contain only finite values")
    if np.any(pixels < 0.0) or np.any(pixels > 1.0):
        raise ValueError("image values must lie in [0, 1]")
    return pixels


def _gaussian_blur(image: NDArray[np.float64], radius: float) -> NDArray[np.float64]:
    squeeze_channel = image.ndim == 3 and image.shape[2] == 1
    prepared = image[..., 0] if squeeze_channel else image
    uint8_image = np.rint(prepared * 255.0).astype(np.uint8)
    blurred = Image.fromarray(uint8_image).filter(ImageFilter.GaussianBlur(radius=radius))
    result = np.asarray(blurred, dtype=np.float64) / 255.0
    return result[..., None] if squeeze_channel else result


def apply_corruption(image: ArrayLike, spec: CorruptionSpec) -> NDArray[np.float64]:
    """Apply one corruption while preserving image shape and the [0, 1] range."""
    pixels = _as_image(image)
    level = spec.severity - 1

    if spec.kind == "brightness":
        corrupted = pixels * _BRIGHTNESS_FACTORS[level]
    elif spec.kind == "contrast":
        factor = _CONTRAST_FACTORS[level]
        channel_axes = (0, 1) if pixels.ndim == 3 else None
        mean = pixels.mean(axis=channel_axes, keepdims=True)
        corrupted = mean + factor * (pixels - mean)
    elif spec.kind == "gaussian_noise":
        # The same seed across severities scales one noise field rather than
        # confounding severity with a newly sampled pattern.
        noise = np.random.default_rng(spec.seed).normal(size=pixels.shape)
        corrupted = pixels + _NOISE_STANDARD_DEVIATIONS[level] * noise
    else:
        corrupted = _gaussian_blur(pixels, _BLUR_RADII[level])

    return np.clip(corrupted, 0.0, 1.0).astype(np.float64, copy=False)
