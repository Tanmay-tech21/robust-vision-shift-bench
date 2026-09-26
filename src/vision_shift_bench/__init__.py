"""Controlled image-corruption and robustness evaluation utilities."""

from .corruptions import CorruptionKind, CorruptionSpec, apply_corruption
from .metrics import distortion_metrics, peak_signal_to_noise_ratio, root_mean_square_error
from .synthetic import make_reference_image

__all__ = [
    "CorruptionKind",
    "CorruptionSpec",
    "apply_corruption",
    "distortion_metrics",
    "make_reference_image",
    "peak_signal_to_noise_ratio",
    "root_mean_square_error",
]
