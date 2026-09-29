"""Controlled image-corruption and robustness evaluation utilities."""

from .adapters import LogitPredictor, NearestCentroidClassifier
from .corruptions import CorruptionKind, CorruptionSpec, apply_corruption
from .evaluation import BenchmarkResult, ConditionResult, PredictionRecord, evaluate_corruptions
from .metrics import distortion_metrics, peak_signal_to_noise_ratio, root_mean_square_error
from .reliability import (
    ReliabilityAnalysis,
    ReliabilityCurve,
    ReliabilityPoint,
    ReliabilitySnapshot,
    analyse_reliability_under_shift,
    summarise_prediction_records,
)
from .robustness import CorruptionCurve, SeverityAnalysis, SeverityPoint, analyse_severity_curves
from .synthetic import make_pattern_dataset, make_reference_image

__all__ = [
    "BenchmarkResult",
    "ConditionResult",
    "CorruptionCurve",
    "CorruptionKind",
    "CorruptionSpec",
    "LogitPredictor",
    "NearestCentroidClassifier",
    "PredictionRecord",
    "ReliabilityAnalysis",
    "ReliabilityCurve",
    "ReliabilityPoint",
    "ReliabilitySnapshot",
    "SeverityAnalysis",
    "SeverityPoint",
    "analyse_severity_curves",
    "analyse_reliability_under_shift",
    "apply_corruption",
    "distortion_metrics",
    "evaluate_corruptions",
    "make_pattern_dataset",
    "make_reference_image",
    "peak_signal_to_noise_ratio",
    "root_mean_square_error",
    "summarise_prediction_records",
]
