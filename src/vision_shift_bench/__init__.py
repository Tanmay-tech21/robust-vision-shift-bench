"""Controlled image-corruption and robustness evaluation utilities."""

from .adapters import BlockMeanCentroidClassifier, LogitPredictor, NearestCentroidClassifier
from .comparison import ComparisonPoint, ModelComparison, compare_models
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
from .reporting import (
    build_benchmark_snapshot,
    render_benchmark_card,
    write_benchmark_artifacts,
)
from .robustness import CorruptionCurve, SeverityAnalysis, SeverityPoint, analyse_severity_curves
from .subgroups import (
    GroupSnapshot,
    SubgroupAnalysis,
    SubgroupCurve,
    SubgroupPoint,
    SubgroupSnapshot,
    analyse_subgroups_under_shift,
    summarise_subgroups,
)
from .synthetic import make_pattern_dataset, make_reference_image

__all__ = [
    "BenchmarkResult",
    "BlockMeanCentroidClassifier",
    "ComparisonPoint",
    "ConditionResult",
    "CorruptionCurve",
    "CorruptionKind",
    "CorruptionSpec",
    "GroupSnapshot",
    "LogitPredictor",
    "NearestCentroidClassifier",
    "ModelComparison",
    "PredictionRecord",
    "ReliabilityAnalysis",
    "ReliabilityCurve",
    "ReliabilityPoint",
    "ReliabilitySnapshot",
    "SeverityAnalysis",
    "SeverityPoint",
    "SubgroupAnalysis",
    "SubgroupCurve",
    "SubgroupPoint",
    "SubgroupSnapshot",
    "analyse_severity_curves",
    "analyse_reliability_under_shift",
    "analyse_subgroups_under_shift",
    "apply_corruption",
    "build_benchmark_snapshot",
    "compare_models",
    "distortion_metrics",
    "evaluate_corruptions",
    "make_pattern_dataset",
    "make_reference_image",
    "peak_signal_to_noise_ratio",
    "root_mean_square_error",
    "render_benchmark_card",
    "summarise_prediction_records",
    "summarise_subgroups",
    "write_benchmark_artifacts",
]
