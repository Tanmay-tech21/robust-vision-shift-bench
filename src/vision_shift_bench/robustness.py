"""Severity-curve summaries and paired uncertainty estimates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .evaluation import BenchmarkResult, ConditionResult


@dataclass(frozen=True)
class SeverityPoint:
    """One corruption severity with uncertainty around paired degradation."""

    severity: int
    accuracy: float
    accuracy_drop: float
    accuracy_drop_lower: float
    accuracy_drop_upper: float
    relative_error_multiplier: float | None
    prediction_flip_rate: float


@dataclass(frozen=True)
class CorruptionCurve:
    """Comparable summary of performance across all five severity levels."""

    kind: str
    seed: int
    normalized_degradation_auc: float
    normalized_accuracy_retention_auc: float
    failure_severity: int | None
    points: tuple[SeverityPoint, ...]


@dataclass(frozen=True)
class SeverityAnalysis:
    """Severity curves sharing one clean baseline and deployment threshold."""

    clean_accuracy: float
    accuracy_floor: float
    confidence_level: float
    bootstrap_resamples: int
    curves: tuple[CorruptionCurve, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def _paired_degradation_interval(
    condition: ConditionResult,
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if not condition.records:
        raise ValueError("condition records are required for paired bootstrap intervals")
    clean_correct = np.fromiter(
        (record.clean_prediction == record.label for record in condition.records),
        dtype=np.float64,
    )
    corrupted_correct = np.fromiter(
        (record.corrupted_prediction == record.label for record in condition.records),
        dtype=np.float64,
    )
    paired_difference = clean_correct - corrupted_correct
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, paired_difference.size, size=(resamples, paired_difference.size))
    bootstrap_degradation = paired_difference[indices].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_degradation, (tail, 1.0 - tail))
    return float(lower), float(upper)


def _normalised_trapezoid(values: np.ndarray, clean_accuracy: float) -> float:
    severities = np.arange(values.size, dtype=np.float64)
    area = np.sum(np.diff(severities) * (values[:-1] + values[1:]) / 2.0)
    return float(area / ((values.size - 1) * clean_accuracy))


def analyse_severity_curves(
    result: BenchmarkResult,
    *,
    accuracy_floor: float = 0.85,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 1000,
    seed: int = 0,
) -> SeverityAnalysis:
    """Build complete severity curves with paired bootstrap intervals.

    Normalised degradation AUC integrates absolute accuracy loss from severity
    zero through five, then divides by the clean-accuracy rectangle. Its
    complement is normalised accuracy retention, making the scale comparable
    across classifiers with different clean performance.
    """
    if not 0.0 < accuracy_floor <= 1.0:
        raise ValueError("accuracy_floor must lie in (0, 1]")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if (
        not isinstance(bootstrap_resamples, int)
        or isinstance(bootstrap_resamples, bool)
        or bootstrap_resamples < 2
    ):
        raise ValueError("bootstrap_resamples must be an integer of at least two")
    if result.clean_accuracy <= 0.0:
        raise ValueError("clean accuracy must be positive for normalisation")

    grouped: dict[str, list[ConditionResult]] = defaultdict(list)
    for condition in result.conditions:
        grouped[condition.kind].append(condition)
    if not grouped:
        raise ValueError("at least one corruption condition is required")

    clean_error = 1.0 - result.clean_accuracy
    curves: list[CorruptionCurve] = []
    for kind_index, kind in enumerate(sorted(grouped)):
        conditions = sorted(grouped[kind], key=lambda condition: condition.severity)
        if [condition.severity for condition in conditions] != list(range(1, 6)):
            raise ValueError(f"{kind} must contain each severity from 1 to 5 exactly once")
        seeds = {condition.seed for condition in conditions}
        if len(seeds) != 1:
            raise ValueError(f"{kind} must use one shared seed across severities")

        points: list[SeverityPoint] = []
        for point_index, condition in enumerate(conditions):
            lower, upper = _paired_degradation_interval(
                condition,
                confidence_level=confidence_level,
                resamples=bootstrap_resamples,
                seed=seed + 10_000 * kind_index + point_index,
            )
            corrupted_error = 1.0 - condition.accuracy
            error_multiplier = corrupted_error / clean_error if clean_error > 0.0 else None
            points.append(
                SeverityPoint(
                    severity=condition.severity,
                    accuracy=condition.accuracy,
                    accuracy_drop=condition.accuracy_drop,
                    accuracy_drop_lower=lower,
                    accuracy_drop_upper=upper,
                    relative_error_multiplier=error_multiplier,
                    prediction_flip_rate=condition.prediction_flip_rate,
                )
            )

        accuracies = np.asarray(
            [result.clean_accuracy, *(point.accuracy for point in points)], dtype=np.float64
        )
        degradation = result.clean_accuracy - accuracies
        degradation_auc = _normalised_trapezoid(degradation, result.clean_accuracy)
        retention_auc = _normalised_trapezoid(accuracies, result.clean_accuracy)
        failure_severity = next(
            (point.severity for point in points if point.accuracy < accuracy_floor), None
        )
        curves.append(
            CorruptionCurve(
                kind=kind,
                seed=conditions[0].seed,
                normalized_degradation_auc=degradation_auc,
                normalized_accuracy_retention_auc=retention_auc,
                failure_severity=failure_severity,
                points=tuple(points),
            )
        )

    return SeverityAnalysis(
        clean_accuracy=result.clean_accuracy,
        accuracy_floor=accuracy_floor,
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        curves=tuple(curves),
    )
