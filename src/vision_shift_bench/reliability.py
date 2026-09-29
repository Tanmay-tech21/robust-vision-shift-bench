"""Calibration and selective-prediction diagnostics under corruption."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Sequence

import numpy as np

from .evaluation import BenchmarkResult, ConditionResult, PredictionRecord


@dataclass(frozen=True)
class ReliabilitySnapshot:
    """Probabilistic and selective metrics for one evaluation condition."""

    accuracy: float
    mean_confidence: float
    negative_log_likelihood: float
    expected_calibration_error: float
    area_under_risk_coverage: float
    risk_at_target_coverage: float


@dataclass(frozen=True)
class ReliabilityPoint:
    """Reliability snapshot attached to a corruption severity."""

    severity: int
    metrics: ReliabilitySnapshot


@dataclass(frozen=True)
class ReliabilityCurve:
    """Reliability trajectory and first operational threshold violations."""

    kind: str
    first_ece_failure_severity: int | None
    first_selective_risk_failure_severity: int | None
    points: tuple[ReliabilityPoint, ...]


@dataclass(frozen=True)
class ReliabilityAnalysis:
    """Clean baseline and per-corruption reliability trajectories."""

    clean: ReliabilitySnapshot
    bin_count: int
    target_coverage: float
    ece_ceiling: float
    selective_risk_ceiling: float
    curves: tuple[ReliabilityCurve, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def summarise_prediction_records(
    records: Sequence[PredictionRecord],
    *,
    shifted: bool,
    bin_count: int = 15,
    target_coverage: float = 0.8,
) -> ReliabilitySnapshot:
    """Calculate proper scoring, calibration, and selective-risk metrics."""
    if not records:
        raise ValueError("prediction records must not be empty")
    if not isinstance(bin_count, int) or isinstance(bin_count, bool) or bin_count < 2:
        raise ValueError("bin_count must be an integer of at least two")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must lie in (0, 1]")

    if shifted:
        predictions = np.fromiter(
            (record.corrupted_prediction for record in records), dtype=np.int64
        )
        confidences = np.fromiter(
            (record.corrupted_confidence for record in records), dtype=np.float64
        )
        true_probabilities = np.fromiter(
            (record.corrupted_true_probability for record in records), dtype=np.float64
        )
    else:
        predictions = np.fromiter(
            (record.clean_prediction for record in records), dtype=np.int64
        )
        confidences = np.fromiter(
            (record.clean_confidence for record in records), dtype=np.float64
        )
        true_probabilities = np.fromiter(
            (record.clean_true_probability for record in records), dtype=np.float64
        )
    labels = np.fromiter((record.label for record in records), dtype=np.int64)
    correct = predictions == labels

    bin_indices = np.minimum((confidences * bin_count).astype(np.int64), bin_count - 1)
    ece = 0.0
    for bin_index in range(bin_count):
        selected = bin_indices == bin_index
        if np.any(selected):
            weight = float(selected.mean())
            ece += weight * abs(float(correct[selected].mean() - confidences[selected].mean()))

    # Index is the secondary key, giving deterministic ordering when confidence
    # ties would otherwise make selective metrics implementation-dependent.
    order = np.lexsort((np.arange(len(records)), -confidences))
    ordered_errors = (~correct[order]).astype(np.float64)
    risk_curve = np.cumsum(ordered_errors) / np.arange(1, len(records) + 1)
    accepted = ceil(target_coverage * len(records))

    return ReliabilitySnapshot(
        accuracy=float(correct.mean()),
        mean_confidence=float(confidences.mean()),
        negative_log_likelihood=float(
            -np.log(np.clip(true_probabilities, np.finfo(np.float64).tiny, 1.0)).mean()
        ),
        expected_calibration_error=ece,
        area_under_risk_coverage=float(risk_curve.mean()),
        risk_at_target_coverage=float(risk_curve[accepted - 1]),
    )


def analyse_reliability_under_shift(
    result: BenchmarkResult,
    *,
    bin_count: int = 15,
    target_coverage: float = 0.8,
    ece_ceiling: float = 0.1,
    selective_risk_ceiling: float = 0.1,
) -> ReliabilityAnalysis:
    """Measure calibration and selective prediction over complete severity curves."""
    if not 0.0 <= ece_ceiling <= 1.0:
        raise ValueError("ece_ceiling must lie in [0, 1]")
    if not 0.0 <= selective_risk_ceiling <= 1.0:
        raise ValueError("selective_risk_ceiling must lie in [0, 1]")
    if not result.conditions:
        raise ValueError("at least one corruption condition is required")

    clean = summarise_prediction_records(
        result.conditions[0].records,
        shifted=False,
        bin_count=bin_count,
        target_coverage=target_coverage,
    )
    grouped: dict[str, list[ConditionResult]] = defaultdict(list)
    for condition in result.conditions:
        grouped[condition.kind].append(condition)

    curves: list[ReliabilityCurve] = []
    for kind in sorted(grouped):
        conditions = sorted(grouped[kind], key=lambda condition: condition.severity)
        if [condition.severity for condition in conditions] != list(range(1, 6)):
            raise ValueError(f"{kind} must contain each severity from 1 to 5 exactly once")
        points = tuple(
            ReliabilityPoint(
                severity=condition.severity,
                metrics=summarise_prediction_records(
                    condition.records,
                    shifted=True,
                    bin_count=bin_count,
                    target_coverage=target_coverage,
                ),
            )
            for condition in conditions
        )
        curves.append(
            ReliabilityCurve(
                kind=kind,
                first_ece_failure_severity=next(
                    (
                        point.severity
                        for point in points
                        if point.metrics.expected_calibration_error > ece_ceiling
                    ),
                    None,
                ),
                first_selective_risk_failure_severity=next(
                    (
                        point.severity
                        for point in points
                        if point.metrics.risk_at_target_coverage
                        > selective_risk_ceiling
                    ),
                    None,
                ),
                points=points,
            )
        )

    return ReliabilityAnalysis(
        clean=clean,
        bin_count=bin_count,
        target_coverage=target_coverage,
        ece_ceiling=ece_ceiling,
        selective_risk_ceiling=selective_risk_ceiling,
        curves=tuple(curves),
    )
