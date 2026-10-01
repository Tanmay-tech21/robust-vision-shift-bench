"""Paired comparisons between classifiers evaluated on identical examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from .evaluation import BenchmarkResult, PredictionRecord


@dataclass(frozen=True)
class ComparisonPoint:
    """Paired accuracy difference for one evaluation condition."""

    kind: str
    severity: int
    baseline_accuracy: float
    candidate_accuracy: float
    accuracy_difference: float
    difference_lower: float
    difference_upper: float
    baseline_only_correct: int
    candidate_only_correct: int
    evidence: str


@dataclass(frozen=True)
class ModelComparison:
    """Cross-model report whose uncertainty preserves example pairing."""

    baseline_name: str
    candidate_name: str
    confidence_level: float
    bootstrap_resamples: int
    baseline_supported_conditions: int
    candidate_supported_conditions: int
    inconclusive_conditions: int
    points: tuple[ComparisonPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def _correctness(
    records: Sequence[PredictionRecord], *, shifted: bool
) -> np.ndarray:
    if shifted:
        return np.fromiter(
            (record.corrupted_prediction == record.label for record in records),
            dtype=np.float64,
        )
    return np.fromiter(
        (record.clean_prediction == record.label for record in records),
        dtype=np.float64,
    )


def _validate_pairing(
    baseline: Sequence[PredictionRecord], candidate: Sequence[PredictionRecord]
) -> None:
    baseline_keys = [(record.index, record.label) for record in baseline]
    candidate_keys = [(record.index, record.label) for record in candidate]
    if baseline_keys != candidate_keys:
        raise ValueError(
            "model results must contain identically ordered example indices and labels"
        )


def _comparison_point(
    kind: str,
    severity: int,
    baseline_records: Sequence[PredictionRecord],
    candidate_records: Sequence[PredictionRecord],
    *,
    shifted: bool,
    confidence_level: float,
    bootstrap_resamples: int,
    seed: int,
) -> ComparisonPoint:
    _validate_pairing(baseline_records, candidate_records)
    baseline_correct = _correctness(baseline_records, shifted=shifted)
    candidate_correct = _correctness(candidate_records, shifted=shifted)
    paired_difference = candidate_correct - baseline_correct
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        paired_difference.size,
        size=(bootstrap_resamples, paired_difference.size),
    )
    bootstrap_difference = paired_difference[indices].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_difference, (tail, 1.0 - tail))
    evidence = (
        "candidate"
        if lower > 0.0
        else "baseline"
        if upper < 0.0
        else "inconclusive"
    )
    return ComparisonPoint(
        kind=kind,
        severity=severity,
        baseline_accuracy=float(baseline_correct.mean()),
        candidate_accuracy=float(candidate_correct.mean()),
        accuracy_difference=float(paired_difference.mean()),
        difference_lower=float(lower),
        difference_upper=float(upper),
        baseline_only_correct=int(np.sum((baseline_correct == 1) & (candidate_correct == 0))),
        candidate_only_correct=int(np.sum((baseline_correct == 0) & (candidate_correct == 1))),
        evidence=evidence,
    )


def compare_models(
    baseline_name: str,
    baseline: BenchmarkResult,
    candidate_name: str,
    candidate: BenchmarkResult,
    *,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 1000,
    seed: int = 0,
) -> ModelComparison:
    """Compare two benchmark runs using paired per-example uncertainty."""
    if not baseline_name.strip() or not candidate_name.strip():
        raise ValueError("model names must not be empty")
    if baseline_name == candidate_name:
        raise ValueError("model names must be distinct")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if (
        not isinstance(bootstrap_resamples, int)
        or isinstance(bootstrap_resamples, bool)
        or bootstrap_resamples < 2
    ):
        raise ValueError("bootstrap_resamples must be an integer of at least two")
    if baseline.example_count != candidate.example_count:
        raise ValueError("model results must contain the same number of examples")

    baseline_conditions = {
        (condition.kind, condition.severity): condition
        for condition in baseline.conditions
    }
    candidate_conditions = {
        (condition.kind, condition.severity): condition
        for condition in candidate.conditions
    }
    if baseline_conditions.keys() != candidate_conditions.keys():
        raise ValueError("model results must contain identical corruption conditions")
    if not baseline_conditions:
        raise ValueError("at least one corruption condition is required")

    first_key = sorted(baseline_conditions)[0]
    first_baseline = baseline_conditions[first_key]
    first_candidate = candidate_conditions[first_key]
    points = [
        _comparison_point(
            "clean",
            0,
            first_baseline.records,
            first_candidate.records,
            shifted=False,
            confidence_level=confidence_level,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
    ]
    for point_index, key in enumerate(sorted(baseline_conditions), start=1):
        baseline_condition = baseline_conditions[key]
        candidate_condition = candidate_conditions[key]
        if baseline_condition.seed != candidate_condition.seed:
            raise ValueError("paired corruption conditions must use identical seeds")
        points.append(
            _comparison_point(
                key[0],
                key[1],
                baseline_condition.records,
                candidate_condition.records,
                shifted=True,
                confidence_level=confidence_level,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed + point_index,
            )
        )

    baseline_supported = sum(point.evidence == "baseline" for point in points)
    candidate_supported = sum(point.evidence == "candidate" for point in points)
    inconclusive = sum(point.evidence == "inconclusive" for point in points)
    return ModelComparison(
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        baseline_supported_conditions=baseline_supported,
        candidate_supported_conditions=candidate_supported,
        inconclusive_conditions=inconclusive,
        points=tuple(points),
    )
