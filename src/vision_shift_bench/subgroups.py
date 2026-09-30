"""Subgroup and worst-case selective-prediction diagnostics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Sequence

import numpy as np

from .evaluation import BenchmarkResult, ConditionResult, PredictionRecord


@dataclass(frozen=True)
class GroupSnapshot:
    """Performance and acceptance behaviour for one labelled group."""

    label: int
    support: int
    accuracy: float
    mean_confidence: float
    accepted_count: int
    coverage: float
    selective_risk: float | None


@dataclass(frozen=True)
class SubgroupSnapshot:
    """Group-level results and worst-case summaries for one condition."""

    groups: tuple[GroupSnapshot, ...]
    macro_accuracy: float
    worst_group_accuracy: float
    macro_selective_risk: float
    worst_group_selective_risk: float
    minimum_group_coverage: float
    coverage_gap: float


@dataclass(frozen=True)
class SubgroupPoint:
    """Subgroup snapshot attached to a corruption severity."""

    severity: int
    metrics: SubgroupSnapshot


@dataclass(frozen=True)
class SubgroupCurve:
    """Subgroup trajectory and first operational threshold violations."""

    kind: str
    first_coverage_gap_failure_severity: int | None
    first_worst_risk_failure_severity: int | None
    points: tuple[SubgroupPoint, ...]


@dataclass(frozen=True)
class SubgroupAnalysis:
    """Clean baseline and subgroup trajectories across corruptions."""

    clean: SubgroupSnapshot
    target_coverage: float
    coverage_gap_ceiling: float
    worst_risk_ceiling: float
    curves: tuple[SubgroupCurve, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def summarise_subgroups(
    records: Sequence[PredictionRecord],
    *,
    shifted: bool,
    target_coverage: float = 0.8,
) -> SubgroupSnapshot:
    """Audit group outcomes under one global confidence-based selector."""
    if not records:
        raise ValueError("prediction records must not be empty")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must lie in (0, 1]")
    indices = np.fromiter((record.index for record in records), dtype=np.int64)
    if len(np.unique(indices)) != len(records):
        raise ValueError("prediction record indices must be unique")

    labels = np.fromiter((record.label for record in records), dtype=np.int64)
    if shifted:
        predictions = np.fromiter(
            (record.corrupted_prediction for record in records), dtype=np.int64
        )
        confidences = np.fromiter(
            (record.corrupted_confidence for record in records), dtype=np.float64
        )
    else:
        predictions = np.fromiter(
            (record.clean_prediction for record in records), dtype=np.int64
        )
        confidences = np.fromiter(
            (record.clean_confidence for record in records), dtype=np.float64
        )
    correct = predictions == labels

    accepted_count = ceil(target_coverage * len(records))
    order = np.lexsort((indices, -confidences))
    accepted = np.zeros(len(records), dtype=bool)
    accepted[order[:accepted_count]] = True

    groups: list[GroupSnapshot] = []
    for label in sorted(np.unique(labels).tolist()):
        selected = labels == label
        group_accepted = selected & accepted
        support = int(selected.sum())
        accepted_in_group = int(group_accepted.sum())
        groups.append(
            GroupSnapshot(
                label=int(label),
                support=support,
                accuracy=float(correct[selected].mean()),
                mean_confidence=float(confidences[selected].mean()),
                accepted_count=accepted_in_group,
                coverage=accepted_in_group / support,
                selective_risk=(
                    float((~correct[group_accepted]).mean())
                    if accepted_in_group
                    else None
                ),
            )
        )

    group_risks = [group.selective_risk for group in groups if group.selective_risk is not None]
    coverages = [group.coverage for group in groups]
    accuracies = [group.accuracy for group in groups]
    return SubgroupSnapshot(
        groups=tuple(groups),
        macro_accuracy=float(np.mean(accuracies)),
        worst_group_accuracy=min(accuracies),
        macro_selective_risk=float(np.mean(group_risks)),
        worst_group_selective_risk=max(group_risks),
        minimum_group_coverage=min(coverages),
        coverage_gap=max(coverages) - min(coverages),
    )


def analyse_subgroups_under_shift(
    result: BenchmarkResult,
    *,
    target_coverage: float = 0.8,
    coverage_gap_ceiling: float = 0.2,
    worst_risk_ceiling: float = 0.15,
) -> SubgroupAnalysis:
    """Measure worst-group outcomes over complete corruption-severity curves."""
    if not 0.0 <= coverage_gap_ceiling <= 1.0:
        raise ValueError("coverage_gap_ceiling must lie in [0, 1]")
    if not 0.0 <= worst_risk_ceiling <= 1.0:
        raise ValueError("worst_risk_ceiling must lie in [0, 1]")
    if not result.conditions:
        raise ValueError("at least one corruption condition is required")

    clean = summarise_subgroups(
        result.conditions[0].records,
        shifted=False,
        target_coverage=target_coverage,
    )
    grouped: dict[str, list[ConditionResult]] = defaultdict(list)
    for condition in result.conditions:
        grouped[condition.kind].append(condition)

    curves: list[SubgroupCurve] = []
    for kind in sorted(grouped):
        conditions = sorted(grouped[kind], key=lambda condition: condition.severity)
        if [condition.severity for condition in conditions] != list(range(1, 6)):
            raise ValueError(f"{kind} must contain each severity from 1 to 5 exactly once")
        points = tuple(
            SubgroupPoint(
                severity=condition.severity,
                metrics=summarise_subgroups(
                    condition.records,
                    shifted=True,
                    target_coverage=target_coverage,
                ),
            )
            for condition in conditions
        )
        curves.append(
            SubgroupCurve(
                kind=kind,
                first_coverage_gap_failure_severity=next(
                    (
                        point.severity
                        for point in points
                        if point.metrics.coverage_gap > coverage_gap_ceiling
                    ),
                    None,
                ),
                first_worst_risk_failure_severity=next(
                    (
                        point.severity
                        for point in points
                        if point.metrics.worst_group_selective_risk > worst_risk_ceiling
                    ),
                    None,
                ),
                points=points,
            )
        )

    return SubgroupAnalysis(
        clean=clean,
        target_coverage=target_coverage,
        coverage_gap_ceiling=coverage_gap_ceiling,
        worst_risk_ceiling=worst_risk_ceiling,
        curves=tuple(curves),
    )
