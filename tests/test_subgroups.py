import pytest

from vision_shift_bench import (
    CorruptionSpec,
    NearestCentroidClassifier,
    PredictionRecord,
    analyse_subgroups_under_shift,
    evaluate_corruptions,
    make_pattern_dataset,
    summarise_subgroups,
)


def test_global_selector_exposes_unequal_group_coverage() -> None:
    records = (
        PredictionRecord(0, 0, 0, 0, 0.95, 0.95, 0.95, 0.95),
        PredictionRecord(1, 0, 1, 1, 0.90, 0.90, 0.10, 0.10),
        PredictionRecord(2, 0, 0, 0, 0.85, 0.85, 0.85, 0.85),
        PredictionRecord(3, 1, 1, 1, 0.40, 0.40, 0.40, 0.40),
        PredictionRecord(4, 1, 1, 1, 0.30, 0.30, 0.30, 0.30),
        PredictionRecord(5, 1, 0, 0, 0.20, 0.20, 0.80, 0.80),
    )

    summary = summarise_subgroups(records, shifted=True, target_coverage=0.5)

    assert [group.coverage for group in summary.groups] == [1.0, 0.0]
    assert summary.coverage_gap == 1.0
    assert summary.minimum_group_coverage == 0.0
    assert summary.worst_group_accuracy == pytest.approx(2 / 3)
    assert summary.worst_group_selective_risk == pytest.approx(1 / 3)
    assert summary.groups[1].selective_risk is None


@pytest.fixture(scope="module")
def benchmark():
    training_images, training_labels = make_pattern_dataset(
        samples_per_class=12, size=16, seed=41
    )
    evaluation_images, evaluation_labels = make_pattern_dataset(
        samples_per_class=18, size=16, seed=43
    )
    model = NearestCentroidClassifier.fit(training_images, training_labels)
    specs = [
        CorruptionSpec(kind, severity, seed=47)
        for kind in ("brightness", "gaussian_blur")
        for severity in range(1, 6)
    ]
    return evaluate_corruptions(model, evaluation_images, evaluation_labels, specs)


def test_subgroup_analysis_covers_complete_curves(benchmark) -> None:
    analysis = analyse_subgroups_under_shift(benchmark)

    assert [curve.kind for curve in analysis.curves] == ["brightness", "gaussian_blur"]
    assert [group.label for group in analysis.clean.groups] == [0, 1, 2]
    for curve in analysis.curves:
        assert [point.severity for point in curve.points] == [1, 2, 3, 4, 5]
        for point in curve.points:
            assert 0.0 <= point.metrics.worst_group_accuracy <= 1.0
            assert 0.0 <= point.metrics.coverage_gap <= 1.0


def test_subgroup_analysis_is_independent_of_condition_order(benchmark) -> None:
    reversed_result = type(benchmark)(
        example_count=benchmark.example_count,
        clean_accuracy=benchmark.clean_accuracy,
        clean_mean_confidence=benchmark.clean_mean_confidence,
        conditions=tuple(reversed(benchmark.conditions)),
    )
    assert analyse_subgroups_under_shift(benchmark) == analyse_subgroups_under_shift(
        reversed_result
    )


def test_confidence_ties_use_example_index() -> None:
    records = (
        PredictionRecord(4, 1, 1, 1, 0.8, 0.8, 0.8, 0.8),
        PredictionRecord(1, 0, 0, 0, 0.8, 0.8, 0.8, 0.8),
    )
    summary = summarise_subgroups(records, shifted=False, target_coverage=0.5)
    assert [group.coverage for group in summary.groups] == [1.0, 0.0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_coverage": 0.0}, "target_coverage"),
        ({"coverage_gap_ceiling": 1.1}, "coverage_gap_ceiling"),
        ({"worst_risk_ceiling": -0.1}, "worst_risk_ceiling"),
    ],
)
def test_invalid_subgroup_settings_are_rejected(benchmark, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        analyse_subgroups_under_shift(benchmark, **kwargs)


def test_duplicate_record_indices_are_rejected() -> None:
    records = (
        PredictionRecord(0, 0, 0, 0, 0.8, 0.8, 0.8, 0.8),
        PredictionRecord(0, 1, 1, 1, 0.7, 0.7, 0.7, 0.7),
    )
    with pytest.raises(ValueError, match="unique"):
        summarise_subgroups(records, shifted=False)
