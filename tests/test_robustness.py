import pytest

from vision_shift_bench import (
    CorruptionSpec,
    NearestCentroidClassifier,
    analyse_severity_curves,
    evaluate_corruptions,
    make_pattern_dataset,
)


@pytest.fixture(scope="module")
def benchmark():
    training_images, training_labels = make_pattern_dataset(
        samples_per_class=15, size=16, seed=3
    )
    evaluation_images, evaluation_labels = make_pattern_dataset(
        samples_per_class=20, size=16, seed=5
    )
    model = NearestCentroidClassifier.fit(training_images, training_labels)
    specs = [
        CorruptionSpec(kind, severity, seed=17)
        for kind in ("brightness", "gaussian_blur")
        for severity in range(1, 6)
    ]
    return evaluate_corruptions(model, evaluation_images, evaluation_labels, specs)


def test_complete_curves_are_normalised_and_ordered(benchmark) -> None:
    analysis = analyse_severity_curves(
        benchmark, bootstrap_resamples=100, seed=23, accuracy_floor=0.8
    )

    assert [curve.kind for curve in analysis.curves] == ["brightness", "gaussian_blur"]
    for curve in analysis.curves:
        assert [point.severity for point in curve.points] == [1, 2, 3, 4, 5]
        assert curve.normalized_accuracy_retention_auc + curve.normalized_degradation_auc == pytest.approx(1.0)
        for point in curve.points:
            assert point.accuracy_drop_lower <= point.accuracy_drop_upper


def test_bootstrap_intervals_are_deterministic(benchmark) -> None:
    first = analyse_severity_curves(benchmark, bootstrap_resamples=100, seed=29)
    second = analyse_severity_curves(benchmark, bootstrap_resamples=100, seed=29)

    assert first == second


def test_failure_point_matches_first_threshold_crossing(benchmark) -> None:
    analysis = analyse_severity_curves(
        benchmark, bootstrap_resamples=50, seed=31, accuracy_floor=0.8
    )
    blur = next(curve for curve in analysis.curves if curve.kind == "gaussian_blur")
    expected = next(
        (point.severity for point in blur.points if point.accuracy < 0.8), None
    )

    assert blur.failure_severity == expected


def test_incomplete_severity_curve_is_rejected(benchmark) -> None:
    incomplete = type(benchmark)(
        example_count=benchmark.example_count,
        clean_accuracy=benchmark.clean_accuracy,
        clean_mean_confidence=benchmark.clean_mean_confidence,
        conditions=benchmark.conditions[:-1],
    )
    with pytest.raises(ValueError, match="each severity"):
        analyse_severity_curves(incomplete, bootstrap_resamples=20)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("accuracy_floor", 0.0, "accuracy_floor"),
        ("confidence_level", 1.0, "confidence_level"),
        ("bootstrap_resamples", 1, "bootstrap_resamples"),
    ],
)
def test_invalid_analysis_settings_are_rejected(
    benchmark, argument: str, value: float, message: str
) -> None:
    kwargs = {argument: value}
    with pytest.raises(ValueError, match=message):
        analyse_severity_curves(benchmark, **kwargs)
