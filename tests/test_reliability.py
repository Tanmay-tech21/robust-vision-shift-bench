import numpy as np
import pytest

from vision_shift_bench import (
    CorruptionSpec,
    NearestCentroidClassifier,
    PredictionRecord,
    analyse_reliability_under_shift,
    evaluate_corruptions,
    make_pattern_dataset,
    summarise_prediction_records,
)


def test_selective_metrics_follow_deterministic_confidence_order() -> None:
    records = (
        PredictionRecord(0, 0, 0, 0, 0.9, 0.9, 0.9, 0.9),
        PredictionRecord(1, 1, 0, 0, 0.8, 0.8, 0.2, 0.2),
        PredictionRecord(2, 1, 1, 1, 0.6, 0.6, 0.6, 0.6),
    )

    metrics = summarise_prediction_records(
        records, shifted=False, bin_count=5, target_coverage=2 / 3
    )

    assert metrics.accuracy == pytest.approx(2 / 3)
    assert metrics.risk_at_target_coverage == 0.5
    assert metrics.area_under_risk_coverage == pytest.approx((0.0 + 0.5 + 1 / 3) / 3)
    assert metrics.negative_log_likelihood == pytest.approx(
        -np.log([0.9, 0.2, 0.6]).mean()
    )


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


def test_reliability_analysis_covers_complete_curves(benchmark) -> None:
    analysis = analyse_reliability_under_shift(
        benchmark, target_coverage=0.8, ece_ceiling=0.1, selective_risk_ceiling=0.1
    )

    assert [curve.kind for curve in analysis.curves] == ["brightness", "gaussian_blur"]
    assert 0.0 <= analysis.clean.expected_calibration_error <= 1.0
    for curve in analysis.curves:
        assert [point.severity for point in curve.points] == [1, 2, 3, 4, 5]
        for point in curve.points:
            assert point.metrics.negative_log_likelihood >= 0.0
            assert 0.0 <= point.metrics.area_under_risk_coverage <= 1.0


def test_clean_summary_is_independent_of_condition_order(benchmark) -> None:
    reversed_result = type(benchmark)(
        example_count=benchmark.example_count,
        clean_accuracy=benchmark.clean_accuracy,
        clean_mean_confidence=benchmark.clean_mean_confidence,
        conditions=tuple(reversed(benchmark.conditions)),
    )

    forward = analyse_reliability_under_shift(benchmark)
    reversed_analysis = analyse_reliability_under_shift(reversed_result)

    assert forward.clean == reversed_analysis.clean


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bin_count": 1}, "bin_count"),
        ({"target_coverage": 0.0}, "target_coverage"),
        ({"ece_ceiling": 1.1}, "ece_ceiling"),
        ({"selective_risk_ceiling": -0.1}, "selective_risk_ceiling"),
    ],
)
def test_invalid_reliability_settings_are_rejected(benchmark, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        analyse_reliability_under_shift(benchmark, **kwargs)
