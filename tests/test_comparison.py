from dataclasses import replace

import pytest

from vision_shift_bench import (
    BlockMeanCentroidClassifier,
    CorruptionSpec,
    NearestCentroidClassifier,
    compare_models,
    evaluate_corruptions,
    make_pattern_dataset,
)


@pytest.fixture(scope="module")
def paired_results():
    training_images, training_labels = make_pattern_dataset(
        samples_per_class=12, size=16, seed=61
    )
    evaluation_images, evaluation_labels = make_pattern_dataset(
        samples_per_class=18, size=16, seed=67
    )
    specs = [
        CorruptionSpec(kind, severity, seed=71)
        for kind in ("brightness", "gaussian_blur")
        for severity in range(1, 6)
    ]
    baseline_model = NearestCentroidClassifier.fit(training_images, training_labels)
    candidate_model = BlockMeanCentroidClassifier.fit(
        training_images, training_labels, block_size=2
    )
    return (
        evaluate_corruptions(
            baseline_model, evaluation_images, evaluation_labels, specs
        ),
        evaluate_corruptions(
            candidate_model, evaluation_images, evaluation_labels, specs
        ),
    )


def test_comparison_is_paired_complete_and_deterministic(paired_results) -> None:
    first = compare_models("pixel", paired_results[0], "pooled", paired_results[1])
    second = compare_models("pixel", paired_results[0], "pooled", paired_results[1])

    assert first == second
    assert first.points[0].kind == "clean"
    assert len(first.points) == 11
    assert (
        first.baseline_supported_conditions
        + first.candidate_supported_conditions
        + first.inconclusive_conditions
        == len(first.points)
    )
    for point in first.points:
        assert point.difference_lower <= point.accuracy_difference <= point.difference_upper


def test_block_mean_classifier_validates_pooling_geometry() -> None:
    images, labels = make_pattern_dataset(samples_per_class=3, size=16, seed=73)
    with pytest.raises(ValueError, match="block_size"):
        BlockMeanCentroidClassifier.fit(images, labels, block_size=0)
    with pytest.raises(ValueError, match="divisible"):
        BlockMeanCentroidClassifier.fit(images, labels, block_size=3)


def test_mismatched_corruption_seed_is_rejected(paired_results) -> None:
    baseline, candidate = paired_results
    changed_condition = replace(candidate.conditions[0], seed=999)
    changed = replace(
        candidate,
        conditions=(changed_condition, *candidate.conditions[1:]),
    )
    with pytest.raises(ValueError, match="identical seeds"):
        compare_models("pixel", baseline, "pooled", changed)


def test_mismatched_example_pairing_is_rejected(paired_results) -> None:
    baseline, candidate = paired_results
    condition = candidate.conditions[0]
    changed_record = replace(condition.records[0], index=999)
    changed_condition = replace(
        condition,
        records=(changed_record, *condition.records[1:]),
    )
    changed = replace(
        candidate,
        conditions=(changed_condition, *candidate.conditions[1:]),
    )
    with pytest.raises(ValueError, match="identically ordered"):
        compare_models("pixel", baseline, "pooled", changed)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"confidence_level": 1.0}, "confidence_level"),
        ({"bootstrap_resamples": 1}, "bootstrap_resamples"),
    ],
)
def test_invalid_comparison_settings_are_rejected(
    paired_results, kwargs, message
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_models("pixel", paired_results[0], "pooled", paired_results[1], **kwargs)
