import numpy as np
import pytest

from vision_shift_bench import (
    CorruptionSpec,
    NearestCentroidClassifier,
    evaluate_corruptions,
    make_pattern_dataset,
)


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray]:
    return make_pattern_dataset(samples_per_class=12, size=16, seed=7)


def test_centroid_adapter_returns_finite_logits(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    images, labels = dataset
    model = NearestCentroidClassifier.fit(images, labels)

    logits = model.predict_logits(images[:5])

    assert logits.shape == (5, 3)
    assert np.all(np.isfinite(logits))


def test_evaluation_is_paired_and_reproducible(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    images, labels = dataset
    model = NearestCentroidClassifier.fit(images, labels)
    specs = [CorruptionSpec("gaussian_noise", 3, seed=19)]

    first = evaluate_corruptions(model, images, labels, specs, batch_size=7)
    second = evaluate_corruptions(model, images, labels, specs, batch_size=11)

    assert first == second
    assert first.example_count == images.shape[0]
    assert len(first.conditions[0].records) == images.shape[0]
    assert [record.index for record in first.conditions[0].records] == list(range(images.shape[0]))


def test_transition_rates_account_for_accuracy_change() -> None:
    images = np.full((4, 4, 4, 1), 0.8)
    labels = np.array([1, 1, 0, 0])

    class ThresholdClassifier:
        def predict_logits(self, batch: np.ndarray) -> np.ndarray:
            score = batch.mean(axis=(1, 2, 3)) - 0.5
            return np.stack((-score, score), axis=1)

    result = evaluate_corruptions(
        ThresholdClassifier(),
        images,
        labels,
        [CorruptionSpec("brightness", 5)],
    )
    condition = result.conditions[0]

    assert result.clean_accuracy == 0.5
    assert condition.accuracy == 0.5
    assert condition.failure_rate == 0.5
    assert condition.recovery_rate == 0.5
    assert condition.accuracy_drop == pytest.approx(
        condition.failure_rate - condition.recovery_rate
    )


def test_summary_can_exclude_or_include_records(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    images, labels = dataset
    model = NearestCentroidClassifier.fit(images, labels)
    result = evaluate_corruptions(model, images, labels, [CorruptionSpec("contrast", 2)])

    assert "records" not in result.to_dict()["conditions"][0]
    assert len(result.to_dict(include_records=True)["conditions"][0]["records"]) == len(labels)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size_is_rejected(
    dataset: tuple[np.ndarray, np.ndarray], batch_size: object
) -> None:
    images, labels = dataset
    model = NearestCentroidClassifier.fit(images, labels)
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_corruptions(
            model,
            images,
            labels,
            [CorruptionSpec("brightness", 1)],
            batch_size=batch_size,  # type: ignore[arg-type]
        )


def test_malformed_classifier_output_is_rejected(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    images, labels = dataset

    class BrokenClassifier:
        def predict_logits(self, batch: np.ndarray) -> np.ndarray:
            return np.zeros(batch.shape[0])

    with pytest.raises(ValueError, match="classifier"):
        evaluate_corruptions(
            BrokenClassifier(), images, labels, [CorruptionSpec("brightness", 1)]
        )


def test_centroid_fit_rejects_nonconsecutive_labels(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    images, labels = dataset
    labels = np.where(labels == 2, 3, labels)
    with pytest.raises(ValueError, match="consecutive"):
        NearestCentroidClassifier.fit(images, labels)
