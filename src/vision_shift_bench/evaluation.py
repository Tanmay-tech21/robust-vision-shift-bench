"""Paired clean-versus-corrupted classifier evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .adapters import LogitPredictor, _as_image_batch
from .corruptions import CorruptionSpec, apply_corruption


@dataclass(frozen=True)
class PredictionRecord:
    """Matched clean and corrupted outcomes for one example."""

    index: int
    label: int
    clean_prediction: int
    corrupted_prediction: int
    clean_confidence: float
    corrupted_confidence: float
    clean_true_probability: float
    corrupted_true_probability: float


@dataclass(frozen=True)
class ConditionResult:
    """Summary for one corruption condition, including paired transitions."""

    kind: str
    severity: int
    seed: int
    accuracy: float
    accuracy_drop: float
    prediction_flip_rate: float
    failure_rate: float
    recovery_rate: float
    mean_confidence: float
    records: tuple[PredictionRecord, ...]


@dataclass(frozen=True)
class BenchmarkResult:
    """Clean baseline and all evaluated corruption conditions."""

    example_count: int
    clean_accuracy: float
    clean_mean_confidence: float
    conditions: tuple[ConditionResult, ...]

    def to_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        """Return a JSON-compatible representation of the result."""
        result = asdict(self)
        if not include_records:
            for condition in result["conditions"]:
                condition.pop("records")
        return result


def _as_labels(labels: ArrayLike, example_count: int) -> NDArray[np.int64]:
    targets = np.asarray(labels)
    if targets.ndim != 1 or targets.shape[0] != example_count:
        raise ValueError("labels must contain one value per image")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("labels must be integers")
    if np.any(targets < 0):
        raise ValueError("labels must be non-negative")
    return targets.astype(np.int64, copy=False)


def _predict(
    classifier: LogitPredictor,
    images: NDArray[np.float64],
    batch_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    rows: list[NDArray[np.float64]] = []
    for start in range(0, images.shape[0], batch_size):
        stop = min(start + batch_size, images.shape[0])
        logits = np.asarray(classifier.predict_logits(images[start:stop]), dtype=np.float64)
        if logits.ndim != 2 or logits.shape[0] != stop - start or logits.shape[1] < 2:
            raise ValueError("classifier must return logits with shape (examples, classes >= 2)")
        if not np.all(np.isfinite(logits)):
            raise ValueError("classifier logits must be finite")
        rows.append(logits)

    all_logits = np.concatenate(rows, axis=0)
    shifted = all_logits - all_logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = all_logits.argmax(axis=1).astype(np.int64)
    confidence = probabilities[np.arange(probabilities.shape[0]), predictions]
    return predictions, confidence, probabilities


def _corrupt_batch(
    images: NDArray[np.float64], spec: CorruptionSpec
) -> NDArray[np.float64]:
    # A deterministic per-example seed avoids applying one identical noise
    # field to every image while retaining reproducibility across reruns.
    return np.stack(
        [
            apply_corruption(
                image,
                CorruptionSpec(spec.kind, spec.severity, seed=spec.seed + index),
            )
            for index, image in enumerate(images)
        ]
    )


def evaluate_corruptions(
    classifier: LogitPredictor,
    images: ArrayLike,
    labels: ArrayLike,
    specs: Sequence[CorruptionSpec],
    *,
    batch_size: int = 64,
) -> BenchmarkResult:
    """Evaluate paired examples and retain transitions for failure analysis."""
    batch = _as_image_batch(images)
    targets = _as_labels(labels, batch.shape[0])
    if not specs:
        raise ValueError("at least one corruption specification is required")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    clean_prediction, clean_confidence, clean_probabilities = _predict(
        classifier, batch, batch_size
    )
    class_count = clean_probabilities.shape[1]
    if targets.max(initial=0) >= class_count:
        raise ValueError("labels contain a class absent from classifier logits")
    clean_correct = clean_prediction == targets
    clean_accuracy = float(clean_correct.mean())

    conditions: list[ConditionResult] = []
    for spec in specs:
        corrupted = _corrupt_batch(batch, spec)
        prediction, confidence, shifted_probabilities = _predict(
            classifier, corrupted, batch_size
        )
        shifted_class_count = shifted_probabilities.shape[1]
        if shifted_class_count != class_count:
            raise ValueError("classifier must return the same class count for every batch")
        correct = prediction == targets
        records = tuple(
            PredictionRecord(
                index=index,
                label=int(targets[index]),
                clean_prediction=int(clean_prediction[index]),
                corrupted_prediction=int(prediction[index]),
                clean_confidence=float(clean_confidence[index]),
                corrupted_confidence=float(confidence[index]),
                clean_true_probability=float(clean_probabilities[index, targets[index]]),
                corrupted_true_probability=float(
                    shifted_probabilities[index, targets[index]]
                ),
            )
            for index in range(batch.shape[0])
        )
        accuracy = float(correct.mean())
        conditions.append(
            ConditionResult(
                kind=spec.kind,
                severity=spec.severity,
                seed=spec.seed,
                accuracy=accuracy,
                accuracy_drop=clean_accuracy - accuracy,
                prediction_flip_rate=float((prediction != clean_prediction).mean()),
                failure_rate=float((clean_correct & ~correct).mean()),
                recovery_rate=float((~clean_correct & correct).mean()),
                mean_confidence=float(confidence.mean()),
                records=records,
            )
        )

    return BenchmarkResult(
        example_count=batch.shape[0],
        clean_accuracy=clean_accuracy,
        clean_mean_confidence=float(clean_confidence.mean()),
        conditions=tuple(conditions),
    )
