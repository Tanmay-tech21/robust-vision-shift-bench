"""Run a deterministic clean-versus-corrupted classifier evaluation."""

from __future__ import annotations

import json

from vision_shift_bench import (
    CorruptionSpec,
    NearestCentroidClassifier,
    evaluate_corruptions,
    make_pattern_dataset,
)


def main() -> None:
    training_images, training_labels = make_pattern_dataset(
        samples_per_class=50, size=24, seed=31
    )
    evaluation_images, evaluation_labels = make_pattern_dataset(
        samples_per_class=80, size=24, seed=73
    )
    classifier = NearestCentroidClassifier.fit(training_images, training_labels)
    specs = [
        CorruptionSpec(kind, severity, seed=101)
        for kind in ("brightness", "contrast", "gaussian_noise", "gaussian_blur")
        for severity in (1, 3, 5)
    ]
    result = evaluate_corruptions(classifier, evaluation_images, evaluation_labels, specs)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
