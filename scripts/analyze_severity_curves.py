"""Summarise full corruption-severity curves with paired uncertainty."""

from __future__ import annotations

import json

from vision_shift_bench import (
    CorruptionSpec,
    NearestCentroidClassifier,
    analyse_severity_curves,
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
        for severity in range(1, 6)
    ]
    benchmark = evaluate_corruptions(classifier, evaluation_images, evaluation_labels, specs)
    analysis = analyse_severity_curves(
        benchmark,
        accuracy_floor=0.85,
        confidence_level=0.95,
        bootstrap_resamples=1000,
        seed=211,
    )
    print(json.dumps(analysis.to_dict(), indent=2))


if __name__ == "__main__":
    main()
