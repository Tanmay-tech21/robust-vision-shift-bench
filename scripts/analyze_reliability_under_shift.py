"""Evaluate calibration and selective risk across corruption severity."""

from __future__ import annotations

import json

from vision_shift_bench import (
    CorruptionSpec,
    NearestCentroidClassifier,
    analyse_reliability_under_shift,
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
    analysis = analyse_reliability_under_shift(
        benchmark,
        bin_count=15,
        target_coverage=0.8,
        ece_ceiling=0.1,
        selective_risk_ceiling=0.1,
    )
    print(json.dumps(analysis.to_dict(), indent=2))


if __name__ == "__main__":
    main()
