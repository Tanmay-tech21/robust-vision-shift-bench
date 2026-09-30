"""Run the deterministic subgroup audit across corruption severities."""

from __future__ import annotations

import json

from vision_shift_bench import (
    CorruptionSpec,
    NearestCentroidClassifier,
    analyse_subgroups_under_shift,
    evaluate_corruptions,
    make_pattern_dataset,
)


def main() -> None:
    training_images, training_labels = make_pattern_dataset(
        samples_per_class=50, size=24, seed=17
    )
    evaluation_images, evaluation_labels = make_pattern_dataset(
        samples_per_class=80, size=24, seed=23
    )
    classifier = NearestCentroidClassifier.fit(training_images, training_labels)
    specs = [
        CorruptionSpec(kind, severity, seed=31)
        for kind in ("brightness", "contrast", "gaussian_noise", "gaussian_blur")
        for severity in range(1, 6)
    ]
    benchmark = evaluate_corruptions(
        classifier, evaluation_images, evaluation_labels, specs, batch_size=37
    )
    analysis = analyse_subgroups_under_shift(
        benchmark,
        target_coverage=0.8,
        coverage_gap_ceiling=0.2,
        worst_risk_ceiling=0.15,
    )
    print(json.dumps(analysis.to_dict(), indent=2))


if __name__ == "__main__":
    main()
