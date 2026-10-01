"""Compare full-resolution and pooled reference classifiers."""

from __future__ import annotations

import json

from vision_shift_bench import (
    BlockMeanCentroidClassifier,
    CorruptionSpec,
    NearestCentroidClassifier,
    compare_models,
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
    models = {
        "pixel-centroid": NearestCentroidClassifier.fit(
            training_images, training_labels
        ),
        "2x2-mean-centroid": BlockMeanCentroidClassifier.fit(
            training_images, training_labels, block_size=2
        ),
    }
    specs = [
        CorruptionSpec(kind, severity, seed=31)
        for kind in ("brightness", "contrast", "gaussian_noise", "gaussian_blur")
        for severity in range(1, 6)
    ]
    results = {
        name: evaluate_corruptions(
            model, evaluation_images, evaluation_labels, specs, batch_size=37
        )
        for name, model in models.items()
    }
    comparison = compare_models(
        "pixel-centroid",
        results["pixel-centroid"],
        "2x2-mean-centroid",
        results["2x2-mean-centroid"],
        bootstrap_resamples=1000,
        seed=53,
    )
    print(json.dumps(comparison.to_dict(), indent=2))


if __name__ == "__main__":
    main()
