"""Validate the corruption suite on a deterministic structured image."""

from __future__ import annotations

import json

from vision_shift_bench import (
    CorruptionSpec,
    apply_corruption,
    distortion_metrics,
    make_reference_image,
)


def main() -> None:
    reference = make_reference_image()
    results: dict[str, list[dict[str, float | int]]] = {}
    for kind in ("brightness", "contrast", "gaussian_noise", "gaussian_blur"):
        results[kind] = []
        for severity in range(1, 6):
            corrupted = apply_corruption(
                reference,
                CorruptionSpec(kind=kind, severity=severity, seed=41),
            )
            results[kind].append(
                {"severity": severity, **distortion_metrics(reference, corrupted)}
            )

    print(
        json.dumps(
            {
                "fixture": "synthetic corruption-pipeline validation",
                "claim_boundary": "No classifier performance is measured in this smoke test.",
                "image_shape": list(reference.shape),
                "corruptions": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
