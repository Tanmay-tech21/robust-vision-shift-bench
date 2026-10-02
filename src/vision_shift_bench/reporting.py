"""Reproducible experiment snapshots and benchmark-card rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapters import BlockMeanCentroidClassifier, NearestCentroidClassifier
from .comparison import compare_models
from .corruptions import CorruptionSpec
from .evaluation import evaluate_corruptions
from .reliability import analyse_reliability_under_shift
from .robustness import analyse_severity_curves
from .subgroups import analyse_subgroups_under_shift
from .synthetic import make_pattern_dataset


CORRUPTION_KINDS = (
    "brightness",
    "contrast",
    "gaussian_noise",
    "gaussian_blur",
)


def build_benchmark_snapshot() -> dict[str, Any]:
    """Run the fixed reference protocol and return one JSON-compatible snapshot."""
    training_images, training_labels = make_pattern_dataset(
        samples_per_class=50, size=24, seed=17
    )
    evaluation_images, evaluation_labels = make_pattern_dataset(
        samples_per_class=80, size=24, seed=23
    )
    specs = [
        CorruptionSpec(kind, severity, seed=31)
        for kind in CORRUPTION_KINDS
        for severity in range(1, 6)
    ]
    baseline = NearestCentroidClassifier.fit(training_images, training_labels)
    candidate = BlockMeanCentroidClassifier.fit(
        training_images, training_labels, block_size=2
    )
    baseline_result = evaluate_corruptions(
        baseline, evaluation_images, evaluation_labels, specs, batch_size=37
    )
    candidate_result = evaluate_corruptions(
        candidate, evaluation_images, evaluation_labels, specs, batch_size=37
    )

    severity = analyse_severity_curves(
        baseline_result,
        accuracy_floor=0.85,
        bootstrap_resamples=1000,
        seed=43,
    )
    reliability = analyse_reliability_under_shift(
        baseline_result,
        bin_count=15,
        target_coverage=0.8,
        ece_ceiling=0.1,
        selective_risk_ceiling=0.1,
    )
    subgroups = analyse_subgroups_under_shift(
        baseline_result,
        target_coverage=0.8,
        coverage_gap_ceiling=0.2,
        worst_risk_ceiling=0.15,
    )
    comparison = compare_models(
        "pixel-centroid",
        baseline_result,
        "2x2-mean-centroid",
        candidate_result,
        bootstrap_resamples=1000,
        seed=53,
    )
    blur_five = next(
        point
        for point in comparison.points
        if point.kind == "gaussian_blur" and point.severity == 5
    )

    return {
        "schema_version": 1,
        "title": "Robust Vision Shift Benchmark Card",
        "status": "controlled synthetic pipeline validation",
        "protocol": {
            "training_examples": int(training_labels.size),
            "evaluation_examples": int(evaluation_labels.size),
            "class_count": int(len(set(training_labels.tolist()))),
            "image_shape": list(training_images.shape[1:]),
            "training_seed": 17,
            "evaluation_seed": 23,
            "corruption_seed": 31,
            "corruptions": list(CORRUPTION_KINDS),
            "severities": [1, 2, 3, 4, 5],
            "batch_size": 37,
            "confidence_level": 0.95,
            "bootstrap_resamples": 1000,
        },
        "models": {
            "baseline": {
                "name": "pixel-centroid",
                "description": "Nearest centroid over flattened RGB pixels.",
            },
            "candidate": {
                "name": "2x2-mean-centroid",
                "description": "Nearest centroid after non-overlapping 2x2 mean pooling.",
            },
        },
        "thresholds": {
            "accuracy_floor": 0.85,
            "ece_ceiling": 0.1,
            "target_coverage": 0.8,
            "selective_risk_ceiling": 0.1,
            "class_coverage_gap_ceiling": 0.2,
            "worst_class_risk_ceiling": 0.15,
        },
        "headline_results": {
            "baseline_clean_accuracy": baseline_result.clean_accuracy,
            "candidate_clean_accuracy": candidate_result.clean_accuracy,
            "blur_severity_5": {
                "baseline_accuracy": blur_five.baseline_accuracy,
                "candidate_accuracy": blur_five.candidate_accuracy,
                "accuracy_difference": blur_five.accuracy_difference,
                "difference_lower": blur_five.difference_lower,
                "difference_upper": blur_five.difference_upper,
                "evidence": blur_five.evidence,
            },
            "comparison_evidence_counts": {
                "baseline": comparison.baseline_supported_conditions,
                "candidate": comparison.candidate_supported_conditions,
                "inconclusive": comparison.inconclusive_conditions,
            },
        },
        "baseline_benchmark": baseline_result.to_dict(include_records=False),
        "severity_analysis": severity.to_dict(),
        "reliability_analysis": reliability.to_dict(),
        "subgroup_analysis": subgroups.to_dict(),
        "model_comparison": comparison.to_dict(),
        "claim_boundaries": [
            "The results validate the benchmark implementation on a controlled synthetic task.",
            "Only Gaussian blur at severity 5 supported a candidate advantage in this protocol.",
            "An interval containing zero is inconclusive; it does not establish equivalence.",
            "The evidence does not establish general robustness or deep-model performance.",
        ],
        "limitations": [
            "The images are synthetic textures rather than naturally captured data.",
            "The reference classifiers are transparent diagnostics, not trained neural networks.",
            "Synthetic corruptions do not reproduce every real deployment shift.",
            "Class labels are diagnostic groups, not protected demographic attributes.",
            "Thresholds are illustrative and require application-specific justification.",
        ],
        "next_validation": [
            "Register held-out logits or a framework model through the predictor interface.",
            "Add naturally captured shifts alongside the synthetic corruption matrix.",
            "Choose operational thresholds from deployment costs before interpreting failures.",
        ],
    }


def _severity_text(value: int | None) -> str:
    return "None" if value is None else str(value)


def render_benchmark_card(snapshot: dict[str, Any]) -> str:
    """Render a compact Markdown card entirely from one experiment snapshot."""
    protocol = snapshot["protocol"]
    thresholds = snapshot["thresholds"]
    headline = snapshot["headline_results"]
    blur = headline["blur_severity_5"]
    evidence = headline["comparison_evidence_counts"]
    severity = {curve["kind"]: curve for curve in snapshot["severity_analysis"]["curves"]}
    reliability = {
        curve["kind"]: curve for curve in snapshot["reliability_analysis"]["curves"]
    }
    subgroups = {
        curve["kind"]: curve for curve in snapshot["subgroup_analysis"]["curves"]
    }

    lines = [
        f"# {snapshot['title']}",
        "",
        f"**Status:** {snapshot['status']}",
        "",
        "This card is generated from the same deterministic snapshot as the machine-readable JSON report.",
        "",
        "## Protocol",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Training examples | {protocol['training_examples']} |",
        f"| Evaluation examples | {protocol['evaluation_examples']} |",
        f"| Image shape | `{' x '.join(map(str, protocol['image_shape']))}` |",
        f"| Corruptions | {', '.join(protocol['corruptions'])} |",
        f"| Severities | {', '.join(map(str, protocol['severities']))} |",
        f"| Seeds | train `{protocol['training_seed']}`, evaluation `{protocol['evaluation_seed']}`, corruption `{protocol['corruption_seed']}` |",
        f"| Paired uncertainty | {protocol['bootstrap_resamples']} resamples, {protocol['confidence_level']:.0%} interval |",
        "",
        "## Reference models",
        "",
        f"- **Baseline, `{snapshot['models']['baseline']['name']}`:** {snapshot['models']['baseline']['description']}",
        f"- **Candidate, `{snapshot['models']['candidate']['name']}`:** {snapshot['models']['candidate']['description']}",
        "",
        "## Headline comparison",
        "",
        "| Condition | Baseline accuracy | Candidate accuracy | Candidate minus baseline | Evidence |",
        "|---|---:|---:|---:|---|",
        f"| Clean | {headline['baseline_clean_accuracy']:.4f} | {headline['candidate_clean_accuracy']:.4f} | {headline['candidate_clean_accuracy'] - headline['baseline_clean_accuracy']:.4f} | inconclusive |",
        f"| Gaussian blur, severity 5 | {blur['baseline_accuracy']:.4f} | {blur['candidate_accuracy']:.4f} | {blur['accuracy_difference']:.4f} `[{blur['difference_lower']:.4f}, {blur['difference_upper']:.4f}]` | {blur['evidence']} |",
        "",
        f"Across 21 paired conditions, evidence supported the candidate in **{evidence['candidate']}**, supported the baseline in **{evidence['baseline']}**, and was inconclusive in **{evidence['inconclusive']}**.",
        "",
        "## First threshold failures for the baseline",
        "",
        f"Thresholds: accuracy `< {thresholds['accuracy_floor']:.2f}`, ECE `> {thresholds['ece_ceiling']:.2f}`, selective risk at {thresholds['target_coverage']:.0%} coverage `> {thresholds['selective_risk_ceiling']:.2f}`, class coverage gap `> {thresholds['class_coverage_gap_ceiling']:.2f}`, worst-class risk `> {thresholds['worst_class_risk_ceiling']:.2f}`.",
        "",
        "| Corruption | Accuracy | ECE | Selective risk | Coverage gap | Worst-class risk |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind in protocol["corruptions"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    kind,
                    _severity_text(severity[kind]["failure_severity"]),
                    _severity_text(reliability[kind]["first_ece_failure_severity"]),
                    _severity_text(
                        reliability[kind]["first_selective_risk_failure_severity"]
                    ),
                    _severity_text(
                        subgroups[kind]["first_coverage_gap_failure_severity"]
                    ),
                    _severity_text(subgroups[kind]["first_worst_risk_failure_severity"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Claim boundaries", ""])
    lines.extend(f"- {item}" for item in snapshot["claim_boundaries"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in snapshot["limitations"])
    lines.extend(["", "## Next validation", ""])
    lines.extend(f"- {item}" for item in snapshot["next_validation"])
    return "\n".join(lines) + "\n"


def write_benchmark_artifacts(output_directory: str | Path) -> tuple[Path, Path]:
    """Generate matching JSON and Markdown artifacts in ``output_directory``."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = build_benchmark_snapshot()
    json_path = output / "benchmark_summary.json"
    markdown_path = output / "BENCHMARK_CARD.md"
    json_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_benchmark_card(snapshot), encoding="utf-8")
    return json_path, markdown_path
