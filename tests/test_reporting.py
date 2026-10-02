import json

from vision_shift_bench import (
    build_benchmark_snapshot,
    render_benchmark_card,
    write_benchmark_artifacts,
)


def test_benchmark_snapshot_is_deterministic_and_has_claim_boundaries():
    first = build_benchmark_snapshot()
    second = build_benchmark_snapshot()

    assert first == second
    assert first["status"] == "controlled synthetic pipeline validation"
    assert first["headline_results"]["comparison_evidence_counts"] == {
        "baseline": 0,
        "candidate": 1,
        "inconclusive": 20,
    }
    assert len(first["claim_boundaries"]) >= 3
    assert len(first["limitations"]) >= 3


def test_markdown_is_rendered_from_snapshot():
    snapshot = build_benchmark_snapshot()
    card = render_benchmark_card(snapshot)

    assert card.startswith("# Robust Vision Shift Benchmark Card")
    assert "## First threshold failures for the baseline" in card
    assert "| gaussian_blur | 4 | 2 | 4 | 3 | 4 |" in card
    assert "does not establish equivalence" in card
    assert "## Limitations" in card


def test_artifacts_round_trip_to_the_same_snapshot(tmp_path):
    json_path, markdown_path = write_benchmark_artifacts(tmp_path)
    snapshot = json.loads(json_path.read_text(encoding="utf-8"))

    assert markdown_path.read_text(encoding="utf-8") == render_benchmark_card(snapshot)
    assert snapshot["protocol"]["evaluation_examples"] == 240
    assert snapshot["headline_results"]["blur_severity_5"]["evidence"] == "candidate"
