# Robust Vision Shift Benchmark Card

**Status:** controlled synthetic pipeline validation

This card is generated from the same deterministic snapshot as the machine-readable JSON report.

## Protocol

| Item | Value |
|---|---|
| Training examples | 150 |
| Evaluation examples | 240 |
| Image shape | `24 x 24 x 3` |
| Corruptions | brightness, contrast, gaussian_noise, gaussian_blur |
| Severities | 1, 2, 3, 4, 5 |
| Seeds | train `17`, evaluation `23`, corruption `31` |
| Paired uncertainty | 1000 resamples, 95% interval |

## Reference models

- **Baseline, `pixel-centroid`:** Nearest centroid over flattened RGB pixels.
- **Candidate, `2x2-mean-centroid`:** Nearest centroid after non-overlapping 2x2 mean pooling.

## Headline comparison

| Condition | Baseline accuracy | Candidate accuracy | Candidate minus baseline | Evidence |
|---|---:|---:|---:|---|
| Clean | 0.9708 | 0.9708 | 0.0000 | inconclusive |
| Gaussian blur, severity 5 | 0.3333 | 0.5375 | 0.2042 `[0.1542, 0.2542]` | candidate |

Across 21 paired conditions, evidence supported the candidate in **1**, supported the baseline in **0**, and was inconclusive in **20**.

## First threshold failures for the baseline

Thresholds: accuracy `< 0.85`, ECE `> 0.10`, selective risk at 80% coverage `> 0.10`, class coverage gap `> 0.20`, worst-class risk `> 0.15`.

| Corruption | Accuracy | ECE | Selective risk | Coverage gap | Worst-class risk |
|---|---:|---:|---:|---:|---:|
| brightness | None | 2 | None | 5 | None |
| contrast | None | 2 | None | 4 | None |
| gaussian_noise | None | None | None | None | None |
| gaussian_blur | 4 | 2 | 4 | 3 | 4 |

## Claim boundaries

- The results validate the benchmark implementation on a controlled synthetic task.
- Only Gaussian blur at severity 5 supported a candidate advantage in this protocol.
- An interval containing zero is inconclusive; it does not establish equivalence.
- The evidence does not establish general robustness or deep-model performance.

## Limitations

- The images are synthetic textures rather than naturally captured data.
- The reference classifiers are transparent diagnostics, not trained neural networks.
- Synthetic corruptions do not reproduce every real deployment shift.
- Class labels are diagnostic groups, not protected demographic attributes.
- Thresholds are illustrative and require application-specific justification.

## Next validation

- Register held-out logits or a framework model through the predictor interface.
- Add naturally captured shifts alongside the synthetic corruption matrix.
- Choose operational thresholds from deployment costs before interpreting failures.
