# Robust Vision Shift Bench

A seven-day laboratory for measuring how image classifiers behave when the
deployment distribution differs from clean evaluation data. The project begins
with deterministic corruption operators and will build towards comparative
robustness reports across corruption types, severities, and subgroups.

## Day 1: define the corruption protocol

The initial measurement layer provides four controlled image transformations:

- brightness reduction;
- contrast reduction;
- additive Gaussian noise; and
- Gaussian blur.

Every corruption uses a discrete severity from 1 to 5. Parameter mappings are
declared in code, stochastic transformations accept explicit seeds, inputs are
never mutated, and outputs preserve shape and the normalised `[0, 1]` range.

A structured synthetic image containing gradients, edges, texture, and a disc
supports a quick validation with RMSE and PSNR. These image-space measurements
verify that the corruption machinery behaves as intended; they do not measure
classifier robustness.

## Day 2: measure paired predictive degradation

The benchmark now accepts any classifier implementing `predict_logits(images)`.
It evaluates a clean batch once, applies each corruption independently to the
same examples, and reports:

- clean and corrupted accuracy;
- absolute accuracy degradation;
- prediction-flip, clean-to-failure, and recovery rates;
- mean predicted confidence; and
- per-example clean/corrupted predictions for diagnosis.

A deterministic nearest-centroid adapter and synthetic texture dataset exercise
the entire pipeline without implying pretrained-model performance. Framework
wrappers can adopt the same interface while keeping evaluation code unchanged.

The deterministic pipeline-validation fixture produced the following endpoints:

| Condition | Accuracy | Accuracy drop | Mean confidence |
|---|---:|---:|---:|
| Clean | 0.9625 | — | 0.9301 |
| Brightness, severity 5 | 0.9625 | 0.0000 | 0.5975 |
| Contrast, severity 5 | 0.9583 | 0.0042 | 0.5944 |
| Gaussian noise, severity 5 | 0.9625 | 0.0000 | 0.9180 |
| Gaussian blur, severity 5 | 0.5417 | 0.4208 | 0.4122 |

These figures validate the evaluation machinery on a synthetic nearest-centroid
task. They are not estimates of deep-model performance or natural-shift
robustness. Notably, brightness reduced confidence without changing accuracy,
whereas severe blur caused both prediction flips and clean-to-corrupted failures.

## Day 3: quantify severity curves and failure points

Every corruption is now evaluated at all five severities. The analysis layer:

- constructs ordered accuracy and degradation curves;
- calculates paired bootstrap intervals for accuracy degradation;
- integrates degradation over severity and normalises it by clean accuracy;
- reports the complementary normalised accuracy-retention area; and
- identifies the first severity below a declared deployment accuracy floor.

The normalised areas include severity zero as the clean baseline. This avoids
treating the first corrupted condition as though it were the unshifted model.
Confidence intervals resample matched clean/corrupted outcomes together so the
pairing information is not discarded.

With an accuracy floor of `0.85`, the deterministic validation run produced:

| Corruption | Normalised retention AUC | First failing severity |
|---|---:|---:|
| Brightness | 1.0000 | None |
| Contrast | 0.9996 | None |
| Gaussian noise | 1.0000 | None |
| Gaussian blur | 0.8688 | 4 |

At blur severity 4, accuracy fell from `0.9625` to `0.5417`. The paired 95%
bootstrap interval for the `0.4208` accuracy drop was `[0.3500, 0.4917]` using
1,000 deterministic resamples. These synthetic results validate the analysis;
they do not establish a robustness ranking for trained vision architectures.

## Day 4: audit calibration and selective risk under corruption

Per-example records now retain the probability assigned to the true class. This
supports severity-wise evaluation of:

- multiclass negative log-likelihood;
- top-label expected calibration error;
- area under the risk-coverage curve; and
- selective risk at a declared target coverage.

The analysis reports the first severity that violates an ECE ceiling and the
first that violates a selective-risk ceiling. Confidence ties use example index
as a deterministic secondary key, keeping AURC exactly reproducible.

Using 15 ECE bins, 80% target coverage, and `0.10` ceilings for ECE and
selective risk, the synthetic validation produced:

| Corruption | First ECE failure | First selective-risk failure |
|---|---:|---:|
| Brightness | 2 | None |
| Contrast | 2 | None |
| Gaussian noise | None | None |
| Gaussian blur | 2 | 4 |

The clean ECE was `0.0719`, NLL was `0.1291`, and risk at 80% coverage was
zero. Brightness severity 2 raised ECE to `0.1300` without changing accuracy,
showing a calibration failure before a classification failure. Blur severity 4
raised selective risk at 80% coverage to `0.3229`; by severity 5, AURC reached
`0.3935`. ECE itself fell from `0.3488` at blur severity 3 to `0.1822` at
severity 5 even as NLL worsened, another warning against relying on one metric.

## Day 5: expose subgroup and worst-case failures

The benchmark now audits a single global confidence-based abstention policy by
class. At every corruption severity it reports:

- support, accuracy, mean confidence, and realised coverage for each class;
- selective risk among the accepted examples in each class;
- macro and worst-class accuracy and selective risk; and
- the gap between maximum and minimum class coverage.

Selection remains global: the most confident examples are accepted regardless
of class. This deliberately reveals whether an apparently reasonable overall
coverage target is achieved by disproportionately rejecting one subgroup.
Confidence ties use example index as a deterministic secondary key.

With 80% target coverage, a `0.20` class-coverage-gap ceiling, and a `0.15`
worst-class-risk ceiling, the synthetic validation produced:

| Corruption | First coverage-gap failure | First worst-risk failure |
|---|---:|---:|
| Brightness | 5 | None |
| Contrast | 4 | None |
| Gaussian noise | None | None |
| Gaussian blur | 3 | 4 |

At blur severity 4, realised class coverage ranged from `0.40` to `1.00`.
The least-covered class had `0.00` accuracy and `1.00` selective risk among its
accepted predictions. This is a controlled diagnostic fixture, not evidence
about a trained model or a protected demographic group. It demonstrates that
aggregate target coverage cannot substitute for subgroup-level risk reporting.

## Day 6: compare models with paired evidence

Cross-model reports now require identically ordered examples, corruption
conditions, and seeds. Accuracy differences are bootstrapped from per-example
paired outcomes, retaining the information lost by independent intervals. Each
condition is labelled as supporting the baseline, supporting the candidate, or
remaining inconclusive according to whether its interval excludes zero.

The deterministic comparison used a pixel-level nearest-centroid reference and
an otherwise matched classifier applying `2x2` mean pooling:

| Condition | Pixel centroid | Mean-pooled centroid | Difference (95% interval) |
|---|---:|---:|---:|
| Clean | 0.9708 | 0.9708 | 0.0000 `[0.0000, 0.0000]` |
| Blur, severity 3 | 0.9417 | 0.9458 | 0.0042 `[0.0000, 0.0125]` |
| Blur, severity 4 | 0.5875 | 0.5875 | 0.0000 `[0.0000, 0.0000]` |
| Blur, severity 5 | 0.3333 | 0.5375 | 0.2042 `[0.1542, 0.2542]` |

Only severe blur at level 5 supported a difference; the other 20 clean or
corrupted conditions were inconclusive. Pooling therefore improved this one
controlled endpoint, not robustness in general. Both systems remain transparent
pipeline-validation references rather than trained deep-learning models.

## Day 7: publish one reproducible benchmark card

The weekly result is now generated as both a human-readable benchmark card and
a machine-readable JSON snapshot. Both artifacts come from one deterministic
evaluation run, preventing hand-copied headline numbers from drifting away from
the underlying analyses. The report records:

- dataset sizes, model definitions, seeds, corruptions, and thresholds;
- paired cross-model evidence with bootstrap intervals;
- first failures for accuracy, calibration, selective risk, and subgroups;
- claim boundaries and limitations; and
- concrete next-validation steps.

The generated card preserves the central finding: `2x2` mean pooling was
supported only for Gaussian blur at severity 5. The remaining 20 paired
conditions were inconclusive. This is evidence for a local effect on a
controlled fixture, not a general architecture ranking.

Regenerate `artifacts/BENCHMARK_CARD.md` and
`artifacts/benchmark_summary.json` with:

```bash
python scripts/build_benchmark_card.py
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/run_corruption_smoke.py
python scripts/evaluate_classifier.py
python scripts/analyze_severity_curves.py
python scripts/analyze_reliability_under_shift.py
python scripts/analyze_subgroups_under_shift.py
python scripts/compare_reference_models.py
python scripts/build_benchmark_card.py
pytest
```

## Evaluation principles

- Clean and corrupted variants must derive from the same underlying example.
- A corruption name and severity are meaningful only with their parameter map.
- Random seeds are part of the experiment specification.
- Image distortion is not a substitute for predictive-performance evaluation.
- Synthetic corruptions are diagnostic proxies, not complete models of natural
  distribution shift.

## Planned progression

1. Deterministic corruption registry and image-space validation (complete)
2. Classifier adapter and clean-versus-corrupted evaluation runner (complete)
3. Severity curves and corruption-normalised degradation metrics (complete)
4. Calibration and selective prediction under corruption (complete)
5. Subgroup and worst-case diagnostics (complete)
6. Cross-model comparison report (complete)
7. Reproducible benchmark card and portfolio integration (complete)
