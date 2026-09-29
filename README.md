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

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/run_corruption_smoke.py
python scripts/evaluate_classifier.py
python scripts/analyze_severity_curves.py
python scripts/analyze_reliability_under_shift.py
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
5. Subgroup and worst-case diagnostics
6. Cross-model comparison report
7. Reproducible benchmark card and portfolio integration
