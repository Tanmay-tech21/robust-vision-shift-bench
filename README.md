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

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/run_corruption_smoke.py
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
2. Classifier adapter and clean-versus-corrupted evaluation runner
3. Severity curves and corruption-normalised degradation metrics
4. Calibration and selective prediction under corruption
5. Subgroup and worst-case diagnostics
6. Cross-model comparison report
7. Reproducible benchmark card and portfolio integration
