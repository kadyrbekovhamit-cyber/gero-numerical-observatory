# GERO Numerical Observatory

A reproducible CPU benchmark for mathematical contracts in ONNX graphs. It runs identical models in **ONNX Runtime** and the official **ONNX ReferenceEvaluator**, checks invariants, preserves artifacts and exports a filterable static dashboard.

[Interactive reports](https://www.gero.uz/stability/) · [Publication and limits](https://www.gero.uz/research/articles/onnx-numerical-observatory-reproduced-snapshot.html) · [Measured snapshot](benchmarks/2026-09-06/latest.json) · [Reverification](benchmarks/2026-09-06/reverification.json)

## Published snapshot, 6 September 2026

| Measure | Result |
|---|---:|
| Seeded synthetic graphs | 123 |
| ORT optimization settings | disabled / all |
| Evaluations | 246 |
| Passed all configured checks | 232 |
| Divergence observations | 14 |
| Distinct flagged graphs | 7 |
| Unit/property tests passed | 57 |
| Flagged archives replayed with the same observation identity | 14 / 14 |

Pinned environment: Python 3.12.13, ONNX 1.19.0, ONNX Runtime 1.22.1, NumPy 2.2.6, macOS 15.5 arm64, CPUExecutionProvider, one thread, opset 19 / IR 10. The CPU model string was unavailable. A complete rerun reproduced all 246 observation identities. Other environments produce new observations and may produce different outcomes.

These are **version-specific, reproduced observations**. This release does not establish seven new bugs, upstream confirmation, latest-version impact, model-level accuracy or regressions between versions. The reference evaluator also exhibits numerical limitations in these cases.

## Run

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pytest -q

# Explore the committed measured snapshot without rerunning it.
.venv/bin/python -m gero_stability.cli export benchmarks/2026-09-06/latest.json --site site
.venv/bin/python -m http.server 4187 --directory site

# Generate a fresh run.
.venv/bin/python -m gero_stability.cli run --output reports/local --random-cases 24
```

Open `http://localhost:4187`. `make verify` runs the tests, a fresh benchmark and export. `make preview` exports `reports/local` and serves the dashboard after a run.

## Contracts and coverage

- Batch partition, permutation, duplication and rank-change invariance for independent examples, not intentional reductions across the batch.
- Finite probabilities, strict bounds `[0,1]`, sums within tolerance, and exactly zero masked entries. The custom fully masked-row contract returns zero.
- Algebraically equivalent graph comparisons and analytic NumPy float64 diagnostics.
- Softmax, LogSoftmax, masked softmax, LayerNormalization, LpNormalization, per-example MSE, cosine similarity and DequantizeLinear (per-tensor/per-axis, int8/uint8).
- **QuantizeLinear and DynamicQuantizeLinear are excluded**, including nodes in subgraphs and local functions.

Before numerical comparison, output shapes and dtypes must match; broadcasting is forbidden. Integer outputs compare exactly. Non-finite outputs do not pass the finite-domain contract, even if both backends return matching NaNs. A batch check failing on non-finite values does not establish true dependence on batch composition.

Acceptance uses `abs(actual - reference) <= atol + rtol * abs(reference)`:

| Dtype | rtol | atol |
|---|---:|---:|
| float16 | 5e-3 | 5e-4 |
| float32 | 1e-5 | 1e-6 |
| float64 | 1e-12 | 1e-13 |

These are explicit project thresholds, not an ONNX-wide accuracy guarantee. LayerNormalization is currently covered in float32; other floating operations use float16/32/64. Each observation records its actual tolerance. Error summaries use only finite pairs and use JSON `null` when undefined.

## Reproduce a finding

Download a bundle through the dashboard or use a ZIP under `benchmarks/2026-09-06/observations/`. The observation's `artifacts.bundle` names its exact archive.

```bash
.venv/bin/python -m gero_stability.cli replay /path/to/extracted/bundle --output reports/replay

# Recheck all 14 bundles and calculate 80-digit Decimal diagnostics.
# Requires the recorded environment for exact observation identity.
.venv/bin/python tools/verify_snapshot.py --output reports/reverification
```

The verification script checks manifests, replays all flagged archives and evaluates the seven graph formulas with Decimal precision 80. It exactly converts stored binary inputs and epsilon, then exports diagnostic values as float64. Its formula checks are algorithmically independent of the backends; they are not third-party validation or a correct-rounding proof. In the LayerNormalization graph, unit scale and zero bias are explicitly checked.

The float32 LayerNormalization case produces 17 non-finite ORT values out of 68. The reference is finite but outside the chosen Decimal diagnostic tolerance. In the other six flagged graphs, ORT is within the diagnostic tolerance while the reference violates finite-domain or accuracy checks. See the full [diagnostic results](benchmarks/2026-09-06/reverification.json), including counts and finite absolute errors.

## Deduplication, baselines and CI

Case identities hash the graph and exact inputs. Comparison keys additionally include tolerances, optimization and contract settings. Observation identities include tester source, environment, outputs and checks, excluding timing. Identical repeated observations collapse; different environments and settings remain distinct. Similarity groups aid triage without claiming a shared cause.

```bash
.venv/bin/python -m gero_stability.cli run --random-cases 24 \
  --baseline benchmarks/2026-09-06/latest.json --output reports/next --fail-on regression
.venv/bin/python -m gero_stability.cli aggregate \
  benchmarks/2026-09-06/latest.json reports/next/latest.json --output reports/combined.json
```

A version regression requires a compatible baseline and a pass-to-numerical-failure transition. ORT versions may differ; changes to the tester, ONNX, NumPy, other dependencies or hardware fingerprint are treated as incompatible. Missing CPU identity requires separate hardware verification when comparing physical machines. Increasing error in an already failing case is not currently classified as a new regression.

GitHub Actions runs the tester and discovery benchmark on Linux and macOS and uploads full artifacts. The discovery benchmark records numerical findings without failing the job; `--fail-on finding` enables a stricter gate. Exit codes are 0 for completed discovery, 2 for a selected gate/argument failure and 3 for execution errors. CI results do not claim to match the recorded machine. Docker support is provided but was not validated for this release.

## Design

The static dashboard supports filters, search, sorting, pagination, shareable state, precise downloads and benchmark views. It uses synthetic data only. [Architecture](docs/ARCHITECTURE.md), [product design](docs/PRODUCT.md) and [original validation notes](docs/VALIDATION.md) document the intended service and bounded prototype. The latter records the initial local validation; the publication record above adds the full rerun, archive replays and Decimal check.

Primary specifications: [ReferenceEvaluator](https://onnx.ai/onnx/api/reference.html), [LayerNormalization](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html), [LogSoftmax](https://onnx.ai/onnx/operators/onnx__LogSoftmax.html), [ORT optimization settings](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html).
