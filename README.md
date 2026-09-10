# GERO Numerical Observatory

[Publication catalogue](reports/README.md) · [Cross-platform publication records](https://www.gero.uz/research/publication-records.html). The catalogue contains 65 reports, notes, essays and preprints, ordered newest first. Individual Zenodo deposits and the earlier 45-item collection are recorded separately.

- [MLX division: extreme scales change a scale-invariant derivative](audits/2026-09-10-mlx-divide-scale-autodiff/) — 1328 native comparisons; 600 mismatches before, zero after; 1886 compatibility comparisons pass. Fresh sequential CPU runs match the original evidence. [Zenodo archive](https://doi.org/10.5281/zenodo.22692449) · [GERO](https://www.gero.uz/research/articles/mlx-divide-scale-autodiff.html) · [56-second video](https://youtube.com/shorts/YWUx2uGVIpo).

- [MLX clipping: finite gradients become zero](audits/2026-09-10-mlx-clip-grad-norm-range/) — 219 input scenarios, 1701 assertions (633 numerical comparisons); 230 failures before, zero after; 39 additional compatibility/autodiff checks pass. CPU only; large-model performance untested. [Zenodo archive](https://doi.org/10.5281/zenodo.22693758).

[Four MLX audit archives now have verified Zenodo DOIs](docs/MLX_ZENODO_PUBLICATION_RECORDS.md).

- [MLX arctan2: scale changes a scale-invariant gradient](audits/2026-09-10-mlx-arctan2-scale-autodiff/) — 1368 native comparisons; 772 mismatches before, zero after; both failed intermediate patches retained.


- [MLX arcsinh/arccosh: representable gradients lost to intermediate overflow](audits/2026-09-10-mlx-inverse-hyperbolic-overflow/) — 518 native comparisons; 188 mismatches before, zero after the real-derivative patch.
- [MLX expm1: a nonzero tail derivative becomes zero](audits/2026-09-10-mlx-expm1-tail-vjp/) — 216 native checks, four repair controls and credited analogous JAX report.

- [MLX logcumsumexp: offsets change the gradient](audits/2026-09-10-mlx-logcumsumexp-shift-vjp/) — combined recurrence and known exp repair pass 1122 gradient checks plus 477 higher-order comparisons.

- [MLX logcumsumexp: missing curvature at a zero incoming gradient](audits/2026-09-10-mlx-logcumsumexp-hessian/) — research C++ prototype passes 477 checks; includes a 24-check GatherQMM follow-up.

- [MLX: a trainable output mask loses its gradient at zero](audits/2026-09-10-mlx-block-mask-output-vjp/) — 377 scenarios, 604 checks; local C++ one-line repair.

- [MLX sorted gather VJP preconditions](audits/2026-09-09-mlx-gather-vjp-preconditions) — repeated selections and broadcast gradients, 69 scenarios / 626 checks, bounded C++ repair and explicit CPU limits.

- [MLX Hadamard adjoint](audits/2026-09-09-mlx-hadamard-adjoint) — explicit matrix reference, 64 scenarios / 624 checks, measured energy updates and a local C++ patch.

- [MLX power: fixed-exponent derivatives at zero](audits/2026-09-09-mlx-power-zero-derivatives) — official wheel and C++ reproduction, 41 scenarios / 396 checks, targeted patch and test-name collision.

A reproducible CPU benchmark for mathematical contracts in ONNX graphs. It runs identical models in **ONNX Runtime** and the official **ONNX ReferenceEvaluator**, checks invariants, preserves artifacts and exports a filterable static dashboard.

[71-second video](https://www.youtube.com/shorts/KTLC9_FomBs) · [Interactive reports](https://www.gero.uz/stability/) · [Publication and limits](https://www.gero.uz/research/articles/onnx-numerical-observatory-reproduced-snapshot.html) · [Measured snapshot](benchmarks/2026-09-06/latest.json) · [Reverification](benchmarks/2026-09-06/reverification.json)

## MLX masked assignment: gradients cross batch examples

A fixed-mask `vmap` assigns a gradient to an unused source element. Actual
forward finite differences and an explicit-loop control agree; a local C++
patch passes 19 scenarios and 255 checks. [Report, tests and patch](audits/2026-09-09-mlx-masked-scatter-batch-vjp/README.md).

## MLX scatter extrema: lost gradients in block updates

A strict, in-bounds C++ block update loses a winning gradient. Independent
finite differences confirm the reference; a minimal extent correction passes
26 targeted checks. Historical ties are treated separately. [Report, tests and
patches](audits/2026-09-09-mlx-scatter-block-gradient/README.md).

## MLX cumprod: NaN Hessians at zero

A smooth polynomial has a correct first gradient but a non-finite Hessian.
Actual CPU reproduction; division-free O(N log N) prototype passes 57 scenarios
and 311 checks. Prior first-order fix credited; partial native rebuild and
performance limitations explicit. [Report, C++ tests and patch](audits/2026-09-09-mlx-cumprod-hessian/README.md).

## MLX complex autodiff, 9 September 2026

The real loss `Re(cos(i*x))` receives a gradient of the wrong sign, causing
the measured descent step to increase the loss. Seven VJPs omit complex
conjugation; arccosh also has a branch-sign error in JVP. A partial native
C++ rebuild changes the result from 46/131 to 131/131 passing scenarios.
The conjugation class has prior public reports; novelty and model effects
are unestablished.

[Report, patch and native reproduction](audits/2026-09-09-mlx-complex-autodiff/) ·
[GERO article](https://www.gero.uz/research/articles/mlx-complex-autodiff-reversed-gradient.html)

## MLX Muon convolution scaling, 9 September 2026

Equivalent 1×1 convolution and linear weights produce equal outputs, losses
and gradients, yet the tested convolution update is half the size. Scaling
uses the wrong dimensions after reshape. A minimal local reorder changes
the focused selection from 12/31 to 31/31 passes. CPU FP32; one defect,
without established full-model effects or maintainer acceptance.

[Read and reproduce](audits/2026-09-09-mlx-muon-convolution-scaling/) ·
[GERO article](https://www.gero.uz/research/articles/mlx-muon-convolution-scaling.html)

## MLX Gaussian NLL stability, 9 September 2026

Finite FP16 loss with a variance gradient of the wrong sign, plus spurious
Inf/NaN from intermediate squares. Native CPU reproduction, Decimal and
forward finite-difference references, 12 bounded regression methods and an
explicit counterexample to the proposed mitigation. Novelty and model-level
impact are unestablished.

[Read and reproduce](audits/2026-09-09-mlx-gaussian-nll-stability/)

## nntrainer Dropout routing audit, 9 September 2026

With two independent inputs and zero drop rate, the tested C++ layer sends
both gradients to input zero. A local index repair changes the measured
selection from 16/19 to 19/19 passing tests. Finite differences of the real
forward pass independently check the derivative. CPU FP32 layer-level
scope only; no full-model/device impact or original-discovery claim.

[Read and reproduce](audits/2026-09-09-nntrainer-dropout-routing/) ·
[Developer report](audits/2026-09-09-nntrainer-dropout-routing/REPORT.md)

## Causal attention audit, 7 September 2026

Native nntrainer causal attention includes future values in tested rectangular
and later incremental cases. A local repair passes 42 tests; 21 failed before.
The evidence includes source pins, native logs, finite differences, independent
batch checks, duplicate review and isolated patches.

[Reproduce the attention audit](audits/2026-09-07-causal-attention/README.md) ·
[Publication record](docs/ATTENTION_PUBLICATION_STATUS.md)

## Cosine and LayerNorm audit, 7 September 2026

Two reproduced implementation reports: finite-input cosine NaNs in MLX and an
incorrect affine LayerNorm input derivative in nntrainer. Local patches pass
101 cosine tests on each CPU/Metal backend and 42 native nntrainer CPU tests.
No exact duplicate identified in the recorded searches; no upstream acceptance
or model-impact claim.

[Reproduce both cases](audits/2026-09-07-cosine-layernorm/README.md) ·
[Read the technical note](https://www.gero.uz/research/articles/cosine-and-layernorm-contract-failures.html) ·
[YouTube](https://www.youtube.com/shorts/XpXZUXWnrW8) · [LinkedIn](https://www.linkedin.com/feed/update/urn:li:share:7502701332026441728/)

## Losses and autodiff audit, 7 September 2026

Three reproduced implementation cases in two numerical failure families: BCE tails
in MLX and nntrainer, and asymmetric logaddexp derivatives in MLX. Tested local
patches, source pins, before/after logs and duplicate-search evidence are included.

[Reproduce the audit](audits/2026-09-07-losses-autodiff/README.md) ·
[Read the technical note](https://www.gero.uz/research/articles/when-small-losses-and-gradients-disappear.html) ·
[Watch the 50-second Short](https://www.youtube.com/shorts/fcOZBV1P6mA) ·
[Publication record](docs/LOSSES_AUTODIFF_PUBLICATION_STATUS.md)

## Activation stability audit, 7 September 2026

Source-pinned ELU/SELU, GELU and Softplus observations, with local patch proposals,
CPU/Metal verification and an explicit prior-art search boundary. This separate
audit does not alter the ONNX benchmark counts or claim upstream confirmation.

[Reproduce the activation audit](audits/2026-09-07-activations/README.md) ·
[Read the technical note](https://www.gero.uz/research/articles/finite-inputs-nonfinite-activations-mlx-nntrainer.html)

## Small-sample validation note

The repository also contains a standalone, standard-library calculation for a
common assurance error: treating zero observed false positives as proof of a
zero false-positive rate. With zero failures in two representative negative
cases, the one-sided exact 95% upper bound is 77.6393%; 59 clean cases are
needed to push that bound below 5%, and 299 to push it below 1%.

[Read the evidence boundary](docs/ZERO_FALSE_POSITIVES.md) ·
[Run the calculation](tools/zero_failure_bounds.py) ·
[Read the public GERO note](https://www.gero.uz/research/articles/zero-observed-false-positives-small-sample.html)

## nntrainer average-pooling audit, 9 September 2026

A separate native C++ audit reproduces a backward-pass error in Samsung
nntrainer average pooling when `padding=same` is distributed asymmetrically.
For input `[1, 2; 3, 4]`, a 2×2 window, stride 1 and unit output gradients,
the tested source returns `[0.25, 0.25, 0.25, 0.25]`; the layer's own forward
Jacobian and central finite differences require
`[0.25, 0.75, 0.75, 2.25]`.

The focused matrix has 8 failures out of 58 tests before the two-line repair
and 58/58 passes after it. A 2021 upstream pull request already contained the
correct bottom/right loop bounds, so this repository does **not** claim first
discovery or established novelty.

[Evidence package](audits/2026-09-09-nntrainer-average-pool-padding/) ·
[Reproduction instructions](audits/2026-09-09-nntrainer-average-pool-padding/BUILD.md) ·
[Source patch](audits/2026-09-09-nntrainer-average-pool-padding/patches/source.patch) ·
[Public-history review](audits/2026-09-09-nntrainer-average-pool-padding/DUPLICATES.md) ·
[GERO article](https://www.gero.uz/research/articles/nntrainer-average-pool-asymmetric-padding.html) ·
[YouTube](https://www.youtube.com/shorts/TbRvf5eZgWQ) ·
[LinkedIn](https://www.linkedin.com/feed/update/urn:li:share:7503300049058271232/)

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

- [MLX gather_qmm: missing transpose in scale/bias gradients](audits/2026-09-09-mlx-gather-qmm-transpose-vjp/) — fresh CPU reproduction, exact loss example and C++ repair; 205 scenarios / 379 checks.
