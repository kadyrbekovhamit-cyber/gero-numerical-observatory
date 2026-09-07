# Duplicate review — 7 September 2026

The search covers public GitHub issues and pull requests, open and closed.
`duplicate-search.json` and `cosine-duplicate-search.json` retain queries,
timestamps, counts and retrieved responses. All recorded pages report
`incomplete_results=false`; each query returned fewer than 100 entries.
This bounds the review; it does not establish absolute novelty.

## Cosine: no exact match identified

The broad MLX `cosine` query returned 24 entries. Exact
`cosine_similarity_loss` returned one documentation change. The additional
`cosine` + `NaN` query returned a matrix-multiplication report whose cosine
is a diagnostic metric, not the failing primitive. `linalg.norm` + `zero`
returned no entries.

- [#336](https://github.com/ml-explore/mlx/pull/336): introduction of additional loss functions and ordinary tests, not this repair.
- [#3990](https://github.com/ml-explore/mlx/pull/3990): docstring reference resolution; no numerical change.
- [#3797](https://github.com/ml-explore/mlx/issues/3797): M5 NAX split-K GEMM; a different primitive and hardware path.
- [#4230](https://github.com/ml-explore/mlx/pull/4230): reduced-precision InstanceNorm. Excluded as a known neighboring normalization case.
- [#2246](https://github.com/ml-explore/mlx/pull/2246): LayerNorm two-pass variance. The moderate common-offset checks in this round passed; no new report is claimed there.

The previous local round recorded large-input cosine as a candidate but did
not include it in the three published losses/autodiff reports. This round
supplies the dedicated reference, zero-vector gradient case and validated
repair. It is one implementation report with two failure mechanisms.

## Quantization: observed failures excluded as prior art

[PR #3912](https://github.com/ml-explore/mlx/pull/3912) already describes
NVFP4's 16-wide legal dimensions interacting incorrectly with 32-wide
packing/tiles. Its discussion includes CPU quantize reshape rejection.
Our CPU dequantize rejection is the counterpart in the same packing path,
not an independently claimed discovery. The six Metal matmul failures at
K=80 and M=33/65 match its matrix-tail case. All are excluded from the new
case count, even though official MLX 0.32.2 still reproduces them here.

[PR #3473](https://github.com/ml-explore/mlx/pull/3473) discusses small
shape-dependent quantized-matmul differences from reduction order. Our
bound allows finite-precision reduction differences. Affine dequantize
intermediate rounding is also allowed; a float32 output alone does not
guarantee that every intermediate operation was float32.

## Masking

[Issue #2395](https://github.com/ml-explore/mlx/issues/2395) already covers
fully masked attention rows. Such rows are excluded from a new-defect claim.
[Issue #3897](https://github.com/ml-explore/mlx/issues/3897) concerns M5 masked
batch differences; this audit ran on M4 and does not claim M5 coverage.

No public issue, PR, comment or maintainer message was submitted in this round.
