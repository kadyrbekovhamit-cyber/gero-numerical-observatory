# nntrainer DivideLayer: finite forward values, incorrect denominator gradients

Independent GERO research by Xamit Kadirbekov · 14 September 2026

**One implementation defect reproduced in the actual C++ DivideLayer.** At pinned nntrainer main `a7ea056e79ab8e14447ea305c1b634e233343258`, finite float32 inputs can produce a zero, infinity or NaN denominator gradient when the exact gradient is finite and representable. A local CPU FP32 correction removes **1,302 failing inputs out of 4,096**. Restoring the original source restores all 1,302 failures.

## The invariant and minimal examples

For `y = a / b`, with incoming gradient `g` and nonzero `b`, the vector–Jacobian product is:

```text
dy/da × g = g / b
dy/db × g = -g × a / b²
```

The tested forward values and both exact gradients are zero or normal, finite float32-range quantities. A mathematically equivalent implementation should not lose them through avoidable intermediate overflow or underflow.

| Float32 inputs | Forward | Denominator gradient before | Expected (approximately) |
|---|---:|---:|---:|
| a = b = 1e20, g = 1 | 1 | 0 | −1e−20 |
| a = b = 1e−30, g = 1 | 1 | −Infinity | −1e30 |
| a = b = 1e−30, g = 0 | 1 | NaN | 0 |

Decimal literals above are rounded to float32 before execution. The archive records their exact bits and exact rational expected values; the decimal gradient labels are explanatory approximations.

## Actual implementation and cause

The [pinned source](https://github.com/nntrainer/nntrainer/blob/a7ea056e79ab8e14447ea305c1b634e233343258/nntrainer/layers/divide_layer.cpp) forms the denominator gradient with float32 tensor operations equivalent to:

```cpp
incoming.multiply(numerator.multiply(-1)).divide(denominator.pow(2))
```

Squaring a large denominator overflows before division. Squaring a tiny denominator underflows to zero. The numerator product can also overflow or underflow. These intermediates need not have the same representability as the final gradient. Replacing the expression with another float32 ordering alone is not a complete range solution.

The native probe instantiates `DivideLayer`, `InitLayerContext`, `Var_Grad` and `RunLayerContext`, then invokes the real forward and backward methods. It contains no replacement implementation or expected-value formula.

## Validation

A separate clean source export and build were used. No earlier Pow correction is present. Of 2,319 recorded source files, only DivideLayer and its regression-test file differ after the candidate correction.

The deterministic input set contains 4,096 unique `(a,b,g)` triples: decimal examples, signed powers of two, and seeded finite-range controls. Inputs are finite, denominators are nonzero, and each exact output is zero or has magnitude between `2^-120` and `2^120`. This selected stress set does **not** estimate how often the defect occurs in ordinary training.

The independent oracle uses exact Python `Fraction` arithmetic and an integer round-to-nearest-even float32 encoder. All 12,288 expected component values were cross-checked using 200-digit Decimal arithmetic, standard binary conversions and adjacent-float distance checks. The failure threshold is more than three float32 ULP, with either signed zero accepted; a nonfinite result fails against a finite oracle.

| Source state | Unique failing inputs | Forward failures | Numerator-gradient failures | Denominator-gradient failures |
|---|---:|---:|---:|---:|
| Original | 1,302 / 4,096 | 0 | 0 | 1,302 |
| Candidate correction | 0 / 4,096 | 0 | 0 | 0 |
| Original formula restored | 1,302 / 4,096 | 0 | 0 | 1,302 |
| Candidate restored | 0 / 4,096 | 0 | 0 | 0 |

The original failures comprise 494 zeros in place of nonzero gradients, 409 infinities, 392 NaNs and seven finite inaccurate values. Candidate outputs match the oracle exactly on this set (zero ULP, ignoring signed zero).

Each triple was evaluated in three tensor shapes, with two backward calls per shape: 24,576 coordinate observations, **not** 24,576 independent inputs. Shape invariance, repeated-call invariance and preservation of input bits passed. Forward values and numerator gradients remain byte-identical between original and corrected runs.

All **25 focused Divide tests pass**: 22 existing semantic cases plus three new regression tests covering six scalar examples. Restoring the old implementation makes all three new regression tests fail; restoring the candidate makes them pass. Original and mutation raw outputs are byte-identical, as are candidate and restored-candidate outputs. The full nntrainer suite was not run.

## Candidate correction and limits

The candidate widens intermediate multiplication and division to binary64 for the equal-shape, contiguous FP32 path, then rounds the result back to float32. Products and quotients constructed from finite float32 inputs fit within binary64's exponent range; the final float32 result can still legitimately overflow when the exact derivative is out of range. Those cases are outside this audit's finite-output set.

The patch preserves the existing fallback for other shapes and data types, and the prior arithmetic for nonfinite inputs or zero denominators. It is a **local correction prototype**, not an accepted upstream change. FP16, broadcasting and broadcast-gradient reduction, noncontiguous tensors, GPU/device kernels, performance and complete-model training were not validated.

Measured effect: the denominator gradient supplied by this C++ layer changes from incorrect or nonfinite values to the expected finite values in the selected cases. No Samsung device, deployed model, user incident, training accuracy or financial loss was measured.

## Prior work and duplicate review

This is not a new numerical phenomenon. GERO previously reported [an analogous divide-autodiff range failure in Apple MLX](https://www.gero.uz/research/articles/mlx-divide-scale-autodiff.html). This report documents a separately executed nntrainer implementation defect.

On 14 September 2026, bounded searches of nntrainer issues and pull requests, the DivideLayer file history and originating PR, the prior GERO nntrainer bundle and the canonical publication catalog found no exact prior report or correction for this denominator-gradient case. PR 2912 concerns L2 preprocessing; PR 4088 concerns AVX2 elementwise division. Neither addresses this backward expression. The archive preserves queries and responses. Search coverage is not a guarantee of exhaustive novelty.

## Evidence and rights

The evidence package includes source hashes and pinned build-source subset, explicit omitted-asset manifest, submodule sources, native probe, exact inputs, raw outputs for four source states, candidate patch, focused-test XML/logs and reproduction instructions. The complete original source remains available at the pinned upstream commit; unrelated application assets and packaging archives are omitted from the compact build-source copy.

nntrainer retains its upstream Apache-2.0 notices and third-party licenses. Original GERO probe code is MIT; report and original diagrams are CC BY 4.0. AI-assisted research preparation and preset synthetic English narration are disclosed. Independent work; no affiliation with Samsung or the nntrainer maintainers is implied.

## Publication and reproducibility links

[Read on GERO](https://www.gero.uz/research/articles/nntrainer-divide-gradient-extremes.html) · [Zenodo archive](https://zenodo.org/records/22755613) · [Download frozen evidence](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/raw/refs/heads/main/catalog/artifacts/gero-nntrainer-divide-evidence-2026-09-14.zip)

Evidence SHA-256: `a62d6f9aeb36604df043520cf2a6ce51b614734c9fe720ca7821ebbd9143862f`.

[Maintainer issue (submitted, not acceptance)](https://github.com/nntrainer/nntrainer/issues/4336).
