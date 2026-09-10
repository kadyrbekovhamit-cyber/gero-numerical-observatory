# MLX expm1: reverse-mode derivatives disappear in the negative tail

Xamit Kadirbekov · 10 September 2026 · MLX 0.32.2 · CPU

For `f(x)=1e8*expm1(x)` at float32 `x=-20`, the mathematical derivative is **0.2061153622**. The tested MLX forward mode returns **0.2061153501**, while reverse mode returns **0**. A local VJP repair combined with a control for the already reported float64-exp defect passes **216 native comparisons**, including second and third derivatives.

The VJP patch alone is **not a complete float64 repair**: the existing CPU exp defect can turn a previously finite derivative at `x=100` into infinity. Both changes and four control variants are included. No performance qualification or upstream acceptance is claimed.

[Detailed Russian report](REPORT_RU.md) · [Build instructions](BUILD.md) · [Python reproduction](probe.py) · [C++ regression](native_regression.cpp) · [VJP patch](expm1-vjp.patch)

## Reproduce the failure

```python
import mlx.core as mx
mx.set_default_device(mx.cpu)
x = mx.array(-20., dtype=mx.float32)
f = lambda z: 1e8 * mx.expm1(z)
print(mx.grad(f)(x))                            # 0
print(mx.jvp(f, [x], [mx.ones_like(x)])[1][0])   # ~0.20611535
```

Run `python3 probe.py` with MLX 0.32.2 installed. It limits numerical threads to one and selects CPU. The wheel may initialize Metal at import; the arithmetic is explicitly on CPU. The native archive has no Metal backend.

| Input | dtype | Forward expm1 | JVP, direction 1 | VJP, cotangent 1 | Analytic derivative |
|---|---|---:|---:|---:|---:|
| -20 | float32 | -1 | 2.06115347e-9 | **0** | 2.06115362e-9 |
| -18 | float32 | -1 | 1.52299915e-8 | **0** | 1.52299797e-8 |
| -10 | float32 | -0.9999545813 | 4.53999310e-5 | 4.54187393e-5 | 4.53999298e-5 |
| -40 | float64 | -1 | 4.24835372e-18 | **0** | 4.24835426e-18 |

The expected derivatives are normal representable values, so derivative underflow does not explain the zeros. The reference is the analytic derivative `exp(x)`, numerically evaluated with Python `math.exp` and C++ `std::exp` in double. The real MLX JVP provides a separate control. [Wheel measurements](wheel-reproduction.json).

Finite differences of the rounded float32 forward output near -20 are not used as a reference: that output is already -1 at neighboring points. This report concerns the derivative of the mathematical operation used by autodiff.

## Cause and repair dependency

The pinned [Expm1::vjp](https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/mlx/primitives.cpp) multiplies the incoming cotangent by `outputs[0] + 1`. Once `expm1(x)` rounds to -1, the derivative vanishes. Relative accuracy is also lost before complete saturation. `Expm1::jvp` instead computes `exp(primals[0])` directly.

The [VJP patch](expm1-vjp.patch) uses that same direct exp formula. It eliminates the tested float32 mismatches, but exposes the separate CPU double-exp precision/range problem already described in [MLX #3047](https://github.com/ml-explore/mlx/issues/3047). At float64 `x=100`, the original VJP is finite while a VJP-only replacement using the defective exp returns infinity. At `x=-1`, it loses double precision.

The included [exp control](exp-control/README.md) uses scalar libm to isolate correctness. It is reused known work, not a new discovery. Neither its speed nor the added exp evaluation in VJP has been benchmarked.

| Native variant | Comparisons | float32 mismatches | float64 mismatches | Total |
|---|---:|---:|---:|---:|
| Original VJP and exp | 216 | 25 | 67 | 92 |
| VJP repair only | 216 | 0 | 87 | 87 |
| Exp control only | 216 | 25 | 28 | 53 |
| VJP repair + exp control | 216 | 0 | 0 | **0** |

[Before](run-before.json) · [VJP only](run-vjp-only.json) · [Exp only](run-exp-only.json) · [Combined](run-combined.json). These are comparisons, not distinct defects. Existing float64 JVP errors belong to the known exp dependency.

The native suite calls actual MLX primitives. It covers float32/float64 scalar points from -80 to 20, directions 0/1/-0.5/2, extra double points -700/-100/100/700, second and third reverse derivatives at -40/-20/-10/0/1, a 2×3 matrix, noncontiguous input, mixed weights and scalar broadcasting. Relative tolerances are `4e-6` for float32 and `2e-14` for float64; expected zeros require exact zero. There is no large absolute tolerance that would hide missing small derivatives. Reference inputs are rounded to their actual dtype first.

## Public history and scope

No matching MLX negative-tail VJP report was found in the recorded public search. Its six broad `expm1` results were [#1277](https://github.com/ml-explore/mlx/issues/1277) (forward Metal behavior), [#1281](https://github.com/ml-explore/mlx/pull/1281) (older tolerances/build), [#973](https://github.com/ml-explore/mlx/pull/973) (operation addition), [#3080](https://github.com/ml-explore/mlx/issues/3080) (arm64 CPU JIT/Float16 compilation), [#4257](https://github.com/ml-explore/mlx/pull/4257) (complex rejection) and [#4227](https://github.com/ml-explore/mlx/pull/4227) (analytic gradient tests, with no defects reported in its chosen series). [Search evidence](publication-duplicate-search.json).

An analogous reverse-mode expm1 failure was already reported for **JAX** in [JAX #39794](https://github.com/jax-ml/jax/issues/39794), opened 6 August 2026. This package does not rerun JAX and does not claim a globally new mechanism. The bounded MLX search does not prove absolute priority.

The wheel is 0.32.2; native baseline is `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`; reviewed public main is `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`. Complete `Expm1::vjp` and `Expm1::jvp` methods match between baseline and saved main. [Source hashes](source-metadata.json) · [Native provenance](native-source-metadata.json).

Validation uses isolated `primitives.cpp` and `unary.cpp` overrides before a pre-existing CPU archive; it is **not a complete current-main build**. CPU float32/float64 were tested. GPU, float16/bfloat16, compile, vmap, NaN/±inf, model-level effects and performance were not. Complex expm1 is unsupported and excluded. The finite suite does not establish arbitrary-order autodiff correctness.

Numerical work was sequential with one numerical thread and no GPU computation. [Publication rerun evidence](publication-rerun.json), [artifact verification](artifact-validation.json), source code and hashes are included; compiled binaries and unfinished scouts are excluded. AI assisted investigation, code and publication preparation. This is an independent author repository report, not an upstream MLX submission.
