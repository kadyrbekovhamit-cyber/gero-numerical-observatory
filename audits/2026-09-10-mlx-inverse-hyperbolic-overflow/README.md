# Finite Values, Missing Derivatives: MLX arcsinh and arccosh

10 September 2026 · Xamit Kadirbekov · Independent numerical audit

In Apple MLX 0.32.2 on CPU, `arcsinh` and `arccosh` can return finite values but zero JVP and VJP because their derivative formula squares a large input. At float16 `x=1000`, both functions return **7.6015625**, while their derivative is **0 instead of approximately 0.001**. The expected derivatives are representable in that dtype.

An isolated native CPU rerun reproduces **188 mismatches out of 518 comparisons before the patch and 0 after**. This is one shared implementation defect affecting two operations, not 188 separate findings. GPU behavior and performance were not evaluated.

[Detailed Russian report](README.ru.md) · [Build instructions](BUILD.md) · [Python reproduction](probe.py) · [C++ regression](native_regression.cpp) · [Research patch](inverse-hyperbolic-real.patch)

## Minimal reproduction

```python
import mlx.core as mx
mx.set_default_device(mx.cpu)
x = mx.array(1000., dtype=mx.float16)
for f in (mx.arcsinh, mx.arccosh):
    print(f(x))                                    # 7.6015625
    print(mx.grad(f)(x))                            # 0
    print(mx.jvp(f, [x], [mx.ones_like(x)])[1][0])   # 0
```

The full [wheel probe](wheel-reproduction.json) includes four dtypes and evaluates the analytic reference after rounding the input to its actual dtype. Reference slopes are `1/hypot(x,1)` for arcsinh and `(1/sqrt(x-1))/sqrt(x+1)` for arccosh.

| Operation | dtype | Input | MLX JVP / VJP, weight 1 | Analytic derivative |
|---|---|---|---|---|
| arcsinh | float16 | 1000 | 0 / 0 | 0.000999999500000375 |
| arccosh | float16 | 1000 | 0 / 0 | 0.001000000500000375 |
| both | float32 | approximately 1e20 | 0 / 0 | approximately 1e-20 |
| both | float64 | 1e200 | 0 / 0 | approximately 1e-200 |
| both | bfloat16 | approximately 9.97277e19 | 0 / 0 | approximately 1.00273e-20 |

Large negative inputs also lose the positive arcsinh derivative. Arccosh is tested inside its real domain, `x>1`.

## A small local slope can matter

For `f(t)=arcsinh(C*t)` or `arccosh(C*t)` at `t=1`, take `C=2^80` in float32 or `C=2^600` in float64. Forward values remain finite. The expected first three derivatives are approximately **1, -1, 2**; finite-C corrections are below the precision displayed.

| Derivative order | Original MLX | Patched MLX | Analytic reference |
|---|---|---|---|
| First | 0 | 1 | approximately 1 |
| Second | -0 | -1 | approximately -1 |
| Third | NaN | 2 | approximately 2 |

Central differences of the actual MLX forward function give **1.0001220703125** in float32 with step `2^-6`, and **1.00000031790114** in float64 with step `2^-10`. They independently support a first derivative near one. Higher derivatives use analytic references. This establishes an order-one gradient error in a small composition; effects on training a particular model were not measured.

## Cause and research repair

Pinned [ArcCosh::jvp](https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/mlx/primitives.cpp#L424) and [ArcSinh::jvp](https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/mlx/primitives.cpp#L478) compute `rsqrt(x*x-1)` and `rsqrt(x*x+1)`. Both VJPs delegate to JVP. Once the intermediate square becomes infinity, its reciprocal square root becomes zero, even though the mathematical derivative is still representable.

For real dtypes the patch selects `s=max(1,abs(x))` and uses:

```text
arcsinh'(x) = rsqrt((x/s)^2 + (1/s)^2) / s
arccosh'(x) = rsqrt(((x-1)/s) * ((x+1)/s)) / s, x>1
```

No large square is formed. Factoring arccosh also reduces cancellation near one: at float32 `x≈1.00010001659`, the baseline slope is 70.7048111 against reference 70.7030442.

The scale is marked `stop_gradient`. Each identity is valid for every fixed positive scale, so a fixed local choice can be used while differentiating with respect to x. This avoids unnecessary differentiation paths through the scale. Second and third derivatives were tested; arbitrary-order correctness is not established.

For nonfinite inputs the helper selects scale one; complete NaN/infinity behavior is outside this audit. Original complex branches and VJP bodies are preserved, without claiming to fix existing complex autodiff limitations. This patch is independent of the float64-exp control required by the earlier logcumsumexp and expm1 reports.

## Native evidence

| dtype | Comparisons | Before: mismatches | After: mismatches |
|---|---|---|---|
| float32 | 195 | 73 | 0 |
| float64 | 195 | 67 | 0 |
| float16 | 64 | 24 | 0 |
| bfloat16 | 64 | 24 | 0 |
| Total | **518** | **188** | **0** |

[Before](run-before.json) · [After](run-after.json) · [Publication rerun](publication-rerun.json) · [Fresh build commands](publication-build-results.json) · [Artifact verification](artifact-validation.json).

Float32/float64 tests cover zero, small, moderate and large arguments, the nearest representable input above one for arccosh, directions `0, 1, -0.5, 2`, second and third derivatives, large-scale compositions, and dense/noncontiguous 2×3 arrays with mixed weights. Float16/bfloat16 cover first-order scalar JVP/VJP; their higher derivatives and arrays were not tested.

Relative tolerances for nonzero references: `3e-13` (float64), `8e-6` (float32), `3e-3` (float16), `2e-2` (bfloat16). At a zero reference the same numerical tolerance is absolute. Every tested nonzero slope lost to zero has relative error one and fails. Reference inputs are rounded to the tested dtype first.

An initial harness created some double values as float32 before casting. Its 390 rows, including 31 patched mismatches, remain in [initial-harness-attempt](initial-harness-attempt/) for provenance and are not evidence of double behavior. Correcting scalar construction, without changing the derivative patch or float32/float64 tolerances, produced 390/390; adding half and bfloat16 expanded the suite to 518/518.

The publication rerun freshly compiled the test and both baseline/patched `primitives.cpp` translation units, then linked each before the same existing CPU archive. All 518 JSON rows on each side exactly match the completed original audit. This is **not a full clean build of current main**. [BUILD.md](BUILD.md) describes the portable runner and an unexecuted fresh-archive setup recipe. Compiled binaries are excluded. Historical scripts prefixed `original-` preserve the original audit setup; use `build_and_test.py` for a portable rerun.

## Versions, public history and limits

Wheel: **MLX 0.32.2**. Native baseline: `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`. Public main reviewed on 10 September 2026: `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`. All four affected methods match between these snapshots. [Source hashes](source-metadata.json) and [original native provenance](native-source-metadata.json) are retained.

The refreshed [bounded GitHub search](publication-duplicate-search.json) found three arcsinh records and two arccosh records. [#4227](https://github.com/ml-explore/mlx/pull/4227) adds gradient-reference tests on moderate inputs; [#3678](https://github.com/ml-explore/mlx/pull/3678) adds aliases; [#3080](https://github.com/ml-explore/mlx/issues/3080) concerns arm64 Float16 JIT compilation. The inspected descriptions do not report this overflow counterexample. Queries combining asinh/overflow and acosh/gradient returned no matches. This does not establish absolute priority or exhaust all discussions.

GPU, speed, compile, vmap, complete nonfinite-domain behavior, full-model effects, security impact and bounty eligibility were not established. Calculations were sequential with one numerical thread; physical CPU affinity was not configured. The native archive has no Metal backend. The original wheel import initialized Metal before arithmetic was explicitly selected on CPU.

AI assisted investigation, code, writing and media production. This is an independent author's report in their own repository, not an upstream MLX submission. Text and original diagrams: CC BY 4.0; code and MLX-derived source/patch: MIT, with [license notices](LICENSE.md).
