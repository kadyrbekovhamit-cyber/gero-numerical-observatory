# Scale Changes the Gradient: MLX arctan2

Xamit Kadirbekov · GERO Research · 10 September 2026 · Independent numerical audit

Apple MLX 0.32.2 can return zero or infinity for a scale-invariant derivative. For any positive C, f(t) = atan2(C·t, C) = atan(t), so f′(1) = 0.5. The forward value remains finite, while extreme but representable scales break the automatic derivative. A local research patch passes all **1368 native comparisons**, versus **772 mismatches before**. These are comparisons for one numerical defect, not 772 distinct findings.

[Russian audit](README.ru.md) · [Native build instructions](BUILD.md) · [Patch](arctan2-scale-autodiff.patch) · [Fresh publication rerun](publication-rerun.json)

## A scale-invariant reproduction

```python
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import mlx.core as mx
mx.set_default_device(mx.cpu)
t = mx.array(1.0, dtype=mx.float32)
for power in (0, -80, 80):
    c = mx.array(2.0 ** power, dtype=mx.float32)
    f = lambda z: mx.arctan2(c * z, c)
    print(power, f(t).item(), mx.grad(f)(t).item())
# MLX 0.32.2 CPU: derivative 0.5, inf, 0.0 respectively.
```

The [recorded wheel reproduction](wheel-reproduction.json) and [full probe](probe.py) include the following results at t = 1. Forward values are approximately π/4. The exact first three derivatives are 0.5, −0.5, 0.5.

| Type | C | First derivative | Second / third | Forward finite difference |
|---|---|---|---|---|
| float32 | 1 | 0.5 | −0.5 / 0.5 | 0.5000209808 |
| float32 | 2⁻⁸⁰ | infinity | NaN / NaN | 0.5000209808 |
| float32 | 2⁸⁰ | 0 | NaN / NaN | 0.5000209808 |
| float64 | 1 | 0.5 | −0.5 / 0.5 | 0.5000000795 |
| float64 | 2⁻⁶⁰⁰ | infinity | NaN / NaN | 0.5000000795 |
| float64 | 2⁶⁰⁰ | 0 | NaN / NaN | 0.5000000795 |

Central finite differences evaluate the actual MLX forward function with h = 2⁻⁶ for float32 and 2⁻¹⁰ for float64. They independently corroborate the first derivative; they are not estimates of the higher derivatives. The exact identity atan2(Ct,C) = atan(t) supplies the higher-order reference. None of these inputs is the undefined origin.

A simpler first-order case also fits float16's representable range: at (y,x) = (1024,1024), the expected partials are ±0.00048828125, but MLX gives signed zeros. At (2⁻¹⁴,2⁻¹⁴), the expected partials are ±8192, but MLX gives infinities. The radial JVP, whose exact result is zero, becomes NaN at both scales.

## Cause and proposed correction

The [pinned upstream source](https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/mlx/primitives.cpp) computes the common denominator y² + x². Squaring can overflow or underflow although the final derivative is representable. The JVP also multiplies unnormalized coordinates and tangents before division.

For finite nonzero inputs, the [patch helper](scaled_partials.cpp.inc) uses a stopped-gradient scale s = max(|y|,|x|), normalized coordinates u = y/s, v = x/s, and D = u² + v². D lies between 1 and 2. A weighted partial has the exact form g·z/(s²D), with z = x for the y partial and z = −y for the x partial.

Normalization alone is insufficient: a representable weighted derivative can still vanish if the unweighted partial or an intermediate product underflows. The final helper selects one of three algebraically equivalent arithmetic orders:

- s ≥ 1: ((g/s)·z/D)/s.
- 0 < s < 1 with finite g/s: ((g/s)·(z/s))/D.
- 0 < s < 1 with overflowing g/s: ((g·z/D)/s)/s.

The code selects divisors before arithmetic, rather than evaluating all unsafe alternatives and selecting their outputs. Scale and the branch predicate are stopped for differentiation; within each chosen branch the expression remains the same mathematical weighted derivative for a fixed positive scale. The JVP processes only the tracked arguments. This patch does not depend on the separately reported exp correction.

## Failed intermediate patches are retained

The [initial normalization artifacts](initial-normalization/) passed the first 1350 comparisons. Extending coverage to 1362 revealed six real regressions: cases that passed before the patch and failed after it. For example, float32 y = 2⁻¹²⁰, x = 2³² and cotangent g = 2¹²⁰ require an x partial of approximately −2⁻⁶⁴; premature normalization lost it.

The [two-order weighted artifacts](weighted-two-order/) passed 1362 comparisons, but a further extension to 1368 exposed six remaining failures with very small weights. For float32 y = 2⁻⁸⁰, x = 2⁻¹²⁰ and g = 2⁻¹²⁰, the weighted y partial should be approximately 2⁻⁸⁰; the intermediate patch returned zero. These six also failed in the baseline, so they are remaining failures, not new regressions. The final three-order helper passes the extended suite. Earlier scripts and logs are historical evidence; use the top-level portable runner for reproduction.

## Validation and references

| Type | Comparisons | Before mismatches | After mismatches |
|---|---|---|---|
| float16 | 294 | 162 | 0 |
| bfloat16 | 300 | 166 | 0 |
| float32 | 387 | 222 | 0 |
| float64 | 387 | 222 | 0 |
| Total | 1368 | 772 | 0 |

The [native harness](native_regression.cpp) tests first-order VJPs and one-sided/joint JVPs across four types, three scales, seven coordinate orientations and multiple weights. It includes 84 radial and 84 angular invariant checks. Float32/64 also cover second and third derivatives, six mixed-Hessian comparisons, noncontiguous matrices and broadcasting from (2,1)/(1,3), with VJP reduction to the original shapes. Weighted anisotropic cases cover bfloat16, float32 and float64.

First-order references use the actual dtype-rounded inputs and a stable standard-library hypot formulation. Composite derivatives use closed-form atan derivatives. Selected anisotropic references use exact powers of two with corrections below the reference precision. Nonzero references are compared by relative error: float64 5e−13, float32 2e−5, float16 3e−3, bfloat16 2e−2. The same values are absolute tolerances for zero references. Tolerances and cases are identical before and after; zeroing any nonzero reference fails.

A fresh publication rerun compiled the harness and both baseline/patched primitives.cpp sequentially, then linked each against an existing CPU archive. All 1368 before rows and all 1368 after rows matched the original JSON results exactly. Compilation, linking and execution used approximately 4.27 seconds of child CPU time; this is a preparation measurement, not a performance benchmark. Numerical threads were limited to one and GPU arithmetic was not used.

Evidence: [before](run-before.json), [after](run-after.json), [original build logs](build-results.json), [fresh build logs](publication-build-results.json), [artifact validator](validate_artifacts.py), [artifact validation](artifact-validation.json), [source hashes](source-metadata.json), [native dependency pins](native-source-metadata.json), [checksums](SHA256SUMS).

The native baseline is ce916dbbcaa88e433b6fd1e60a17f766d49c27fe. Its arctan2 JVP/VJP methods match public main 81ba1c6a0e50a9268b931579c2d4f1158b9aab5a, checked again for publication. The archive SHA-256 is 7826e14e2d1ee526a16352829d460ae39c048e331c339be0b6f8d67d74e68a06. Validation overrides selected translation units before that archive; it is not a clean full build of current main. The complete dependency-archive rebuild recipe is supplied but was not executed for publication.

## Related reports and scope

Four focused GitHub issue/PR searches were repeated on 10 September 2026 and returned 11 unique results. [Search metadata](publication-duplicate-search.json) records the bounded search. [Issue #2451](https://github.com/ml-explore/mlx/issues/2451) and [PR #2453](https://github.com/ml-explore/mlx/pull/2453) concern an older wrong gradient formula at moderate inputs. [PR #3633](https://github.com/ml-explore/mlx/pull/3633) addresses partially tracked JVP argument indexing; [PR #3738](https://github.com/ml-explore/mlx/pull/3738) addresses stale tangent reuse. These differ from the scale overflow/underflow mechanism here. No direct duplicate was found in these results; absolute priority is not claimed.

This is an independent, AI-assisted report and research patch, without a claim of upstream acceptance. It has not been submitted as an MLX issue or PR. GPU behavior, speed, compile/vmap transformations, complete subnormal handling, nonfinite inputs, the undefined origin, and full-model consequences have not been established. Higher-derivative coverage is limited to float32/64. Complex inputs are outside this real-valued audit. Passing the suite does not prove numerical stability for every possible input and weight.

[Licenses and attribution](LICENSE.md): report and original diagrams CC BY 4.0; code and MLX-derived patches retain MIT notices. Author: Xamit Kadirbekov, GERO Research.
