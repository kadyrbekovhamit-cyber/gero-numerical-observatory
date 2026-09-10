# Same Ratio, Wrong Derivative: MLX Division at Extreme Scales

Xamit Kadirbekov · GERO Research · 10 September 2026 · Independent numerical audit

For positive C, f(t) = C/(C·t) = 1/t. At t = 1, the exact derivative is −1 at every scale. Apple MLX 0.32.2 on CPU instead returns −infinity at very small scales and signed zero at very large scales, despite a finite forward value of 1. A local research patch passes **1328 native comparisons**, versus **600 mismatches before**. A further **1886 comparisons** pass with the previously published inverse-hyperbolic and arctan2 patches applied together.

[Original Russian audit](README.ru.md) · [CPU build instructions](BUILD.md) · [Patch](divide-scale-autodiff.patch) · [Fresh rerun](publication-rerun.json)

## Minimal reproduction

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
    f = lambda z: c / (c * z)
    print(power, f(t).item(), mx.grad(f)(t).item())
# MLX 0.32.2 CPU: forward value 1; derivatives -1, -inf, -0.
```

The [wheel reproduction](wheel-reproduction.json) and [full probe](probe.py) record actual MLX forward, reverse and forward-mode behavior. All denominators in this example are finite and nonzero. The first three exact derivatives of 1/t at t = 1 are −1, 2 and −6.

| Type | C | First derivative | Second / third | Forward finite difference |
|---|---|---|---|---|
| float32 | 1 | −1 | 2 / −6 | −1.00024604797 |
| float32 | 2⁻⁸⁰ | −infinity | NaN / NaN | −1.00024604797 |
| float32 | 2⁸⁰ | −0 | NaN / NaN | −1.00024604797 |
| float64 | 1 | −1 | 2 / −6 | −1.00000095368 |
| float64 | 2⁻⁶⁰⁰ | −infinity | NaN / NaN | −1.00000095368 |
| float64 | 2⁶⁰⁰ | −0 | NaN / NaN | −1.00000095368 |

Central finite differences of the actual MLX forward function use h = 2⁻⁶ for float32 and 2⁻¹⁰ for float64. They corroborate the first derivative only. Higher-order references come from the exact identity 1/t.

For a/b at (a,b) = (C,C), the expected partials are [1/C, −1/C]. With float16 C = 1024, the denominator partial becomes signed zero instead of −1/1024. With C = 2⁻¹⁴, it becomes −infinity instead of −16384. Both reference values are representable. A joint equal-direction JVP should cancel to zero but becomes NaN at both scales.

## Cause and proposed repair

The [pinned upstream source](https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/mlx/primitives.cpp) computes the denominator derivative using −(g·a)/(b²). The product or square can overflow or underflow even when the final weighted derivative is representable. Computing an unweighted partial first is also insufficient when a later weight would restore a representable result.

The [helper](weighted_partial.cpp.inc) orders a and g by absolute magnitude, retaining their signs: L has the larger magnitude and S the smaller. It chooses among algebraically equivalent arithmetic orders:

| Condition | Weighted denominator partial |
|---|---|
| abs(b) ≥ 1 and abs(S) ≥ abs(b) | −(L/b)·(S/b) |
| abs(b) ≥ 1 and abs(S) < abs(b) | −((L/b)·S)/b |
| abs(b) < 1 and L/b is finite | −(L/b)·(S/b) |
| abs(b) < 1 and L/b overflows | −((L·S)/b)/b |

Divisors are selected before arithmetic; the code does not compute all unsafe alternatives and then select their outputs. Only choice predicates are stopped for differentiation. The input values and weights remain differentiable. The patch changes the real denominator branch of both JVP and VJP. Original numerator and complex branches are preserved, including conjugation behavior.

Exact power-of-two stress cases cover loss of a/b before weighting, overflow of g/b, overflow of both the numerator product and denominator square, and underflow of both. The harness also reverses the roles of a and g and checks negative numerators. This is a research prototype; the selected cases do not establish stability for every combination of values and weights.

## Validation

| Type | Comparisons | Before mismatches | After mismatches |
|---|---|---|---|
| float16 | 280 | 114 | 0 |
| bfloat16 | 280 | 122 | 0 |
| float32 | 376 | 182 | 0 |
| float64 | 376 | 182 | 0 |
| complex64 controls | 16 | 0 | 0 |
| Total | 1328 | 600 | 0 |

The [native harness](native_regression.cpp) covers one-sided and joint JVP/VJP rules, signs, zero numerators and zero weights, common and opposite scaling directions, broadcasting from (2,1)/(1,3), noncontiguous matrices, and 112 weighted exponent cases. Float32/64 coverage includes first through third derivatives of 1/t at six points and six mixed-Hessian comparisons.

There are 36 higher-order zero-gradient checks: for R(t) = (C/(Ct) − 1)² at 1, the first three derivatives are 0, 2 and −12; for I(t) = (Ct)/(Ct), all three are zero. Moderate complex64 controls use independent std::complex<double> references and pass before and after. These controls preserve existing complex behavior; they do not fix or establish extreme-scale complex stability.

Nonzero references use relative tolerances: float64 5e−13, float32 2e−5, float16 3e−3, bfloat16 2e−2; complex components use 8e−6. The same values are absolute tolerances for zero references. Inputs, references and tolerances are identical before and after. Zeroing any nonzero reference fails.

A fresh publication run compiled the harness and both baseline/patched translation units independently, then compiled the combined three-patch unit and both bundled compatibility harnesses. Every row matched the historical results exactly. The compatibility suites contain 518 inverse-hyperbolic and 1368 arctan2 comparisons, all passing. Compilation, linking and execution used approximately 7.98 seconds of child CPU time, sequentially, with one numerical thread. This preparation timing is not a performance benchmark.

Evidence: [before](run-before.json), [after](run-after.json), [hyperbolic compatibility](compat-run-hyperbolic.json), [arctan2 compatibility](compat-run-arctan2.json), [fresh build logs](publication-build-results.json), [artifact checks](artifact-validation.json), [source metadata](source-metadata.json), [native dependency pins](native-source-metadata.json), [checksums](SHA256SUMS).

The native baseline is ce916dbbcaa88e433b6fd1e60a17f766d49c27fe. Its targeted Divide methods match public main 81ba1c6a0e50a9268b931579c2d4f1158b9aab5a. The CPU archive SHA-256 is 7826e14e2d1ee526a16352829d460ae39c048e331c339be0b6f8d67d74e68a06. Selected translation units override that prebuilt archive; this is not a clean full build of current main. The supplied full dependency rebuild recipe was not executed for publication.

## Related reports and limits

The original four focused GitHub searches returned 18 unique issues/PRs; their [metadata and links](duplicate-search.json) are retained. [PR #2178](https://github.com/ml-explore/mlx/pull/2178) concerns complex VJP behavior that this patch preserves. [PR #3733](https://github.com/ml-explore/mlx/pull/3733) concerns zero-cotangent behavior at singular sqrt/rsqrt inputs; this reproduction has a nonzero denominator. [Issue #2451](https://github.com/ml-explore/mlx/issues/2451) concerns an older arctan2 formula. No direct duplicate was identified in the bounded results; absolute priority is not claimed. The same four searches were repeated for publication and again returned 18 unique results, without incomplete results or network failures; see the [publication search log](publication-duplicate-search.json).

GPU behavior, speed, compile/vmap transformations, complete subnormal handling, zero denominators, nonfinite inputs and full-model impact were not tested. Higher derivatives were tested only for float32/64. The compatibility result concerns only the three named patches and their stated suites. It does not cover every previous MLX repair.

Independent, AI-assisted research and publication preparation; no upstream acceptance, production-readiness, security impact or award claim. This report is published in the author's repository and has not been submitted as an MLX issue or PR. [Licenses](LICENSE.md): original report and diagrams CC BY 4.0; code and MLX-derived patches retain MIT notices.
