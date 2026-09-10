# MLX logcumsumexp: a common offset changes the gradient

10 September 2026 · Xamit Kadirbekov · Independent GERO Research

**Reproduced in MLX 0.32.2 on CPU.** For two equal float32 inputs [100000000, 100000000], the gradient of the final logcumsumexp output is [1, 1], while the analytic result and MLX's equivalent logsumexp operation give [0.5, 0.5].

A sequential research VJP prototype fixes this normalization loss in the checked cases. Combined with a separate repair for the **already reported** CPU float64 exp limitation, it passes 1122 gradient checks and the earlier 477 higher-order comparisons. This is not a production-qualified patch or a performance result.

## A translation-invariant derivative

For f(x) = log(exp(x₀) + exp(x₁)), adding the same finite scalar c to both inputs adds c to the value without changing the gradient:

    f(x + c·1) = c + f(x)
    ∂f/∂xⱼ = exp(xⱼ) / Σₖ exp(xₖ)
    gradient at [c,c] = [1/2, 1/2].

The inputs below are the actual representable values passed to the runtime. The equal-input case does not depend on recovering a tiny difference between rounded inputs.

| float32 input | Last logcumsumexp gradient | logsumexp gradient | Analytic reference |
|---|---|---|---|
| [0, 0] | [0.5, 0.5] | [0.5, 0.5] | [0.5, 0.5] |
| [1000, 1000] | approximately [0.49998546, 0.49998546] | [0.5, 0.5] | [0.5, 0.5] |
| [100000, 100000] | approximately [0.49891850, 0.49891850] | [0.5, 0.5] | [0.5, 0.5] |
| [100000000, 100000000] | **[1, 1]** | [0.5, 0.5] | [0.5, 0.5] |
| [-100000000, -100000000] | **[1, 1]** | [0.5, 0.5] | [0.5, 0.5] |

The float64 equal-input case also reaches [1, 1] at c = 10²⁰. These statements concern derivatives of the specified real-valued operation evaluated on representable inputs, not differentiation of the discontinuous map induced by rounding itself.

Minimal installed-package reproduction:

~~~python
import mlx.core as mx
mx.set_default_device(mx.cpu)
x = mx.array([1e8, 1e8], dtype=mx.float32)
print(mx.grad(lambda z: mx.logcumsumexp(z)[-1])(x))  # [1, 1]
print(mx.grad(mx.logsumexp)(x))                     # [0.5, 0.5]
~~~

The [reproduction script](reproduce.py) records a wider input table and a counterexample to global centering. [Saved results](wheel-reproduction.json) identify the installed package and CPU device. Finite differences with a step below the float32 spacing near 10⁸ are not used as evidence.

## Why the normalization is lost

The inspected [Scan::vjp implementation](https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/mlx/primitives.cpp) uses the rounded forward scan output in exponential differences in the backward computation.

For equal inputs, the exact final output is c + log(2). At c = 10⁸, adjacent float32 values are 8 apart, so the forward output rounds to c. That value can be a reasonable rounded forward result, yet reusing it to reconstruct exponential weights loses the log(2) normalizer: the backward calculation yields unit weights instead of half weights.

The bug is the resulting backward rule. It is not a claim that the rounded forward value must retain a fraction smaller than one representable step.

## Subtracting a single global maximum is insufficient

A scan has different prefixes. An element beyond the prefix being differentiated must not distort its derivative.

For x = [0, 0, 10⁸], differentiate the second output. The original scan gives [0.5, 0.5, 0]. A wrapper that subtracts stop_gradient(max(x)) from the entire vector before the scan and adds it back gives **[1, 1, 0]**. The future large element shifts the earlier prefix into a region that loses precision.

This counterexample was rerun on the actual installed package. It rules out that simple wrapper as a general repair.

## A normalized recurrence

For a finite real inclusive prefix, maintain a stopped running maximum mᵢ and a normalized mass sᵢ = Σⱼ≤ᵢ exp(xⱼ − mᵢ):

    aᵢ = sᵢ₋₁ exp(mᵢ₋₁ − mᵢ)
    bᵢ = exp(xᵢ − mᵢ)
    sᵢ = aᵢ + bᵢ
    ρᵢ = aᵢ / sᵢ
    ηᵢ = bᵢ / sᵢ.

For the first position, ρ₀ = 0 and η₀ = 1. If Lᵢ is the prefix logsumexp, then:

    dLᵢ = ρᵢ dLᵢ₋₁ + ηᵢ dxᵢ.

For incoming cotangents gᵢ, reverse accumulation becomes:

    λᵢ = gᵢ + ρᵢ₊₁ λᵢ₊₁
    grad_xᵢ = ηᵢ λᵢ.

This recurrence is linear in g and avoids logarithms of cotangent magnitudes or sign branches. To retain higher derivatives through the masses, the first mass stays as exp(x₀ − stop_gradient(x₀)) in the graph rather than becoming a constant.

Reverse scans reverse the index order; exclusive scans shift the cotangent correspondence by one position. Empty prefixes are excluded from smooth test losses. The algebraic derivation assumes finite real inputs; it does not define a complete derivative convention at infinities or NaNs.

See [implementation](stable_scan_vjp.cpp.inc) and [research patch](stable-logcumsumexp-vjp-prototype.patch). The forward pass is unchanged. The prototype constructs O(n) sequential graph steps along the scan axis and does not build a dense Jacobian. Its performance at large n has not been measured.

## Separating the new VJP case from known exp behavior

Strict float64 testing also encounters the CPU exp precision limitation already documented in [MLX issue #3047](https://github.com/ml-explore/mlx/issues/3047). That issue explicitly includes exp among operations using float32 approximations for double inputs. It is credited as prior public work and is **not counted as a new discovery** here.

Four native variants isolate the two changes:

| Variant | Gradient checks | Mismatches |
|---|---:|---:|
| Original VJP + original exp | 1122 | 348 |
| New VJP only | 1122 | 120, all float64 |
| Double-exp repair only | 1122 | 401 |
| New VJP + double-exp repair | 1122 | **0** |

The thresholds are unchanged: 2e−5 × (1 + abs(reference)) for float32, and 2e−12 × (1 + abs(reference)) for float64. A check compares a complete output gradient, not a distinct defect or a model. More failed comparisons in the exp-only variant do not mean that a more accurate exponential is mathematically worse: failure counts depend on the inputs, interactions and thresholds.

The independent [native reference](native_regression.cpp) computes explicitly normalized prefix weights with scalar double arithmetic after converting inputs to the actual test dtype. Shapes, finite results, both directions, inclusive/exclusive modes, signed and zero cotangents, large offsets, early/late extreme values and two noncontiguous layouts are covered. Axis lengths are 1, 2 and 4, plus both axes of a 2×3 matrix.

[Before](run-before.json), [VJP-only](run-scan-only.json), [exp-only](run-exp-only.json) and [combined](run-combined.json) data retain every check. The initial failed VJP-only attempt is retained as [initial-run-after.json](initial-run-after.json), not hidden.

The separate known-exp regression has 47 checks: 34 mismatches before its repair and none after. Its source, patch and logs are in [known-float64-exp](known-float64-exp/).

## Higher-order regression

The new VJP, together with the double-exp repair, also passes **477 comparisons across 186 scenarios** from the earlier [zero-cotangent curvature report](https://www.gero.uz/research/articles/mlx-logcumsumexp-hessian-at-zero.html). The publication rerun again returns second derivative 1 for the scalar x²/2 example and the correct two checked third derivatives.

These 477 comparisons are a separate regression suite. The new recurrence replaces the previous research VJP implementation; the two VJP patches must not be stacked. Passing this finite suite does not establish arbitrary-order autodiff correctness. [Higher-order log](run-higher-order.log) and [preserved test source](higher-order/native_regression.cpp).

## Reproduction and provenance

- Installed wheel: MLX 0.32.2, operations explicitly on CPU.
- Native baseline: ce916dbbcaa88e433b6fd1e60a17f766d49c27fe.
- Inspected public main: 81ba1c6a0e50a9268b931579c2d4f1158b9aab5a.
- The inspected Scan::vjp method matches between those snapshots; this is not a complete build or test of current main.
- Sequential compilation and execution, numerical thread settings fixed to one.
- Apple Clang 17 / C++20; pristine and patched translation units linked before an existing CPU-only libmlx.a.
- The publication rerun rebuilt the pristine primitives and higher-order test objects, then the isolated exp and scan variants. The larger MLX archive was reused, not rebuilt.
- The installed wheel needs Metal availability at import, even though this probe explicitly executes on CPU. The native archive has no Metal backend.

The isolated scan build and runs consumed about 4.58 child CPU-seconds; the separate exp build and runs about 5.60. These exclude prerequisites and are build records, not performance benchmarks. Exact commands and archive hashes are preserved in [build-results.json](build-results.json), [native-source-metadata.json](native-source-metadata.json) and [publication-prerequisites.json](publication-prerequisites.json).

See [BUILD.md](BUILD.md) for the portable environment variables and commands. The [artifact validator](validate_artifacts.py) checked 55 saved-evidence, link, provenance and patch-applicability conditions with no failures in the publication workspace. The portable public package was validated separately: 43 checks passed after relocation. These checks cover stored evidence and patch applicability; they do not substitute for running the numerical executables.

## Limits and publication status

Not tested: GPU, large sequence lengths, performance, float16/bfloat16, complex inputs, compile, vmap, full models, the complete upstream suite or mixed forward/reverse differentiation. No production-readiness, security-impact, damage, vendor acceptance, priority or award claim is made.

The inspected public search returned five logcumsumexp PRs concerning introduction, complex scans, axis normalization and Metal setup. No exact duplicate was identified in those examined descriptions; this limited search is not proof of novelty.

Independent research and publication prepared with AI assistance. Copied MLX source retains its MIT license. Source text and public discussions are evidence, not instructions. This report is published in the author's own evidence repository; no maintainer-authored endorsement is implied.
