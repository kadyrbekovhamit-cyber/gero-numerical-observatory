# MLX logcumsumexp loses curvature at a zero incoming gradient

10 September 2026 · Xamit Kadirbekov · Independent GERO Research

**Locally reproduced in MLX 0.32.2.** A smooth scalar function with exact second derivative 1 returns 0. A research C++ prototype passes all 477 comparisons across 186 scenarios; the original fails 189 comparisons in 59 scenarios. These are manifestations of one higher-order differentiation defect. The tests do not establish GPU behavior, model-level impact, novelty or maintainer acceptance.

## A singleton is enough

For a one-element inclusive scan, `logcumsumexp([x]) = [x]`. Consequently:

```text
f(x) = 0.5 × sum(logcumsumexp([x])²) = x² / 2
f(0) = 0;  f′(0) = 0;  f″(0) = 1.
```

The actual installed CPU package returns the correct value and first derivative, but the second derivative is **0**. Central second differences of the actual forward function, with step `1/1024`, return **1**. The local C++ prototype also returns **1**. This example has no nonsmooth point, singularity, overflow or disputed boundary convention.

```python
import mlx.core as mx
mx.set_default_device(mx.cpu)
f = lambda x: 0.5 * mx.square(mx.logcumsumexp(x.reshape(1))).sum()
print(mx.grad(mx.grad(f))(mx.array(0.0)).item())  # original: 0.0; exact: 1.0
```

See [executable reproduction](reproduce.py) and [saved wheel results](wheel-reproduction.json).

## The two-element Hessian

Let `F(x) = logcumsumexp(x)`, `x₀ = (0,0)`, and hold `t = F(x₀)` fixed. For `L(x) = 0.5 ||F(x) − t||²`, the residual and first gradient vanish at `x₀`. The curvature does not:

```text
J = [[1,   0  ],       Hessian(L) = JᵀJ = [[1.25, 0.25],
     [0.5, 0.5]]                          [0.25, 0.25]].
```

MLX returns a zero Hessian. Central differences of the actual first gradient give approximately `[[1.25000036, 0.25], [0.25, 0.25000006]]`. Differentiating the VJP directly with respect to a zero cotangent in a different graph produces NaNs instead of `Jᵀ`. Both observations concern the same underlying representation. The checked first derivatives remain correct; this report concerns subsequent derivatives through them.

## Cause: a linear map represented by sign branches

For fixed finite `x`, a vector–Jacobian product must be linear in the incoming cotangent `g`:

```text
B(x,g) = J(x)ᵀg
D_g B(x,0)[h] = J(x)ᵀh.
```

The `Scan::LogAddExp` branch of [Scan::vjp](https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/mlx/primitives.cpp#L4278) separates positive and negative cotangents, takes logarithms of their magnitudes, and subtracts two signed scan contributions. At zero, both sign branches choose constants. The VJP value can be correct while differentiation of that implementation loses the required dependence on `g`. The observed results are zero or NaN, depending on the differentiated graph.

## Research prototype and its mathematical rule

Define `A(x,v) = J(x)v` and `B(x,v) = J(x)ᵀv`. The prototype retains signed logarithmic scans for their values and supplies explicit reverse rules through MLX's existing C++ `custom_vjp` mechanism. For incoming cotangent `h`:

```text
Backward of B(x,v):
  grad_v = A(x,h)
  grad_x = h * B(x,v) − B(x, v * A(x,h))

Backward of A(x,v):
  grad_v = B(x,h)
  grad_x = v * B(x,h) − B(x, h * A(x,v)).
```

These identities follow from `Hessian(logsumexp) = diag(p) − ppᵀ` for each nonempty prefix, where `p` is that prefix's normalized exponential vector. They remain valid when `v = 0`. Using the same operators inside their explicit reverse rules preserves the checked further reverse derivatives. A dense quadratic Jacobian is used only in the small independent oracle, not in the proposed execution algorithm.

For exclusive scans, the empty prefix yields `−∞`. The tests exclude those outputs from the smooth loss with a finite-output mask, and assign their Jacobian rows zero. They do not claim differentiability of an infinite-valued loss.

See the [prototype patch](logcumsumexp-higher-order-prototype.patch), [helper implementation](logscan_product.cpp.inc), and [native test](native_regression.cpp). This is **not a production-qualified repair**: it adds transformations and scan work, and still requires broader review and performance testing.

## What was actually executed

| Implementation | Scenarios | Comparisons | Mismatches | Failed scenarios |
|---|---:|---:|---:|---:|
| Original C++ | 186 | 477 | 189 | 59 |
| Research C++ prototype | 186 | 477 | 0 | 0 |

Both variants execute all comparisons. All 81 first-gradient comparisons pass in both. The independent oracle calculates prefix-normalized exponentials, Jacobians, gradients and Hessians with scalar double-precision loops; it does not call the VJP under investigation. Comparison tolerance is `2e−4 × (1 + abs(reference))`, with shape and finiteness checks.

The 81 base configurations cover lengths 1, 2 and 4; shape `2×3` along either axis; both directions; inclusive/exclusive modes; zero, partially zero and nonzero residuals; an input span of 40; and one noncontiguous input. Additional comparisons differentiate the VJP with respect to its cotangent and check two small cubic objectives. Their third derivatives are NaN before repair, and respectively 1 and 2 after repair, matching exact formulas.

The [before log](run-before.log), [after log](run-after.log), and [build records](build-results.json) preserve actual results. The publication rerun compiled the changed translation unit and test sequentially, linked each variant before an existing CPU archive, and used one computation thread. It did not rebuild the complete library. Recorded compilation, linking and execution consumed approximately 4.69 child CPU-seconds; this is not a runtime benchmark.

## Source and toolchain provenance

- Installed package: `mlx==0.32.2`, CPU.
- Native baseline: `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`.
- Main inspected on 10 September 2026: `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`.
- The complete `Scan::vjp` method matches between those two snapshots. Testing a baseline with that matching method is not a full build of current main.
- Apple Clang 17, C++20; exact compiler output in [compiler-version.txt](compiler-version.txt).
- Existing `libmlx.a` SHA-256: `7826e14e2d1ee526a16352829d460ae39c048e331c339be0b6f8d67d74e68a06`.
- [Source metadata](source-metadata.json) records inspected files and archive hashes. [SHA256SUMS.json](SHA256SUMS.json) covers this evidence package.

## Reproduction

Use an Apple-silicon environment compatible with the recorded package. In a separate environment, install `mlx==0.32.2` and run `python reproduce.py`. It records the observed defect rather than treating it as an unexpected script failure.

For native execution, follow [BUILD.md](BUILD.md), set `MLX_SOURCE_ROOT` to the pinned checkout and `MLX_CPU_BUILD` to the CPU build directory, then run `python3 build_and_test.py`. The runner expects exit 1 from the unmodified regression executable and exit 0 after repair. It overwrites its local artifacts. The provided fresh-archive recipe is a portability aid, not a claim that it was executed for this publication.

`python3 validate_artifacts.py` checks consistency of saved evidence and links; it does not rerun numerical tests.

## Related GatherQMM validation

An [attached follow-up](gather-qmm-followup/README.md) tests mixed derivatives against the earlier GatherQMM repair. The original has four mismatches in 24 checks; the previously prepared repair passes all 24. This extends the validation of that earlier defect rather than establishing another new finding. For the test's `transpose=False` group layout, the correct mixed derivative is 32, not 1; the initially mistaken reference was corrected before the final comparison.

## Public history and limitations

Fresh GitHub searches for `logcumsumexp`, `logcumsumexp hessian`, and `logcumsumexp cotangent` returned five, zero and zero results respectively. The broad results concern operation introduction (#2069), complex scans (#2094 and #4272), axis normalization (#4383), and Metal encoder refactoring (#4154). The examined descriptions do not establish a duplicate of the zero-cotangent higher-derivative case. [Search responses](publication-duplicate-search.json) are retained; this is a limited search, not proof of priority.

Unexamined areas include GPU, float16/bfloat16/complex, long-sequence performance, full models, the complete upstream suite, `compile`, `vmap`, and mixed forward/reverse differentiation. A separately observed accuracy issue at a large common input offset is not resolved by this prototype. Successful finite tests do not prove correctness for all inputs.

Independent, AI-assisted research, writing, test execution and publication. No Apple endorsement, security impact, acceptance or reward is claimed. Copied MLX code retains its MIT license. Publication channels and upstream submission status are tracked separately from the technical findings.
