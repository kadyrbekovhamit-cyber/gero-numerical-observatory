# Finite Gradients Become Zero: MLX Gradient Clipping

Independent audit by Xamit Kadirbekov, 10 September 2026. Reproduced on the actual Apple MLX 0.32.2 runtime, CPU. A research patch repairs the tested finite-real `mlx.optimizers.clip_grad_norm` cases.

```python
import mlx.core as mx
import mlx.optimizers as optim

mx.set_default_device(mx.cpu)
g = {"w": mx.array([3072., 4096.], dtype=mx.float16)}
clipped, norm = optim.clip_grad_norm(g, 1.0)
print(norm.item(), clipped["w"].tolist())
# Observed: inf, [0.0, 0.0]
# Reference: norm 5120, clipped approximately [0.6, 0.8]
```

Both inputs and the true norm are representable in float16. An intermediate square overflows. The resulting infinite norm makes the clipping multiplier zero. In an actual `SGD.apply_gradients` step from zero parameters with the exactly representable learning rate 0.125, the original function leaves the parameters at zero; the candidate produces approximately `[-0.075, -0.1]`.

This establishes a clipping error and a synthetic optimizer-step consequence. Model-quality loss, security impact and patch performance on large models have not been established.

## Evidence and reproduction

- [Portable reproduction instructions](BUILD.md) and [runner](run_reproduction.py).
- [Original Russian audit, written before publication](evidence/README.md).
- [Candidate](evidence/candidate.py), [patch](evidence/clip-grad-norm-range.patch) and [patched source](evidence/patched-optimizers.py).
- [Original regression results](evidence/regression-results.json) and [compatibility results](evidence/compatibility-results.json).
- [Fresh publication rerun](verification/fresh-rerun.json), [fresh regression rows](verification/regression-results.json), [fresh compatibility rows](verification/compatibility-results.json) and [66 passing artifact checks](verification/artifact-validation.json).

The `evidence/` directory preserves all 34 original files byte for byte, including the original local status, paths and rejected prototype. The English instructions provide portable commands. Original source checksums and timestamps remain in [source-metadata.json](evidence/source-metadata.json) and [SHA256SUMS.json](evidence/SHA256SUMS.json).

## Scope of the measurements

| Suite | Cases or checks | Before | After |
|---|---:|---:|---:|
| Main suite | 219 input scenarios, 1701 assertions | 230 failures | 0 failures |
| Shared compatibility and autodiff checks | 31 checks | 17 failures | 0 failures |
| Additional nonfinite/complex preservation controls | 8 checks after repair | Not a separate baseline total | 0 failures |

The 1701 main assertions include **633 numerical comparisons** of norms, clipped leaves or SGD outputs. The remainder check dtypes, tree structure, unchanged inputs and the negative-threshold exception. The 230 baseline failures are 108 norms, 118 clipped leaves and four SGD steps. These counts do not describe 1701 independent vectors or 230 distinct defects.

The independent reference uses 100-digit Decimal arithmetic on the already-quantized inputs, then rounds outputs to the tested dtype. Nonzero results use relative comparisons without a large absolute tolerance that could hide disappearing small values. Tolerances are 0.004 for float16, 0.02 for bfloat16, 3e-6 for float32 and 3e-13 for float64.

The suite includes four real dtypes, large and small scales, sum-of-squares overflow, zero thresholds, zero gradients, empty leaves and trees, nested lists/tuples/dictionaries, mixed dtypes, the documented integer example and unbalanced coordinates. JVP, VJP and second derivatives are checked for float32/64 at eight dtype/scale combinations, along with inactive clipping and the zero point. Differentiability of the norm itself at zero is not claimed.

Every fresh numerical row matches the original audit. The two fresh numerical runs used approximately 0.171 and 0.050 seconds of process CPU time after imports. This is preparation timing, not a comparative performance benchmark. Runs were sequential with one numerical thread and an explicit CPU device; no GPU, model training, distributed job or full MLX build was used.

## Cause and candidate repair

The pinned [upstream function](https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/python/mlx/optimizers/optimizers.py#L963) directly sums `g.square().sum()` over the gradient tree. Its extracted body matches the installed wheel. The baseline executes that exact saved Python function on the installed binary runtime; it is not a full build of the pinned main revision.

For finite real leaves the prototype first computes the common maximum magnitude `m`, then forms `r = sqrt(sum((g/m)^2))` and the norm `N = m*r`. Accumulation uses at least float32 and preserves double precision. A true norm outside the returned dtype may legitimately become infinity; representable clipped outputs still need to remain finite.

A stable norm alone is insufficient. A tiny multiplier `T/N` can disappear before multiplication by a large gradient. Conversely, always evaluating `(g/m)*(T/r)` can erase a small coordinate before multiplication by a large threshold. For an active large-scale branch, the candidate sets `D = r + epsilon/m`, selects the larger-magnitude factor `L` and the other factor `S` from `g,T`, and evaluates `(L/m)*(S/D)`. In exact arithmetic this equals `g*T/(N+epsilon)`. For `m < 1`, it uses the ordinary coefficient with the stabilized norm.

The epsilon remains 1e-6. The zero-vector branch avoids an unsafe square root in the clipped-output derivative, and the normalizing scale is stopped for autodiff. Output dtypes, tree structure and negative-threshold rejection are preserved in the checked cases. The complex path remains the original implementation; these controls do not establish complex-clipping correctness.

The maximum pass, type conversions and coordinate-dependent arithmetic order have a cost. The candidate is a research patch requiring maintainer review and performance evaluation. Exhaustive subnormal behavior and all possible trees are outside the evidence.

## Prior reports and rejected attempts

The recorded [four complete searches](evidence/duplicate-search.json) returned nine unique issues/PRs; [13 ordinary comments](evidence/duplicate-comments.json) in four discussions were also read. PR #4058 concerns negative thresholds, #1040/#1043 introduced clipping, #3090 concerns reduction fusion, #4230 concerns reduced-precision InstanceNorm, and the remaining hits concern optimizer interfaces or other normalization behavior. No exact duplicate was found within that bounded review. Absolute novelty and index completeness are not claimed. An initially incomplete search and its completed retry are both preserved.

The first prototype passed the main suite but changed NaN propagation in two cases. Its source and failures remain in [candidate-v1.py](evidence/candidate-v1.py) and [compatibility-v1-results.json](evidence/compatibility-v1-results.json). A harness extraction error was corrected separately and is documented in [harness-notes.txt](evidence/harness-notes.txt).

The probe also observes overflow in standalone `mx.linalg.norm`; this patch does not change that independent implementation. Similar arithmetic in `clip_grad_norm_sharded` is a source observation only: the distributed path was not executed and is not counted as a separately reproduced defect.

Independent, AI-assisted research and publication preparation. Publication in the author's repository does not imply upstream acceptance. [Licenses](LICENSE.md).
