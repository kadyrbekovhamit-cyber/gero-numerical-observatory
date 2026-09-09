# Orthogonal Is Not Symmetric: MLX Hadamard Gradients

9 September 2026 · GERO Research · Xamit Kadirbekov

**Locally reproduced reverse-derivative defect in MLX 0.32.2.** The Hadamard transform uses nonsymmetric matrices for factors 20 and 28, but its VJP applies the forward operator instead of its transpose. A simple quadratic loss then receives a gradient pointing partly in the wrong direction.

Fresh publication reruns confirm **64/64 scenarios and 624/624 checks passing after the proposed local C++ patch**, versus 34/64 scenarios and 186 failed checks before it. An additional eight-check experiment executes the actual loss update at four sizes. Novelty, maintainer acceptance and full-model impact remain unestablished.

## Minimal example

```python
import mlx.core as mx
mx.set_default_device(mx.cpu)
x = mx.array([1.] + [0.] * 19)
loss = lambda x: 0.5 * mx.sum(mx.square(mx.hadamard_transform(x)))
g = mx.grad(loss)(x)
print(g[0].item())              # -0.5; expected approximately 1
print(loss(x - 0.125*g).item())  # 0.57031256; initial loss 0.5
```

The public [Python probe](probe.py) selects CPU and sets computational thread environment variables to one before importing MLX. The [recorded results](probe-results.json) include a direct-matrix adjoint reference, actual-forward finite differences and symmetric controls.

## Why the correct gradient is known

Let `H` be the real Hadamard matrix and `Q=H/√N`. The source tables contain only +1 and −1. Recomputing their products in exact integer arithmetic establishes `HᵀH=N·I`. Therefore, in exact arithmetic:

```text
QᵀQ = I
L(x) = 0.5 × ||Qx||² = 0.5 × ||x||²
∇L(x) = QᵀQx = x
∇²L(x) = I
```

For this real linear operation, the forward directional derivative is `Qv`, while the reverse derivative must be `Qᵀc`. Orthogonality alone does not imply `Q=Qᵀ`. The [exact certificate](exact-matrix-certificate.json) records 0 asymmetric entries for H12, 300 for H20 and 588 for H28. Powers of two also provide symmetric controls.

The central domain is finite real inputs with a supported positive last-axis size `N=m·2ᵏ`, where `m∈{1,12,20,28}`, and a fixed real scale. The [official operation contract](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.hadamard_transform.html) describes supported sizes and normalization. Tests use CPU and the dtypes listed below. The exact matrix argument is separated from finite-precision implementation tolerances.

For arbitrary scale `s`, the corresponding energy gradient is `N·s²·x`. The native tests use this reference for normalized, positive, negative and zero scales. This is a smooth quadratic objective with no tie or subgradient convention to choose.

## A measured optimization step raises the loss

For `x=e₀` and learning rate `η=1/8`, the original implementation returns approximately −0.5 for the first gradient component instead of 1. The ideal corrected step is `(1−η)e₀`, yielding loss `0.5×(7/8)²=0.3828125`.

The original audit gave that corrected loss analytically. **For publication, a separate native program also executed the actual forward after both the original and patched updates.** Its [source](e0_step.cpp), [runner](run_e0.py) and logs preserve the distinction.

| N | Original first gradient | Original loss after step | Patched first gradient | Patched loss after step |
|---:|---:|---:|---:|---:|
| 20 | −0.5 | 0.5703125596 | 1 | 0.3828125000 |
| 28 | −0.5 | 0.5703125000 | 0.9999999404 | 0.3828124702 |
| 40 | −0.4999999702 | 0.5703125596 | 1.000000119 | 0.3828125000 |
| 56 | −0.5000000596 | 0.5703125000 | 0.9999996424 | 0.3828125298 |

Initial measured losses are approximately 0.5; their small deviations and the patched values above are FP32 rounding. The sign reversal and loss increase are much larger than that rounding. First-coordinate finite differences of the official wheel's actual forward give approximately 1.

[Original update log](run-e0-before.log) · [Patched update log](run-e0-after.log). The supplemental selection passes 0/8 checks before and 8/8 after. These eight checks are separate from the 624-check primary suite. No real LLM or complete training pipeline was studied.

## Source defect and proposed repair

At [pinned `Hadamard::vjp`, around line 6295](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp#L6295), the reverse rule delegates to JVP:

```cpp
return jvp(primals, cotangents, argnums);
```

That is correct for the symmetric families. For factors 20 and 28 it applies `sH` instead of `sHᵀ`. With normalized scale, the erroneous energy gradient is `Q²x` instead of `QᵀQx`.

The [local C++ patch](hadamard-adjoint.patch) uses the source factorization `H_N=H_m⊗H_(2ᵏ)`. For factors 20 and 28, it applies the transposed small factor through a matrix multiplication and reuses the existing symmetric power-of-two transform. Scale is applied once. Factors 1 and 12 retain their existing VJP path.

The additional constant matrix is at most 28×28; the proposed implementation does not materialize an N×N matrix. It changes only the reverse rule plus the required header include. Forward, JVP and primitive serialization state remain unchanged.

This is a correctness-oriented implementation. Large-array performance, memory behavior and GPU execution were not benchmarked. Maintainer review, a complete build and the upstream suite remain necessary before production acceptance.

## Actual validation results

| Selection | Original | Patched |
|---|---:|---:|
| Primary scenarios passed | 34/64 | 64/64 |
| Primary checks passed | 438/624 | 624/624 |
| Primary failed checks | 186 | 0 |
| Supplemental e₀ update checks passed | 0/8 | 8/8 |

The [native harness](native_regression.cpp) covers forward, JVP and VJP against an explicit matrix product; cotangent linearity and zero cotangents; quadratic-energy gradients; Hessian-vector products through forward and reverse differentiation; zero third derivatives; selected actual-forward finite differences; `grad(vmap)`, `vmap(grad)` and a last-axis batch layout.

- FP32 single-vector sizes: 1, 2, 4, 12, 24, 48, 20, 40, 80, 28, 56 and 112, each with normalized, +0.25, −0.125 and zero scale.
- FP32 batches of two: sizes 4, 12, 20, 28, 40 and 56.
- FP16 and BF16 batches of two: sizes 12, 20, 28, 40 and 56, using moderate representable inputs and normalized scale.

Low-precision checks use broader tolerances. They do not establish numerical stability over the complete FP16/BF16 domain. The exact tolerances and deterministic vectors are in the harness. The 186 failures are manifestations of one reverse-rule defect, not 186 independent findings.

[Primary baseline log](run-before.log) · [Primary patched log](run-after.log) · [Build commands and exit codes](build-results.json).

## Versions and reproduction boundary

- Official wheel: MLX 0.32.2, CPU.
- Inspected main: `24c699ecee2f7c8b2040de8da1c8382c8bcf31c7`.
- Compatible native base: `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`.
- Hadamard VJP/JVP bodies and constant matrix tables match between the inspected main and compatible base.
- Apple clang 17.0.0, C++20, macOS arm64. Test and original/patched translation units compiled sequentially with `-O0` and linked ahead of an existing CPU-only `libmlx.a`.
- Reused archive SHA-256: `7826e14e2d1ee526a16352829d460ae39c048e331c339be0b6f8d67d74e68a06`.

**This is a partial native rebuild on a compatible base, not a clean full build of main.** The official wheel and original checkout were not modified. Computational thread environment variables were set to one; no GPU numerical execution was used for this report.

All 33 supplied artifact hashes matched before copying. Publication verification repeated the wheel probe, primary native compilation/execution and exact integer certificate checks. The Hadamard-only artifact validator passed 45 checks. Related FFT/pad validation from the supplied validator is outside this publication and was excluded from that 45-check selection.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python probe.py

export MLX_SOURCE_ROOT=/absolute/path/to/mlx-at-ce916db
export MLX_CPU_BUILD=/absolute/path/to/cpu-build
python3 build_and_test.py
python3 run_e0.py
python3 validate_artifacts.py
```

[BUILD.md](BUILD.md) describes the required archive layout and the unexecuted fresh-build setup recipe. Source and result hashes are recorded in [source-metadata.json](source-metadata.json) and [SHA256SUMS.json](SHA256SUMS.json).

## Prior work and limits

The original [PR #1249](https://github.com/ml-explore/mlx/pull/1249) introduced this transform and its derivative rules. The existing [gradient test](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/python/tests/test_ops.py#L3940) exercises powers of two, which are symmetric controls.

[Issue #4049](https://github.com/ml-explore/mlx/issues/4049) and [PR #4054](https://github.com/ml-explore/mlx/pull/4054) concern Metal kernel launch behavior for the non-power-of-two factors. This report's CPU reverse-derivative failure is a different mechanism and also reproduces at sizes 40 and 56.

Four recorded repository searches returned 1, 2, 6 and 17 results for Hadamard-related queries. No exact matching adjoint report was identified in those results. The search is not exhaustive and does not establish priority. [Public-history review](SOURCES.md).

Only the listed sizes, scales, layouts and finite input samples were executed. Larger shapes, Metal/CUDA, compiled graphs and complete-model effects remain untested. Passing a targeted suite is not proof over all possible states. This report does not present previously known FFT or pad observations as new findings.

Prepared with AI assistance. Numerical evidence is actual execution, independent algebra, exact integer products and finite differences. Independent GERO Research; no Apple endorsement. Included MLX source retains its [MIT license](MLX-LICENSE.txt).
