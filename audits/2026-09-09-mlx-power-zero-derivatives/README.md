# A Constant, a Polynomial, a NaN: MLX power Derivatives at Zero

9 September 2026 · GERO Research · Xamit Kadirbekov

**Locally reproduced numerical defect in MLX 0.32.2.** For a fixed zero exponent, `power(x,0)` returns the constant one, but its base derivative at zero is NaN instead of zero. Repeated differentiation exposes the same mechanism in positive integer powers. A polynomial expressed through an array of exponents can therefore acquire a NaN first gradient.

Fresh publication reruns on the official Python wheel and actual C++ code confirm the result. A proposed local patch passes **41 scenarios and 396 checks**; the baseline passes 29 scenarios and fails 42 checks. This is a remaining case in a known Power VJP error family, not a claim of established novelty. Full-model impact has not been measured.

## Minimal reproduction

```python
import mlx.core as mx
mx.set_default_device(mx.cpu)
x = mx.array(0.0)

print(mx.grad(lambda t: mx.power(t, 0.0))(x))
# MLX 0.32.2: nan; expected: 0

print(mx.grad(mx.grad(lambda t: mx.power(t, 1.0)))(x))
# MLX 0.32.2: nan; expected: 0
```

The full [Python probe](probe.py) and its freshly regenerated [results](probe-results.json) include a direct-polynomial control, actual-forward finite differences and mixed derivatives at positive bases.

| Fixed function | Derivative checked at zero | Original | Expected |
|---|---:|---:|---:|
| `power(x,0)` | first | NaN | 0 |
| `power(x,1)` | second | NaN | 0 |
| `power(x,2)` | third | NaN | 0 |
| `power(x,3)` | fourth | NaN | 0 |

The first and second derivatives of the square at zero are currently correct: 0 and 2. The historical square-first-derivative case fixed in [PR #505](https://github.com/ml-explore/mlx/pull/505) is not being presented as still broken.

## Preconditions and mathematical reference

The central claim concerns real, fixed nonnegative integer exponents, finite CPU float32 inputs and differentiation with respect to the base. These fixed-exponent functions are polynomials under MLX's forward convention for exponent zero. No claim of joint differentiability with respect to both base and exponent at `(0,0)` is made.

For `f(x)=x^n`, the kth derivative is the falling factorial `n!/(n-k)!` times `x^(n-k)` when `k≤n`, and identically zero when `k>n`. Thus the derivative immediately beyond the polynomial's degree must be zero, including at the origin. There is no subgradient convention to choose.

## An ordinary polynomial receives a NaN first gradient

```python
exponents = mx.array([0., 1., 2.])
coefficients = mx.array([1., 2., 3.])
model = lambda x: (coefficients * mx.power(x, exponents)).sum()
loss = lambda x: 0.5 * (model(x) - 4.) ** 2
```

This represents `model(x)=1+2x+3x²`. At zero, the exact model derivative is 2, the loss is 4.5 and its derivative is −6. Coordinate central differences of the actual model forward with step `1/1024` give exactly 2. Writing `1+2*x+3*x*x` directly in the same wheel also returns the correct derivative.

| Quantity at `x=0` | Original wheel / C++ | Patched C++ |
|---|---:|---:|
| Model value | 1 | 1 |
| Model derivative | NaN | 2 |
| Loss value | 4.5 | 4.5 |
| Loss derivative | NaN | −6 |
| Loss after `x -= 0.125 * gradient` | NaN | 0.017578125 |

The corrected update is `x=0.75`. Then the model is `4.1875`, so the loss is `0.5×0.1875²=0.017578125`. The C++ harness reevaluates the actual forward after the update; this is an executed measurement, not only a symbolic prediction. This small synthetic example does not establish an effect on any complete training pipeline.

## Root cause and narrow repair

In [pinned `Power::vjp`, around line 3452](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp#L3452), the base derivative contains `b * power(a,b-1)`. At the real point `a=0,b=0`, this evaluates `0×Inf` and produces NaN. Differentiating positive integer powers repeatedly eventually reaches the same helper case.

The [proposed patch](power-zero-derivatives.patch) uses a safe auxiliary base only in that real zero/zero case:

```text
safe_base = 1 if a == 0 and b == 0 else a
base_derivative = b * power(safe_base, b - 1)
```

The exponent remains a multiplier. The forward operation is unchanged. The real-path guard avoids creating the singular intermediate; it is not merely masking an already computed NaN. `Power::jvp` delegates to the VJP helper and receives the same correction. The complex path is left unchanged.

A broad replacement whenever `b=0` would risk mixed derivatives. For positive `a`, `∂²(a^b)/(∂b∂a)` at `b=0` equals `1/a`, not 1. The proposed guard requires `a=0` as well, so it leaves the positive-base region unchanged. Both mixed second derivatives, pure second derivatives and one mixed third derivative were checked at positive bases, including exponent zero; these controls pass before and after the patch.

This is a targeted repair, not universal handling of power singularities or overflow. Small nonzero bases, extreme exponents and the full floating-point range were not exhaustively tested. The guard adds elementwise work; large-array performance has not been benchmarked.

## A separate test-discovery problem

At the pinned revision, `TestAutograd` in [test_autograd.py](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/python/tests/test_autograd.py#L699) defines `test_power_grad` twice, at lines 699 and 734. The later Python definition replaces the earlier method in the class namespace. The earlier method contains three historical assertions involving zero bases and an ordinary square.

The [AST audit](test-name-audit.json) records both definitions. The patch renames the earlier method to `test_power_grad_base`, preserves the later method and adds two Python regression methods. Their syntax and name uniqueness were checked. **These Python additions were not run against a newly built patched wheel.** Equivalent numerical cases were executed in the native C++ harness. The test-name collision is a separate coverage issue, not proof that it caused this defect to escape review.

## Native results and test coverage

| Implementation | Scenarios passed | Checks | Failed checks |
|---|---:|---:|---:|
| Original C++ | 29/41 | 396 | 42 |
| Patched C++ | 41/41 | 396 | 0 |

The [C++ harness](native_regression.cpp) includes:

- 25 fixed-integer cases: exponents 0–4 at `+0`, `−0`, `−2`, `0.5` and `2`; forward values, derivatives through order five, JVP/VJP, zero cotangents and actual-forward first-derivative finite differences.
- 12 positive-base cases with analytic references for mixed derivatives, including exponent zero.
- Four additional cases: `x^1.5` first-derivative control at zero, the polynomial loss/update, and two broadcasting layouts.

[Original execution log](run-before.log) · [Patched execution log](run-after.log) · [Actual build commands and exit codes](build-results.json).

The 42 failures concern one numerical mechanism; they are not 42 independent discoveries. Passing targeted tests does not prove correctness for all states or substitute for the complete MLX suite.

## Source pins, compiler and reproducibility

- Inspected MLX main: `24c699ecee2f7c8b2040de8da1c8382c8bcf31c7`.
- Compatible native base: `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`.
- `Power::vjp` and `Power::jvp` were re-extracted from the base and confirmed byte-identical to the inspected main.
- Apple clang 17.0.0, C++20, macOS arm64, CPU float32; computational thread environment variables set to one.
- Original/patched translation units were linked ahead of a reused CPU-only MLX archive, SHA-256 `7826e14e2d1ee526a16352829d460ae39c048e331c339be0b6f8d67d74e68a06`.

**This is a partial native rebuild on a compatible base, not a clean full build of current main.** All 32 supplied input artifact hashes matched before copying. Publication verification repeated the wheel probe, native compilation and both native runs. Artifact validation passed 52 checks, including applying the patch to disposable pinned source copies.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python probe.py

export MLX_SOURCE_ROOT=/absolute/path/to/mlx-at-ce916db
export MLX_CPU_BUILD=/absolute/path/to/cpu-build
python3 build_and_test.py
```

See [BUILD.md](BUILD.md) for the archive setup and build boundary. The original checkout and other audit patches were not edited.

## Prior work and limits

[Issue #504](https://github.com/ml-explore/mlx/issues/504) and [PR #505](https://github.com/ml-explore/mlx/pull/505), merged in January 2024, addressed NaN in the first derivative of a square at zero. The historical change removed division by the base and introduced the present derivative formula. This report identifies a remaining fixed-zero-exponent case and its higher-order consequences.

Four recorded repository searches found no exact matching report among the returned results. That is a limited observation, not proof of priority. Search results do not cover every comment, commit or external discussion. Maintainer acceptance is not established.

GPU, FP16/BF16, compiled execution, general complex-power derivatives, exponent derivatives at `(0,0)`, singular fractional powers and extreme scales were not validated. No device harm, security impact, reward eligibility or complete-model training effect is asserted.

Prepared with AI assistance. The measurements come from actual local executions, supported by algebra and finite differences. Independent GERO Research; no Apple endorsement. Included MLX source retains its [MIT license](MLX-LICENSE.txt). [Source ledger](SOURCES.md) · [File checksums](SHA256SUMS.json).
