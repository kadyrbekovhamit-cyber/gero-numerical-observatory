# MLX Muon: convolution shape changes the update scale

**Locally reproduced implementation defect, 9 September 2026.** Equivalent
1×1 convolution and linear weights have equal outputs, losses and gradients,
but the convolution's Muon step is half the size in the measured example.
A minimal reorder repairs all 31 focused regression tests; the original
passes 12 and fails 19. These are manifestations of **one defect**.

This is a technical report for code correction. Maintainer confirmation,
novelty, upstream acceptance and full-model effects are not established.

[GERO article and short video](https://www.gero.uz/research/articles/mlx-muon-convolution-scaling.html)
· [Patch](muon-shape-scaling.patch) · [Tests](regression.py)
· [Native results](probe-results.json) · [Sources](SOURCES.md)

## Version and execution boundary

| Component | Verified version / scope |
|---|---|
| Released MLX | Official `mlx==0.32.2` wheel; three tensor/matrix comparisons |
| MLX Python source | `24c699ecee2f7c8b2040de8da1c8382c8bcf31c7` |
| Native core for pinned Python | Wheel 0.32.2, **not** a new native build of the commit |
| Environment | Python 3.12.14; macOS 15.5 arm64; CPU; FP32 |
| Author's Muon reference | `f98f1cacc0263b04290753e32be8d498c1efc806`; source review only, not executed |
| Test tolerance | `atol=2e-6`, `rtol=2e-5` |

The pinned commit was still the upstream main HEAD when checked on
9 September 2026 before publication. The original audit's 20 recorded
checksums matched before packaging. `probe.py` and `regression.py` were
rerun sequentially for publication. Code snapshots and scripts are copied
byte-for-byte; log path prefixes are reduced to filenames in this public
package. See [validation](validation.json) and [publication checks](publication-checks.json).

## Expected property and assumptions

For positive tensor dimensions, finite FP32 parameters and gradients,
identical optimizer options and consistent momentum states, represent the
same update as either `(d0, d1, ..., dk)` or `(d0, product(d1:))`.
After reshaping back, the parameter update should agree within the declared
tolerance. The tested domain is eight small shapes and three optimizer
configurations, not every possible tensor or floating-point state.

MLX flattens trailing dimensions for Newton–Schulz. The scale must be based
on that matrix, consistently with the author's reference implementation:

```text
rows = d0
cols = product(d1:)
scale = sqrt(max(1, rows / cols))
```

At [optimizers.py lines 940–962](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/python/mlx/optimizers/optimizers.py#L940),
`Muon.apply_single` currently restores the original shape first:

```python
update = self._zeropower_via_newtonschulz5(update, steps=self.ns_steps)
if reshape_needed:
    update = mx.reshape(update, original_shape)
lr *= max(1, update.shape[-2] / update.shape[-1]) ** 0.5
```

For an MLX Conv2d weight `(Cout, H, W, Cin)`, this uses `W/Cin` rather than
`Cout/(H*W*Cin)`. Holding the flattened Newton–Schulz update fixed gives:

```text
observed / intended scale =
sqrt(max(1, W/Cin) / max(1, Cout/(H*W*Cin)))

(8,1,1,2): actual scale = sqrt(max(1,1/2)) = 1
           matrix scale = sqrt(max(1,8/2)) = 2
           ratio = 1/2
```

This algebraic prediction is distinct from the native measurements below.
For other shapes it can also over-scale the step.

## Deterministic native measurements

Zero initial parameters; gradient `(arange(N)+1)/16`; learning rate 0.01;
momentum 0; Nesterov false; weight decay 0; default five Newton–Schulz steps.
Both the released optimizer and pinned Python module reproduce:

| Tensor shape | Matrix shape | Tensor step norm | Matrix step norm | Ratio |
|---|---|---:|---:|---:|
| `(8,1,1,2)` | `(8,2)` | 0.011393075 | 0.022786150 | 0.5 |
| `(1,1,8,2)` | `(1,16)` | 0.013928725 | 0.006964363 | 2.0 |
| `(2,2,2,2)` | `(2,8)` | 0.009926165 | 0.009926165 | 1.0 |

The last row is a negative control. Full input and update vectors are in
[probe-results.json](probe-results.json).

The real forward/gradient check uses the pinned Python optimizer:

```python
x = mx.array([[[[1., -.5], [.25, .75]], [[-1., .5], [.5, 1.]]]])
w = ((mx.arange(16, dtype=mx.float32) + 1) / 32).reshape(8, 2)
linear = x @ w.T
conv = mx.conv2d(x, w.reshape(8, 1, 1, 2))
# Loss: mean(square(output)); compute gradients using mx.grad.
# Apply separate Muon instances with the same options above.
```

| Quantity | Measured result |
|---|---:|
| Maximum output difference | 0 |
| Both losses | 0.0851593017578125 |
| Maximum gradient difference | 0 |
| Linear step norm | 0.026587212458252907 |
| Convolution step norm | 0.013293605297803879 |
| Maximum updated-weight difference | 0.005442783236503601 |

The step ratio is one half within FP32 rounding. Model accuracy or
training convergence was not measured.

## Minimal repair and before/after results

[The patch](muon-shape-scaling.patch) moves the existing scale computation
above the reshape. No public API or Newton–Schulz, momentum or weight-decay
calculation is changed. It applies cleanly; its output is byte-identical to
the tested candidate; the AST outside `Muon.apply_single` is unchanged.
Only local source copies were patched.

| Focused test selection | Original | Patched |
|---|---:|---:|
| Passed | 12 / 31 | 31 / 31 |
| Assertion failures | 19 | 0 |
| Execution errors | 0 | 0 |

Coverage in [regression.py](regression.py):

- 24 tensor/matrix comparisons: eight shapes × three combinations of
  momentum, Nesterov and weight decay, each checking three updates,
  momentum state, dtype, shape and unchanged inputs.
- Four Python `math` scale checks with `ns_steps=0`, deliberately isolating
  scale. Other comparisons use the default five iterations.
- Two real 1×1 convolution/linear checks with momentum 0 and 0.95.
- The unchanged body of upstream `test_muon` and helper `tree_equal` in a
  CPU `unittest.TestCase`. This is **not** the complete upstream test suite
  or its `MLXTestCase` fixture.

The existing upstream convolution test uses shape `(16,8,3,3)`, where
both scale rules yield 1, and does not assert the update's numerical scale.
Its success therefore does not cover the failing condition.

[Before log](before.log) · [After log](after.log) · [Machine-readable summary](regression-results.json)

## Reproduction

On an MLX-compatible Mac with Python 3.12:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python probe.py
.venv/bin/python regression.py
```

Run sequentially. The scripts select `mx.cpu`, set OMP/OpenBLAS/MKL/vecLib/
NumExpr thread limits to 1 before import and set a 30-second CPU-time limit
per process. They use no worker pool; this is not physical-core affinity.
They overwrite local result JSON and before/after logs. On macOS, the MLX
wheel may initialize Metal at import even though calculations use CPU.

`regression.py` runs and records both variants, including the intentionally
failing baseline. A completed script's exit status alone is not proof that
every test passed: inspect `regression-results.json`. Expected counts are
31/19/0 tests/failures/errors before, and 31/0/0 after.

## Limits and publication status

No full-model training, GPU execution, FP16/BF16, distributed optimization,
performance benchmark or native rebuild of main was performed. A passing
finite test selection does not prove universal correctness. The author's
PyTorch code was read as an algorithm source, not numerically cross-executed.

A scoped public-history search found no exact duplicate. That is not proof
of novelty. The technical discrepancy and patch are locally reproduced;
upstream acceptance and consequences for Apple products are unestablished.

Prepared with AI assistance. Measurements are actual local executions and
the algebraic check is independently specified; neither is a third-party
audit. Included MLX source retains its MIT license. Independent reproduction
and scoped numerical-correctness reviews are welcome through GERO.

#MachineLearning #Muon #MLX #NumericalComputing #SoftwareTesting
