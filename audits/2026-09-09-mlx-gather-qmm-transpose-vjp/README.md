# A Missing Transpose in MLX Quantized Gradients

9 September 2026 · Xamit Kadirbekov · Independent GERO Research

**Locally reproduced numerical-correctness defect in MLX 0.32.2.** With `transpose=False`, affine `gather_qmm` computes scale and bias gradients in the wrong physical weight orientation. A square matrix can silently receive incorrect values; a rectangular matrix can receive an incorrectly shaped bias gradient or raise an exception while computing the scale gradient.

A fresh local C++ rerun passes **205 scenarios and 379 checks** after the orientation repair. The original variant records 70 failures, including exceptions, across 57 scenarios. This is a partial native rebuild on a compatible base, not a clean build of current main. Full-model impact, GPU behavior, novelty and maintainer acceptance are unestablished.

## Minimal example

Use one 32×32 physical weight matrix. Every 4-bit code is 1, every scale is 1 and every bias is 0; group size is 32. Thus every dequantized weight is exactly 1. There is no quantization approximation in this example.

```python
import mlx.core as mx
mx.set_default_device(mx.cpu)
x = mx.array([1.] + [0.] * 31).reshape(1, 1, 32)
w = mx.full((1, 32, 4), 0x11111111, dtype=mx.uint32)
s = mx.ones((1, 32, 1))
b = mx.zeros((1, 32, 1))
ids = mx.array([0], dtype=mx.uint32)
f = lambda s, b: mx.gather_qmm(
    x, w, s, b, rhs_indices=ids,
    transpose=False, group_size=32, bits=4)
y, (ds, db) = mx.vjp(f, [s, b], [mx.ones((1, 1, 32))])
print(ds.reshape(-1).tolist())  # [1,1,...,1]; expected [32,0,...,0]
print(db.reshape(-1).tolist())  # [1,1,...,1]; expected [32,0,...,0]
```

Each output is the dot product of `x=[1,0,...,0]` with one weight column. Only physical row 0 contributes. Its group scale and group bias each affect all 32 outputs; the remaining rows contribute nothing.

| Quantity for the output sum | Independent reference | Official MLX 0.32.2 CPU |
|---|---|---|
| Forward | 32 ones | Correct |
| Input gradient | 32 in every coordinate | Correct |
| Scale gradient | `[32,0,...,0]` | `[1,1,...,1]` |
| Bias gradient | `[32,0,...,0]` | `[1,1,...,1]` |
| Scale finite differences, coordinates 0 and 1 | `[32,0]` | `[32,0]` |
| Bias finite differences, coordinates 0 and 1 | `[32,0]` | `[32,0]` |

Finite differences evaluate the actual forward operation with step `1/1024`. Inputs are finite FP32 tensors with compatible dimensions and in-range unsigned indices. An explicit `rhs_indices=[0]` reaches the gathered path; omitting both index arrays instead exercises the ordinary `quantized_matmul` fallback, which passes the tested controls.

[Executable reproduction](reproduce.py) · [Fresh wheel results](wheel-reproduction.json) · [Earlier comparison probe](probe.py) · [Recorded comparison results](probe-results.json)

## An actual update increases the loss

Keep `x`, the codes and scales fixed. Set the target to `[2,0,...,0]` and train only the group biases:

```text
L(b) = 0.5 * sum_j (y_j(b) - target_j)^2
y_j(b) = 1 + b[0] for all j
L(b) = 0.5 * ((b[0]-1)^2 + 31*(b[0]+1)^2)
dL/db[0] = 32*b[0] + 30
dL/db[k] = 0 for k != 0
```

At zero biases, the loss is 16 and the correct gradient is `[30,0,...,0]`. The original implementation returns `[-1,1,...,1]`.

For the identical learning rate `1/32`:

| Update | New first bias | Actual loss after update |
|---|---|---|
| Original MLX gradient | `1/32` | `16.953125` |
| Independent exact gradient | `-30/32` | `1.9375` |
| Locally patched C++ gradient | `-30/32` | `1.9375` |

For the wrong update, every output becomes `33/32`, so the loss is `(31² + 31*33²)/(2*32²) = 16.953125`. For the correct update, every output becomes `1/16`, giving `(31² + 31)/(2*16²) = 1.9375`.

Both native variants execute the actual forward operation after the update. The suite also compares the gradient with its explicit reference and checks forward finite differences at two bias coordinates. This is a small synthetic optimization example; no production-model training or financial impact was measured.

[Native loss experiment](native_regression.cpp) · [Original log](run-before.log) · [Patched log](run-after.log)

## Rectangular weights expose the shape error

For a physical 32×64 weight matrix with group size 32, the bias parameter has shape `(1,32,2)`. On the official wheel:

- The isolated bias VJP returns `(1,64,1)` instead.
- The isolated scale VJP raises a broadcasting exception between `(1,64,1,32)` and `(1,32,2,32)`.

The wrong orientation can stay hidden when both matrix dimensions are equal. Rectangular tests expose the dimensional mismatch before some numeric comparisons can run.

## Mathematical cause and proposed repair

The [official operation contract](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html) defines the non-transposed operation as `X @ W`. For physical weight row `k`, column `j`, expert `e` and column group `g(j)`:

```text
W_e[k,j] = scales_e[k,g(j)] * codes_e[k,j] + biases_e[k,g(j)]
R_e = sum over q with rhs[q]=e of transpose(X_lhs[q]) * G_q
db_e[k,g] = sum over j in group g of R_e[k,j]
ds_e[k,g] = sum over j in group g of R_e[k,j] * codes_e[k,j]
```

Here `G` is the output cotangent. Repeated selections must accumulate. Packed integer codes are fixed; this audit does not differentiate integer parameters.

The shared [gather_mm_grad helper](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp#L121) supplies the opposite orientation, `GᵀX`. [GatherQMM::vjp](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp#L3802) immediately splits its last dimension into quantization groups. For `transpose=False`, it first needs to restore the physical weight orientation.

The proposed change is:

```cpp
auto dw = gather_mm_grad(/* existing arguments */);
if (!transpose_) {
  dw = swapaxes(dw, -1, -2, stream());
}
dsb = unflatten(dw, -1, {-1, group_size_}, stream());
```

[Complete patch](gather-qmm-transpose-vjp.patch)

This patch changes only that orientation step. It does **not** include the previously published [sorted-index/broadcast optimization-precondition repair](https://www.gero.uz/research/articles/mlx-gather-vjp-preconditions.html). The minimal new counterexample uses the ordinary unsorted backward path, so it does not depend on enabling the sorted optimization. The two reports describe distinct causes; this patch is not claimed to fix all gathered-gradient cases.

## Actual validation results

The native harness uses independent double-precision scalar loops over unpacked integer codes to compute forward values and gradients. Its reference does not call MLX autodiff. Comparisons require the expected shape and finite values, with numerical tolerance `3e-5*(1+abs(reference))`.

The suite contains 34 tensor configurations, six execution scenarios per configuration, and one separate quadratic-update scenario:

- `transpose=True` and `False`; `(K,N)` of `(32,32)`, `(32,64)`, `(64,32)` and selected `(64,64)` controls.
- Affine 4-bit/8-bit codes, group sizes 32/64 and FP32 parameters.
- Repeated right indices, two explicit index arrays, left broadcasting with the sorted optimization disabled, and right-sorted calls under the required mapping conditions.
- Contiguous and selected strided inputs/cotangents; forward, joint and isolated input/scale/bias VJPs; zero and negative cotangents.
- No-index fallback controls and actual-forward finite differences in the quadratic example.

| Native variant | Scenarios attempted | Failed scenarios | Comparisons reached | Failure records, including exceptions |
|---|---|---|---|---|
| Original compatible base | `205` | `57` | `315` | `70` |
| Local orientation patch | `205` | `0` | `379` | `0` |

The comparison counts differ because exceptions stop some original scenarios before later assertions. The 70 failure records are manifestations of one orientation defect, not 70 separate discoveries. In this bounded suite, forward, isolated input gradients, `transpose=True` and the no-index fallback pass before the repair too.

[C++ harness](native_regression.cpp) · [Commands, exit codes and timings](build-results.json) · [Artifact validation](artifact-validation.json)

## Versions and reproduction boundary

- Official installed wheel: MLX `0.32.2`, CPU.
- Native source base: `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`.
- Inspected main: `24c699ecee2f7c8b2040de8da1c8382c8bcf31c7`.
- Apple clang 17.0.0, C++20, macOS arm64. Changed/test translation units compile at `-O0` and link ahead of an existing CPU archive.
- CPU archive SHA-256: `7826e14e2d1ee526a16352829d460ae39c048e331c339be0b6f8d67d74e68a06`.

Main adds global-scale support and index handling in GatherQMM. The relevant affine reduction block is unchanged; the [complete method comparison](GatherQMM-baseline-to-upstream.diff) records the differences. Execution used the compatible base, while current-main conclusions come from pinned source inspection and patch-application checks. **A clean full-main build and the complete upstream test suite were not run.**

Publication verification freshly repeated the small wheel reproduction and native before/after builds and runs. The earlier broad wheel comparison is retained as prior evidence and was not rerun for publication. All computations were sequential on CPU with numerical thread environment variables set to one. The wheel import required visibility of the Metal device outside the sandbox, but the probes explicitly selected CPU for computation.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python reproduce.py

export MLX_SOURCE_ROOT=/absolute/path/to/mlx-at-ce916db
export MLX_CPU_BUILD=/absolute/path/to/cpu-build
python3 build_and_test.py
python3 validate_artifacts.py
```

[BUILD.md](BUILD.md) describes the compatible archive setup. Public command records use descriptive path placeholders; new executions record their actual paths. [Versions and source hashes](source-metadata.json) · [Public package checksums](SHA256SUMS.json).

## Prior work and remaining limits

Four recorded GitHub API searches succeeded; a fifth returned HTTP 403. [Raw queries and responses](duplicate-search.json) are retained. Related [PR #4392](https://github.com/ml-explore/mlx/pull/4392) and [PR #4051](https://github.com/ml-explore/mlx/pull/4051) address non-transposed forward GPU dispatch/kernel behavior. They do not supply this missing orientation step in the pinned common affine VJP block. The existing [test_gather_qmm_grad](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/python/tests/test_quantized.py#L1716) uses `transpose=True`.

No exact matching report was identified in the inspected public material. This limited search does not establish priority or rule out private reports. The shallow local history does not establish when the defect was introduced.

Confirmed execution scope is CPU/FP32 affine VJP for the listed finite shapes and inputs. Metal/CUDA, other dtypes, non-affine modes, full models, performance and combined behavior with the separate sorted-path patch remain untested here. Passing this suite is not a proof over all possible states. No security impact, reward eligibility or upstream acceptance is claimed.

Prepared with AI assistance. Numerical claims are backed by actual local execution, independent algebra and finite differences. Independent GERO Research; no Apple endorsement. Included MLX code retains its [MIT license](MLX-LICENSE.txt).
