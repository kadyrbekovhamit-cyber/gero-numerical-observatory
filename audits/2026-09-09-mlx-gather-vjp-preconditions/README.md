# Sorted Indices, Wrong Gradients: MLX gather_mm and gather_qmm

9 September 2026 · Independent GERO Research · Xamit Kadirbekov

**Status: locally reproduced numerical-correctness defect.** The official MLX 0.32.2 CPU wheel produces incorrect gradients for valid sorted-index calls. One shared optimization-precondition mechanism affects `gather_mm` and the tested affine paths of `gather_qmm`.

Fresh native execution of the proposed C++ repair passes **69 scenarios and 626 checks**. The original variant fails 71 checks across 19 scenarios. This is a partial rebuild on a compatible base, not a clean build of current main. Novelty, maintainer acceptance, full-model impact and reward eligibility are unestablished.

## Minimal example

```python
import mlx.core as mx
mx.set_default_device(mx.cpu)
a = mx.array([2., 7.]).reshape(2, 1, 1)
b = mx.array([3., 5.]).reshape(2, 1, 1)
ids = mx.array([0, 0], dtype=mx.uint32)
f = lambda a, b: mx.gather_mm(
    a, b, lhs_indices=ids, sorted_indices=True)
y, (da, db) = mx.vjp(f, [a, b], [mx.ones((2, 1, 1))])
print(y[0].reshape(-1).tolist())  # [6, 10]: correct forward
print(da.reshape(-1).tolist())    # [3, 5]; expected [8, 0]
print(db.reshape(-1).tolist())    # [2, 7]; expected [2, 2]
```

The two outputs are `Y=[3a0, 5a0]`. For their sum, differentiation gives `dA=[8,0]` and `dB=[a0,a0]=[2,2]`. The second left matrix is not selected at all. Its nonzero gradient is not an ordinary rounding discrepancy.

The [official gather_mm contract](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html) permits the sorted optimization flag when one supplied index array is sorted; it does not require uniqueness. The [gather_qmm documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html) describes the corresponding quantized operation. The central examples use in-range unsigned indices, compatible finite FP32 arrays and a single nondecreasing index array. Disabling `sorted_indices` gives the expected gradients in these reproductions.

[Executable probe](probe.py) · [Fresh wheel results](probe-results.json)

## A measured step increases the loss

Keep `B` fixed and use the smooth quadratic loss:

```text
Y0 = 3 a0,  Y1 = 5 a0
L(A) = 0.5 (Y0 - Y1)^2 = 2 a0^2
gradient = [4 a0, 0]
at A=[2,7]: L=8 and gradient=[8,0]
```

The original sorted path returns `[-12,20]`. At learning rate `1/32`:

| Variant | Gradient | Updated A | Actual loss after update |
|---|---|---|---|
| Original sorted path | `[-12,20]` | `[2.375,6.375]` | `11.28125` |
| Correct / patched path | `[8,0]` | `[1.75,7]` | `6.125` |

Both losses come from evaluating the actual forward operation after the update. The native suite also checks the loss, both gradient coordinates, finite differences and the summed Hessian rows `[4,0]`. Central differences with step `1/1024` give exactly `[8,0]` in this small FP32 example. There is no nonsmooth tie or subgradient convention involved.

[Wheel loss and quantized probe](probe_quantized_and_loss.py) · [Measured numbers](quantized-and-loss-results.json) · [Native loss test](native_regression.cpp)

## The quantized case has an exact weight reference

In the minimal affine `gather_qmm` example, each packed 4-bit code is 1, the scales are 1, and the two biases are 2 and 4. The dequantized 32×32 matrices therefore contain exactly 3 and 5. There is no quantization approximation in these particular reference weights.

With two left input vectors containing 2 and 7, left indices `[0,0]`, and unit output cotangents:

| Gradient, first coordinate of each batch | Expected | Original sorted path |
|---|---|---|
| Input `dX` | `[256,0]` | `[96,160]` |
| Scale `dS` | `[64,64]` | `[64,224]` |
| Bias `dBias` | `[64,64]` | `[64,224]` |

The reference is ordinary scalar differentiation of `code*scale+bias`. The native suite additionally uses varied exact integer codes with 4-bit and 8-bit packing, independent double-precision scalar loops, and separate as well as joint input/scale/bias VJPs. Packed integer codes are held fixed; no derivative with respect to those integer codes is claimed.

## A logical slice depends on its synthetic backing tail

```python
backing = mx.array([2., 101., 202.]).reshape(3, 1, 1)
mx.eval(backing)
a = backing[:1]  # the logical input contains only 2
b = mx.array([3., 5., 7.]).reshape(3, 1, 1)
ids = mx.array([0, 1, 2], dtype=mx.uint32)
f = lambda b: mx.gather_mm(
    a, b, rhs_indices=ids, sorted_indices=True).sum()
print(f(b).item())                      # 30: correct
print(mx.grad(f)(b).reshape(-1).tolist()) # [2,101,202], expected [2,2,2]
```

Changing only the backing tail to `[-31,47]` changes the gradient to `[2,-31,47]`, while the visible input and forward result remain unchanged. With the sorted flag disabled, both tails give `[2,2,2]`. After the local C++ patch, both tails also pass with the flag enabled.

**This test uses an explicitly allocated, initialized synthetic array within one process.** It establishes a dependence on values outside the logical slice. It does not establish access outside the allocated buffer, another user's data, another process, or a remotely exploitable condition. No such impact is claimed.

The saved [CPU implementation](upstream-mlx_backend_cpu_masked_mm.cpp) advances through segments supplied by the backward path. In this case, the optimization creates segments incompatible with the broadcast left operand's logical row count.

[Controlled synthetic probe](probe_broadcast_tail.py) · [Fresh results](broadcast-tail-results.json)

## Cause and proposed repair

For matrix-level gathers, the required reverse rule is:

```text
Yq = A[iq] B[jq]
dA[l] = sum over q with iq=l of Gq transpose(B[jq])
dB[l] = sum over q with jq=l of transpose(A[iq]) Gq
```

The derivatives must accumulate repeated selections and respect broadcasting. At pinned main, [GatherMM::vjp](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp#L6080) and [GatherQMM::vjp](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp#L3730) enable the optimized path with:

```cpp
bool sorted = left_sorted_ || right_sorted_;
```

That segmented path assumes the left rows already correspond to output rows. With repeated left indices, equal element counts do not establish that correspondence. With broadcasting, the original left operand can contain fewer logical rows than the segments assume. The shared `gather_mm_grad` helper does not apply the missing left selection or expand the broadcast operand.

The local patch instead gates this optimization after the existing shape comparison:

```cpp
bool sorted = right_sorted_ && no_broadcast;
```

Other cases use the existing gather/scatter formulas. The ordinary right-sorted, non-broadcast fast path is retained. This changes the precondition in two VJP methods; it does not introduce a new kernel.

[Proposed C++ patch](gather-vjp-preconditions.patch)

**This is a bounded repair proposal.** CPU `GatherMM` supports FP32 in the inspected backend. Some FP16 quantized cases can reach an unsupported CPU fallback after this change. The retained exploratory run contains 46 exceptions and is not counted as repaired numerical behavior. GPU execution and performance remain untested; review is needed before adopting the patch in production.

## Actual validation results

| Native variant | Scenarios passed | Checks passed | Failed checks |
|---|---|---|---|
| Original compatible base | `50/69` | `555/626` | `71` |
| Local patch on that base | `69/69` | `626/626` | `0` |

The primary suite covers FP32 left/right indices, repeated and omitted selections, identity selection, broadcasting, strided matrices, separate and joint VJPs, zero cotangents, negative cotangents for ordinary matrix multiplication, affine 4/8-bit packed weights, actual-forward finite differences, a second derivative, and two initialized synthetic backing tails. Both-explicit-index calls provide additional controls; the central contract counterexamples supply one sorted array.

Reference outputs and gradients are computed by independent scalar loops in double. The primary comparison uses `3e-5*(1+abs(reference))` together with exact shape checks and finite-value checks. The small integer examples above are exact enough to separate the defect from tolerance choices.

Fresh publication verification repeated all three CPU wheel probes and the native build/run sequence. All **38 original artifact hashes** matched before copying; the artifact validator passed **45/45** checks. The 71 failed checks are manifestations of the shared precondition problem, not 71 independent findings.

[Baseline log](run-before.log) · [Patched log](run-after.log) · [Build commands and exit codes](build-results.json) · [Artifact validation](validation-results.json)

## Versions and reproduction boundary

- Official wheel: MLX `0.32.2`, CPU.
- Inspected main: `24c699ecee2f7c8b2040de8da1c8382c8bcf31c7`.
- Native compatible base: `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`.
- `GatherMM::vjp` and the shared `gather_mm_grad` body match between the two source versions.
- `GatherQMM::vjp` differs: main adds global-scale handling and argument-index checks. The faulty condition and relevant affine paths remain. [Complete method comparison](GatherQMM-baseline-to-upstream.diff).
- Apple clang 17.0.0, C++20, macOS arm64; sequential `-O0` translation-unit builds linked ahead of an existing CPU-only archive.
- Reused archive SHA-256: `7826e14e2d1ee526a16352829d460ae39c048e331c339be0b6f8d67d74e68a06`.

The proposed patch applies to the saved main source; a separate application check verifies that. Actual native execution used pristine/patched translation units from the compatible base with the reused archive. **This is not a full clean build of current main.** The original checkout and installed wheel were not modified. Computational thread environment variables were set to one.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python probe.py
.venv/bin/python probe_quantized_and_loss.py
.venv/bin/python probe_broadcast_tail.py

export MLX_SOURCE_ROOT=/absolute/path/to/mlx-at-ce916db
export MLX_CPU_BUILD=/absolute/path/to/cpu-build
python3 build_and_test.py
python3 validate_artifacts.py
```

[BUILD.md](BUILD.md) describes the required local archive and compilation database. The public command records replace private absolute paths with descriptive variables; executing the runner writes actual commands for the new environment. [Source versions and hashes](source-metadata.json) · [Package checksums](SHA256SUMS.json).

## Prior work and limits

Seven recorded GitHub queries did not identify an exact matching counterexample. That bounded search does not establish priority. [PR #2335](https://github.com/ml-explore/mlx/pull/2335) introduced an MoE backward optimization; the recorded discussion does not contain these examples. [Issue #4253](https://github.com/ml-explore/mlx/issues/4253) and [PR #4261](https://github.com/ml-explore/mlx/pull/4261) concern a strided-input forward problem. Here, forward is correct and backward fails even on a contiguous FP32 example.

The inspected `test_gather_mm_sorted_vjp` and `test_gather_qmm_grad` exercise right-sorted calls without left broadcasting. That control passes in the new harness too. Raw search results and related discussions are retained in the package.

Only the listed finite inputs, shapes and affine modes were tested. Metal/CUDA, non-affine quantization, quantized backward with `transpose=False`, large-model training, performance and the full upstream test suite remain outside the executed scope. Passing these tests is not a proof over all states. Maintainer acceptance, security impact and reward eligibility are unestablished.

Prepared with AI assistance. Measurements come from actual code execution, independent algebra and finite differences. Independent GERO Research; no Apple endorsement. Included MLX code retains its [MIT license](MLX-LICENSE.txt).
