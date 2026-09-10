# MLX: a trainable output mask loses its gradient at zero

10 September 2026 · GERO Research / Xamit Kadirbekov · AI-assisted preparation

A first-order derivative defect is reproduced in **MLX 0.32.2**: `block_masked_mm` applies the floating-point output mask an extra time inside that mask's own backward rule. A zero mask receives zero gradient even when the derivative of the executed forward function is nonzero. Fractional masks rescale the gradient and negative masks can reverse its sign.

Fresh local C++ validation records **604 comparisons across 377 scenarios**: 77 mismatches before the one-line repair and none after it. This is one defect, not 77 discoveries. The experiment concerns differentiable real output masks. It does not establish a problem with every MLX model, ordinary matrix multiplication, or the tested input gradients under a fixed Boolean output mask.

## Minimal example

Let `A` be an all-ones 2×3 matrix and `B` an all-ones 3×2 matrix. Choose block size 32 and a single real output-mask value `m`. There are four output elements, each equal to `3m`:

```text
F(m) = sum(block_masked_mm(A, B, 32, m)) = 12m
F'(m) = 12
```

The [public API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.block_masked_mm.html) supports output masks and block sizes 32 or 64. The single 1×1 mask has the required tile dimensions here. [PR #1152](https://github.com/ml-explore/mlx/pull/1152) explicitly introduced multiplicative floating-point masks and their derivatives; this is not an attempt to differentiate a Boolean selection decision.

```python
import mlx.core as mx
mx.set_default_device(mx.cpu)
a = mx.ones((2, 3))
b = mx.ones((3, 2))
f = lambda m: mx.block_masked_mm(a, b, 32, m).sum()
m = mx.zeros((1, 1))
print(f(m).item())            # 0: correct forward
print(mx.grad(f)(m).item())   # 0: expected 12
```

| Mask m | Actual forward F(m) | Original gradient | Actual-forward finite difference | Patched C++ gradient |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 12 | 12 |
| 0.5 | 6 | 6 | 12 | 12 |
| 1 | 12 | 12 | 12 | 12 |
| −2 | −24 | −24 | 12 | 12 |

The finite-difference step is `1/1024`. The arithmetic values in this minimal example are exactly representable in FP32. Both the installed wheel and the native baseline actually execute the forward operation; the reference is not inferred only from source code. The native repaired variant then returns 12 for all four masks.

[Reproduction](reproduce.py) · [Fresh wheel results](wheel-reproduction.json) · [Native test](native_regression.cpp)

## A concrete optimization step stalls

Keep the same matrices and make every target output equal to 3. Train only the real output mask:

```text
L(m) = 0.5 * sum over four elements of (3m - 3)^2
     = 18(m - 1)^2
L'(m) = 36(m - 1)
```

At `m=0`, the exact derivative is −36. The faulty rule returns 0. With learning rate `1/36`, a correct update sets `m=1`, reaching the target exactly; the original update leaves `m=0`.

| Execution | Initial loss | Computed gradient | Updated mask | Actual next loss |
|---|---:|---:|---:|---:|
| MLX 0.32.2 wheel / original C++ | 18 | 0 | 0 | 18 |
| Patched C++ | 18 | −36 | 1 | 0 |

The native test reevaluates the actual forward after each update. This small synthetic objective demonstrates stalled learning for this parameter and initialization. It is not a measurement of production-model quality or a general guarantee that the same step size is appropriate elsewhere.

## Cause and one-line repair

Let `C` be the matrix product after applying any input masks, before the output mask. Let `E` expand the output mask `O` over its blocks, and let `G` be the incoming gradient:

```text
Y = C * E(O)
Correct output-mask VJP = reduce_to_mask_shape(G * C)
Original output-mask VJP = reduce_to_mask_shape(G * C * E(O))
                         = O * correct_output_mask_VJP
```

`reduce_to_mask_shape` includes sums within partial edge tiles and across dimensions broadcast by the mask. The extra mask factor is not an IEEE-754 rounding effect.

In [BlockMaskedMM::vjp at the inspected revision](https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/mlx/primitives.cpp#L5997), the branch computing the output-mask derivative recomputes `C` using the original output mask. Remove that argument **only in this derivative branch**:

```diff
       auto C = block_masked_mm(
           primals[0],
           primals[1],
           block_size_,
-          primals[2],
+          std::nullopt,
           lhs_mask,
           rhs_mask,
           stream());
```

Input masks remain applied. The repair uses no division by the output mask and works at zero. The rest of the operation is unchanged in the supplied patch.

[Patch](output-mask-vjp.patch) · [Saved upstream source](upstream-mlx_primitives.cpp) · [Base-to-upstream method comparison](BlockMaskedMM-baseline-to-upstream.diff)

## Actual before/after validation

The independent reference in `native_regression.cpp` uses scalar loops with `double` accumulation for forward values and each argument's derivative. It does not call the VJP under test to produce its expected gradients. Each array comparison checks shape, finite values and componentwise error at most `1e-5 * (1 + abs(reference))`.

| Variant | Executed scenarios | Array/scalar comparisons | Mismatches | Scenarios with mismatches |
|---|---:|---:|---:|---:|
| Original C++ | 377 | 604 | 77 | 76 |
| Patched C++ | 377 | 604 | 0 | 0 |

Both variants reach all 604 comparisons. No exceptions or shape failures were recorded. All original mismatches belong to output-mask derivatives or the demonstrated optimization step.

The suite covers:

- Blocks 32 and 64; rectangular matrices and partial edge tiles.
- Zero, one, fractional, negative and heterogeneous floating-point output masks.
- Input masks present/absent and a no-output-mask control.
- Batch broadcasting for inputs and masks, including a scalar broadcast mask.
- Noncontiguous inputs and incoming gradients.
- Joint VJPs, individual-argument VJPs and reordered differentiable arguments.
- Input gradients with a fixed Boolean output mask.
- The four finite-difference cases and the actual optimization step.

The tests use finite synthetic FP32 arrays with compatible matrix and mask shapes. Zero output-mask entries are part of the tested domain, not excluded as nondifferentiable: the floating-mask forward expression is linear in each mask entry.

[Before log](run-before.log) · [After log](run-after.log) · [Build commands and timings](build-results.json)

Compilation and execution were sequential with compute-library thread settings fixed at 1. The publication rerun used about **3.66 child CPU-seconds** for compilation, linking and the two native executions; the earlier recorded run used about 3.45. These are local process accounting measurements, not portable performance benchmarks. The fresh wheel reproduction used about 0.005 CPU-seconds within its measured section.

## Reproducibility and versions

| Component | Recorded version or scope |
|---|---|
| Installed package | MLX 0.32.2 |
| Native base | `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe` |
| Inspected public source | `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a` |
| Native platform | macOS arm64, Apple Clang 17, C++20, Accelerate |
| Local compilation | Changed translation unit and test compiled at `-O0` |
| Runtime | FP32 CPU, compute-library thread limits set to 1 |
| Archive SHA-256 | `7826e14e2d1ee526a16352829d460ae39c048e331c339be0b6f8d67d74e68a06` |

The complete `BlockMaskedMM::vjp` method matches byte-for-byte between the pinned native base and inspected public source. Each native variant links a separate pristine or repaired `primitives.cpp` before the existing CPU archive. This is **a partial native rebuild, not a complete clean build of the current main branch**. Source/archive hashes and provenance are in [source-metadata.json](source-metadata.json).

Installed-wheel reproduction, on a compatible macOS environment:

```sh
python3 -m venv .venv
.venv/bin/pip install mlx==0.32.2
.venv/bin/python reproduce.py
```

Native reproduction with an existing compatible CPU archive and its `compile_commands.json`:

```sh
export MLX_SOURCE_ROOT=/absolute/path/to/mlx-source
export MLX_CPU_BUILD=/absolute/path/to/cpu-build
python3 build_and_test.py
python3 validate_artifacts.py
```

See [BUILD.md](BUILD.md) for the archive setup recipe and dependency/build limits. The runner reads the pinned base from Git, writes only beside itself and preserves the source working tree. `validate_artifacts.py` checks consistency of saved evidence; it does not replace numerical execution. [File checksums](SHA256SUMS.json) identify the published artifacts.

## Existing tests and related reports

The inspected [test_block_masked_matmul](https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/python/tests/test_blas.py#L1072) differentiates the matrices and input masks while capturing the output mask in a closure. That specific test does not request an output-mask derivative, which explains why it does not catch this case. This says nothing about every possible downstream test.

The publication check repeated three GitHub issue/PR searches. Relevant results include [PR #978](https://github.com/ml-explore/mlx/pull/978), introducing masked multiplication, and [PR #1152](https://github.com/ml-explore/mlx/pull/1152), adding multiplicative floating masks and their VJPs. They document the intended feature. [PR #4441](https://github.com/ml-explore/mlx/pull/4441) repairs `Pad::vjp` axis handling, including higher-order paths; it is a different cause from the first-order extra-mask multiplication reproduced here.

No exact output-mask-gradient correction was identified in the reviewed search results. Search coverage does not establish originality or exclude private or differently titled reports. [Search responses](publication-duplicate-search.json) and [related report bodies](related-public-reports.json) are included.

## Limits and remaining questions

This report does not test GPU execution, other dtypes, higher-order derivatives, full models, the entire upstream test suite or all possible shapes. It does not establish security impact, maintainer acceptance or reward eligibility. Successful tests are bounded evidence, not a proof for every state.

The suspected `needs_lhs_mask_vjp` / `needs_rhs_mask_vjp` bookkeeping issue remains unconfirmed as a separate numerical defect: later branches can recompute missing intermediates. Mixed GatherQMM derivatives are outside this package. No additional finding is asserted from those hypotheses.

Independent research, with AI assistance disclosed. No Apple endorsement or sponsorship. The executable examples, algebra, patch and limitations are provided for review and repair.
