# An Unused Value Gets a Gradient: MLX Masked Assignment under vmap

9 September 2026 · GERO Research · Xamit Kadirbekov

**Locally reproduced numerical defect.** With a fixed boolean mask, MLX's mapped assignment reads the source separately for each batch example, but its reverse derivative uses a global source counter. Unused source elements can receive another example's gradient. The forward operation remains correct in the checked cases.

Fresh publication reruns reproduce the result on the official **MLX 0.32.2 wheel, CPU float32**, and on actual C++ code. The local C++ patch passes **19 scenarios and 255 checks**; the original passes 8 scenarios and fails 72 checks. These failures concern one mechanism, not 72 separate defects. Full-model impact and novelty remain unestablished.

## Minimal public Python reproduction

```python
import mlx.core as mx
mx.set_default_device(mx.cpu)

mask = mx.array([[True, False, True, False],
                 [False, True, False, True]])
source = mx.array([[10., 20., 99.], [30., 40., 88.]])
cotangent = mx.array([[1., 2., 3., 4.], [10., 20., 30., 40.]])

def assign(dst, src, m):
    out = dst + 0
    out[m] = src
    return out

def loss(src):
    out = mx.vmap(assign)(mx.zeros((2, 4)), src, mask)
    return (out * cotangent).sum()

print(mx.grad(loss)(source))
# MLX 0.32.2: [[1, 3, 20], [40, 0, 0]]
# Expected:   [[1, 3,  0], [20,40, 0]]
```

The full executable [probe](probe.py) also compares an explicit Python loop, coordinate finite differences and forward/reverse derivative consistency. [Wheel results](probe-results.json) were regenerated for publication.

| Quantity | Measured result |
|---|---|
| Forward output, mapped and loop | `[[10,0,20,0],[0,30,0,40]]` |
| Expected source gradient | `[[1,3,0],[20,40,0]]` |
| Actual mapped source gradient | `[[1,3,20],[40,0,0]]` |
| Explicit-loop source gradient | `[[1,3,0],[20,40,0]]` |
| Actual-forward coordinate finite differences | `[[1,3,0],[20,40,0]]` |

`source[0,2] = 99` is unused. Changing it cannot change the output or loss, yet its mapped gradient is 20. The destination gradient is correct in this example.

## Preconditions and why the input is valid

The masks are fixed booleans and exactly match each destination row. Each source row contains three elements and its mask selects two positions. All inputs are finite float32 values; all forward source reads lie in bounds.

The [official boolean assignment contract](https://ml-explore.github.io/mlx/build/html/usage/indexing.html#boolean-mask-assignment) permits a non-scalar source with more elements than the mask consumes. The example therefore does not rely on an insufficient source or malformed mask. For a fixed mask the operation is linear, with a unique derivative; no convention at a nondifferentiable extremum is involved.

## Independent mathematical checks

Write the source entries as `s00,s01,s02,s10,s11,s12`. The weighted forward loss is exactly

```text
L(s) = s00 + 3*s01 + 20*s10 + 40*s11.
```

Its derivative is `[1,3,0,20,40,0]`. Central coordinate differences of the actual MLX forward, with step `1/256`, recover precisely that vector. An explicit per-example loop gives the same gradient.

There is also a direct adjoint counterexample. Let `v` be one only at the unused source entry `s02`. The actual forward-mode derivative is `Jv=0`. Thus `<Jv,c>=0`, while the mapped reverse derivative gives `<v,J^T c>=20`. The explicit-loop control returns zero on both sides.

## A measured optimization step goes uphill

Change only `cotangent[1,1]` from `20` to `-20` and use step size `eta=1/8`. Now

```text
L(s) = s00 + 3*s01 - 20*s10 + 40*s11 = 1070.
true gradient g = [1,3,0,-20,40,0]
mapped gradient h = [1,3,-20,40,0,0]
g dot h = -790,  g dot g = 2010
L(s - eta*h) = 1070 + 98.75 = 1168.75
L(s - eta*g) = 1070 - 251.25 = 818.75
```

| Update | Loss measured after executing the forward again |
|---|---:|
| Official wheel / original C++ gradient | 1168.75 |
| Explicit-loop gradient | 818.75 |
| Patched C++ gradient | 818.75 |

These are executed results, not only a symbolic prediction. This tiny synthetic linear loss establishes a wrong update direction in this case; it does not establish training degradation in a deployed model.

## Root cause and proposed repair

In [pinned `primitives.cpp`, lines 4767 onward](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp#L4767), `MaskedScatter::vjp` flattens the full batched mask and computes one exclusive prefix sum:

```cpp
const array mask_flat = flatten(mask_b, s);
const array idx_src =
    cumsum(astype(mask_flat, int32, s), 0, false, false, s);
```

At the primitive boundary the leading dimension is the batch dimension. Let each flattened destination row have length `M`, source row length be `K`, and

```text
p[b,j] = number of true mask entries before j within row b.
```

For a true mask position the correct flattened source-gradient address is `b*K+p[b,j]`. The original instead uses `sum(T[r] for r<b)+p[b,j]`, where `T[r]` is the number of true positions in earlier rows. These agree when every previous source row is fully consumed, which explains passing controls.

The [patch](masked-scatter-batch-vjp.patch) changes only `MaskedScatter::vjp`:

1. Reshape the expanded mask into batch rows and compute an exclusive prefix sum separately within each row.
2. Add the source-row offset `b*K` using int64 indices.
3. Map false positions to an in-bounds address with a zero contribution; zero-valued updates alone do not make an index valid.
4. Handle empty source or mask before dividing by a batch size.

Forward, JVP and vmap are unchanged. Existing outer broadcast and reshape derivatives handle shared arguments. The implementation remains linear in the source/destination element counts, but changes index width and allocation layout. Large-array performance has not been measured.

## Executed regression coverage

The [C++ harness](native_regression.cpp) calls actual public `masked_scatter`, `vmap`, `vjp`, `jvp` and `grad` operations. Its independent reference explicitly identifies the source or destination entry responsible for each output.

Checks cover forward values; both argument gradients together and separately; cotangent scaling; JVP/VJP adjoint consistency; coordinate differences for source and destination; a quadratic-loss gradient and Hessian-vector product; and the optimization step above.

Scenarios include padded sources, varying true counts, an empty selected region in the first or last example, shared source/mask/destination, scalar sources, block masks over leading axes, non-leading mapped axes, nested vmap, all-true/all-false masks and empty inputs.

| Actual C++ variant | Scenarios passed | Checks | Failed checks |
|---|---:|---:|---:|
| Original | 8/19 | 255 | 72 |
| Local patch | 19/19 | 255 | 0 |

[Before log](run-before.log) · [After log](run-after.log). All original forward and JVP controls pass. These are targeted regression tests, not the full MLX suite or a proof for every possible state.

## Versions, commands and build boundary

- Inspected main: `24c699ecee2f7c8b2040de8da1c8382c8bcf31c7`, rechecked unchanged before publication.
- Native base: `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`.
- Actual VJP, JVP, vmap, public masked-scatter function and two CPU forward functions match the inspected source byte for byte.
- Official Python wheel: MLX 0.32.2, CPU float32; freshly rerun for publication.
- Native compiler: Apple clang 17.0.0, C++20, macOS 15.5 arm64.

The native verification separately compiles pristine and patched `primitives.cpp` and links each object ahead of a reused CPU-only MLX archive. **This is a partial native rebuild, not a clean full build of current main.** Archive/source hashes are in [source-metadata.json](source-metadata.json); commands and exit codes are in [build-results.json](build-results.json). The public runner uses environment-configured paths and the pinned baseline revision.

```sh
# Official wheel probe on a supported host:
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python probe.py

# Native build setup is documented in BUILD.md:
export MLX_SOURCE_ROOT=/absolute/path/to/mlx-source
export MLX_CPU_BUILD=/absolute/path/to/cpu-build
python3 build_and_test.py
./native-before  # expected: 72 failing checks, exit 1
./native-after   # expected: 255 checks pass, exit 0
```

[Build setup and dependency boundary](BUILD.md). Runs were sequential, CPU-only, with compute-thread counts limited to one. The 31 original artifact hashes matched; the regenerated package passed 49 artifact checks. The original audit directory was left unchanged.

## History, uncertainty and attribution

The original [boolean-assignment PR #2663](https://github.com/ml-explore/mlx/pull/2663) already contained a global reverse counter. That establishes source history, not prior reporting of this precise defect. [PR #3633](https://github.com/ml-explore/mlx/pull/3633) fixes another mechanism in JVP; the present example has a correct JVP and incorrect VJP. The saved limited search did not identify an exact batch-VJP report. It did not cover every comment, commit, private report or fork. **Novelty and maintainer acceptance are unestablished.**

GPU, FP16/BF16, complex inputs, compiled execution, large arrays and full-model effects were not verified. No physical memory violation, security impact, device harm or reward eligibility is asserted.

Prepared with AI assistance; measurements come from actual local execution, independent algebra and finite differences. This is independent GERO Research work, not an Apple-endorsed audit. Included MLX code retains its MIT license. [Sources](SOURCES.md) · [Checksums](SHA256SUMS.json).

#MLX #Autodiff #MachineLearning #NumericalComputing #SoftwareTesting
