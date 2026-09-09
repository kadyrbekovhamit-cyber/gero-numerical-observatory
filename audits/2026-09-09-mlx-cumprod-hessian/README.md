# A smooth polynomial, a NaN Hessian: MLX cumprod at zero

**Locally reproduced in MLX 0.32.2; C++ prototype validated, 9 September 2026.**
One higher-order autodiff defect is reproduced. The first gradient is correct
in the minimal case, but differentiating it again produces non-finite values.
The candidate repair is a division-free prototype with **O(N log N) scalar
work**, not a production-ready performance improvement.

[GERO article and short video](https://www.gero.uz/research/articles/mlx-cumprod-hessian-at-zero.html)
· [C++ harness](native_regression.cpp) · [Patch](cumprod-polynomial-vjp.patch)
· [Sources and prior work](SOURCES.md)

## Minimal mathematical example

For a real vector x=[a,b,c], forward inclusive cumulative product gives:

```text
F(a,b,c) = sum(cumprod([a,b,c])) = a + ab + abc
gradient = [1+b+bc, a+ac, ab]
Hessian  = [[0, 1+c, b],
            [1+c, 0, a],
            [b, a, 0]]
```

At [0,2,3], the released CPU float32 wheel returns F=0 and gradient=[9,0,0],
both correct. Repeated reverse-mode differentiation returns:

```text
MLX Hessian:             Exact Hessian:
[[NaN, 4, 2],            [[0, 4, 2],
 [NaN, 0, 0],             [4, 0, 0],
 [NaN, 0, 0]]             [2, 0, 0]]
```

This is a polynomial, smooth everywhere. There is no mathematical singularity
at the zero input. Differentiating the explicit polynomial with the same MLX
wheel gives the correct Hessian. Central finite differences of MLX's actual
first gradient also recover it. The full matrices and inputs are retained in
[probe-results.json](probe-results.json).

```python
import mlx.core as mx
mx.set_default_device(mx.cpu)
f = lambda x: mx.sum(mx.cumprod(x))
x = mx.array([0., 2., 3.])
g = mx.grad(f)
H = mx.stack([mx.grad(lambda z: g(z)[i])(x) for i in range(3)])
print(g(x))
print(H)
```

An even smaller native case uses a single zero: inclusive cumprod is the
identity function, whose second derivative must be zero. The original
implementation returns NaN. Inclusive/exclusive and both scan directions
are covered by the native selection.

## Cause: differentiating the zero-handling graph

The Prod branch of `Scan::vjp` in
[pinned primitives.cpp](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp)
uses division by the input. Special treatment of the first zero and `where`
produce the correct first derivative, but the backward graph still contains
zero divisions when differentiated again. Masking a result does not give that
graph the derivatives of the underlying smooth polynomial.

The earlier [PR #1167, Stable cumprod grad at 0](https://github.com/ml-explore/mlx/pull/1167),
merged on 31 May 2024, repaired first-order gradients. Its added tests check
first gradients, not repeated differentiation. This report concerns a
higher-order failure beyond that inspected test coverage; it does not claim
that zero-input cumprod gradients had never been studied.

## Division-free repair: mathematical construction

For a real inclusive forward scan, incoming cotangents c_i, and finite small
inputs in the tested domain:

```text
P_i     = product(x_k for k < i)  # empty product = 1
B_(n-1) = c_(n-1)
B_i     = c_i + x_(i+1) * B_(i+1)
dL/dx_i = P_i * B_i
```

Expanding the recurrence sums the derivatives of the product monomials.
There is no division by x_i and no branch on its value. Exclusive mode shifts
the cotangents by one position and pads with zero. Reverse mode reverses the
direction. The implementation retains complex conjugation in its input path,
but **complex dtype was not validated**.

The prototype composes affine maps `t -> b+a*t` over distances 1,2,4,...;
each round doubles the covered suffix. An exclusive cumprod supplies P_i.
It modifies only the Prod branch of `Scan::vjp`. The source patch applies
cleanly to the inspected main and produces the saved candidate byte-for-byte.

The total scalar work is O(N log N), with O(log N) array-operation rounds,
where N is the scan-axis length. A work-efficient associative scan, memory
use and realistic performance need further study before upstream adoption.
No speedup or acceptable large-array overhead is claimed. Repeated use of
the patched scan in higher derivatives is covered only by the finite tests.

## Actual before/after results

| Native CPU float32 variant | Passing scenarios | Failing scenarios | Failed checks |
|---|---:|---:|---:|
| Original | 13 / 57 | 44 | 172 / 311 |
| Division-free prototype | 57 / 57 | 0 | 0 / 311 |

These are **57 scenarios and 311 numerical checks**, not the complete MLX
suite. A vector comparison is one check; individual elements are not counted
as separate checks. The 44 failures are manifestations of one defect.

The native reference independently enumerates each product monomial and its
derivatives in double precision, without autodiff. Weighted output sums use
weights cycling [1,-0.5,2]; the Python minimal example uses unit weights.

The selection covers:

- Both directions and inclusive/exclusive mode; zero at different positions,
  multiple zeros, all-zero inputs and nonzero controls.
- Lengths 0,1,3,5,8; vectors and two-dimensional arrays with axes 1 and -2.
- Forward values, first gradients and every Hessian row for these arrays.
- Mixed third and fourth derivatives at three zeros in all four modes.
  The third derivative is 1 for inclusive and 0 for exclusive; the fourth is 0.
- A finite-difference check of the first gradient, which passes even before
  repair and independently supports the exact Hessian.

Main tolerance: `2e-5*(1+abs(reference))`. Central differences use h=1/256
and `1e-3*(1+abs(reference))`. Small exactly representable input values
avoid unrelated overflow/underflow in this selection. Passing these tests
does not prove correctness for all finite inputs or derivative orders.

[Before log](run-before.log) · [After log](run-after.log)
· [Counts](validation-results.json) · [Commands](build-results.json)

## Versions and what was rebuilt

| Component | Recorded version / method |
|---|---|
| Wheel | MLX 0.32.2, CPU float32 |
| Inspected main | 24c699ecee2f7c8b2040de8da1c8382c8bcf31c7 |
| Native compatible base | ce916dbbcaa88e433b6fd1e60a17f766d49c27fe |
| Compiler | Apple clang 17.0.0 (clang-1700.0.13.5), C++20 |
| Environment | macOS 15.5 arm64; Python 3.12.14 |
| Native build method | Recompiled pristine/patched primitives.cpp, linked ahead of an existing CPU-only static archive |

**This is a partial native rebuild on a compatible base, not a fresh complete
build of main.** The entire Scan::vjp method is byte-identical between the
base and inspected main. The archive hash and source hash are recorded in
[source-metadata.json](source-metadata.json). The original checkout, archive
and installed wheel were not edited.

All 31 supplied manifest entries matched before publication. The wheel probe
and native before/after compilation and execution were rerun sequentially.
The portable runner changes only filesystem references; the public Python
probe omits unrelated scouting checks while retaining the six cumprod cases.
Local machine paths in build metadata are replaced by explicit placeholders.

## Reproduce

On a compatible Mac, from this evidence directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python probe.py

# Use a compatible CPU static build of the recorded base:
export MLX_SOURCE_ROOT=/absolute/path/to/mlx-at-ce916db
export MLX_CPU_BUILD=/absolute/path/to/cpu-build
.venv/bin/python build_and_test.py
```

[BUILD.md](BUILD.md) documents the build setup recipe and its validation
boundary. The runner reads the existing compile_commands.json, rebuilds the
harness and both translation-unit variants, and runs each in sequence. It
expects baseline exit code 1 and patched exit code 0. Inspect the recorded
numerical counts, not only the runner's successful exit code.

Thread-related environment variables are set to one. No physical-core
affinity is claimed. The reused archive is CPU-only; a macOS wheel import
can initialize Metal even though these calculations explicitly use CPU.

## Limits and attribution

Not tested: Metal/CUDA, complex values, FP16/BF16, compiled graphs, large
arrays, extreme scales, higher derivatives beyond the selected orders or
full-model training. The prototype does not add a cumprod JVP implementation.
Performance acceptability, maintainer acceptance and novelty are unestablished.
No device harm, security impact or entitlement to a reward is claimed.

The issue/PR search found no exact match within the recorded selection;
it cannot exclude private reports, unindexed discussions or other wording.
See [SOURCES.md](SOURCES.md) for prior work and search limits.

Prepared with AI assistance. Numerical evidence is actual local execution,
supported by explicit polynomial and finite-difference references. This is
independent GERO Research work, not an Apple-endorsed or third-party audit.
Included MLX source retains its MIT license.

#MachineLearning #MLX #Autodiff #NumericalComputing #SoftwareTesting
