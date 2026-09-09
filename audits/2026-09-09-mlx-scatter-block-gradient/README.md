# A missing gradient in MLX block updates

**Local C++ reproduction, CPU float32, 9 September 2026.**

`scatter_max` and `scatter_min` can lose a winning update's gradient when an
update spans more than one element along an indexed axis. The counterexample
uses valid, in-bounds blocks and strictly distinct competing values. Its
derivative is unambiguous. A minimal shape correction passes **26/26 targeted
checks across two scenarios**; the baseline fails 12 checks in that selection.

[GERO article and short video](https://www.gero.uz/research/articles/mlx-scatter-block-gradient.html)
· [C++ tests](native_regression.cpp) · [Minimal patch](scatter-block-extent-only.patch)
· [Sources](SOURCES.md)

## Minimal example: the last winning element loses its gradient

```text
source       = [0, 0, 0, 0]
indices      = [0, 1], axis = 0
updates      = [[3, 4], [5, 6]]  # shape (2, 2)
cotangent    = [2, -3, 4, 2]

scatter_max  = [3, 5, 6, 0]
source VJP   = [0, 0, 0, 2]     # correct
updates VJP:
  expected   = [[2, 0], [-3, 4]]
  actual     = [[2, 0], [-3, 0]]
```

Block one writes positions 0 and 1; block two writes positions 1 and 2.
At the overlap, 5 beats 4. Every winning comparison is strict. All indices
and block ends are within the four-element source.

Define the scalar loss as the dot product of the output and the cotangent.
In a sufficiently small neighborhood that preserves the winners,

```text
L = 2*u00 - 3*u10 + 4*u11 + 2*source3
dL/dupdates = [[2, 0], [-3, 4]]
```

This is a locally linear function. Coordinate-wise central differences of
the **actual MLX C++ forward operation**, using h=1/256, independently return
`[2, 0, -3, 4]`. Both source and update coordinates are checked.
Negating the updates and using `scatter_min` reproduces the same gradient
loss with the same expected derivative vector.

A separate invariant checks the sum of contributions. Add one scalar t to
every source and update value. The extrema shift by t, so the loss shifts
by `t * sum(cotangent) = 5*t`. The original backward gives 1 instead of 5.

The block example is a **C++ API reproduction**. This exact block layout
was not independently reproduced through Python's `.at` interface.

## Cause and minimal repair

In `Scatter::vjp`, the max/min branch derives gather slice sizes from the
cotangent shape, then forces every indexed axis to length one:

```cpp
auto slice_sizes = cotangents[0].shape();
for (auto ax : axes_) {
  slice_sizes[ax] = 1;
}
```

The forward operation supports blocks. The backward gather must use those
block extents too. In this example, the indexed extent is two, not one.
The smaller gather samples only block starts and broadcasts them across
the update shape, causing the final winning element to lose its contribution.

The [minimal patch](scatter-block-extent-only.patch) changes only that shape:

```cpp
auto slice_sizes = Shape(
    updates.shape().end() - values.ndim(), updates.shape().end());
```

It takes the trailing dimensions of the update tensor corresponding to the
source rank. It adds no new array operations and leaves equality handling
unchanged. No performance measurement or full-suite compatibility claim is made.

The patch includes exact source line locations against
[the inspected commit](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp).

## Actual test results

| Native variant | Scenarios passed | Checks | Failed checks |
|---|---:|---:|---:|
| Baseline, strict blocks only | 0/2 | 26 | 12 |
| Minimal block-extent patch, strict blocks only | 2/2 | 26 | 0 |
| Baseline, extended selection | 10/40 | 388 | 176 |
| Extended block-and-tie patch | 40/40 | 388 | 0 |

[Baseline strict-block log](run-before-strict.log)
· [Minimal-patch strict-block log](run-block-only-strict.log)
· [Extended before log](run-before.log) · [Extended after log](run-after.log)

These are targeted local scenarios, **not the complete MLX test suite**.
One vector comparison counts as one check. The 176 failing extended checks
are not 176 independent defects. Some extended checks test a proposed new
tie policy. The minimal patch was rerun only on the two strict block scenarios;
the 40/40 result must not be attributed to it.

The C++ harness tests forward values, VJPs for both arguments and each
argument separately, cotangent linearity, a shared shift and its second
derivative. Coordinate finite differences are used only for strict block
cases. At ties, the finite-difference reference is the smooth shared shift.
Main tolerance: `2e-6*(1+abs(reference))`; finite differences use h=1/256
and `1e-4*(1+abs(reference))`. All test inputs are small finite FP32 values.

## Equal values: a separate, historically tested behavior

On the official MLX 0.32.2 wheel, this composition is the identity for every
finite scalar x, yet its computed derivative is 5:

```python
import mlx.core as mx
mx.set_default_device(mx.cpu)
def f(x):
    idx = mx.array([0, 0, 0, 0])
    return x.reshape(1).at[idx].maximum(mx.broadcast_to(x, (4,))).sum()
print(mx.grad(f)(mx.array(2.0)))  # observed 5; identity derivative is 1
```

The analogous minimum composition behaves the same way. With `f(x)**2/2`
at x=2, the observed first derivative is 10 rather than 2, and the second is
25 rather than 1. Ordinary binary `maximum(x,x)` and `minimum(x,x)` controls
return the expected derivative 1. See [wheel results](probe-results.json).

For a finite max/min, valid winner weights sum to one. Sending the full
cotangent to the source and each tied update violates that conservation
property for the smooth shared-input identity. The issue is broader than
choosing one particular subgradient at an isolated kink.

**This equality behavior is not presented as a new unknown discovery.**
[PR #431](https://github.com/ml-explore/mlx/pull/431), merged in January 2024,
already tested propagation to both a tied source and update. Its author
[explicitly asked about that choice](https://github.com/ml-explore/mlx/pull/431#issuecomment-1886993971).
That history matters for duplicate assessment and compatibility.

The [extended experimental patch](scatter-extrema-vjp.patch) also normalizes
tie contributions: a tied winning update takes priority over the source;
multiple winning updates divide the cotangent equally. `SliceUpdate` uses
a strict source comparison within the updated slice. This is **one possible
policy**, not the only mathematically valid convention. Maintainer agreement
and compatibility review would be required before changing historical behavior.
It adds winner-count arrays and extra scatter/gather work; performance was
not measured. The simpler block-extent repair does not require that change.

## Source versions and publication verification

| Component | Recorded version or method |
|---|---|
| Official wheel used for equality cases | MLX 0.32.2, CPU float32 |
| Inspected main source | `24c699ecee2f7c8b2040de8da1c8382c8bcf31c7` |
| Compatible native base | `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe` |
| Compiler | Apple clang 17.0.0, C++20 |
| Native method | Recompile original/patched `primitives.cpp`; link before an existing CPU-only `libmlx.a` |

The audited `Scatter::vjp` and `SliceUpdate::vjp` methods are byte-identical
between the compatible base and inspected main. **This was a partial rebuild,
not a clean full build of current main.** The main commit was rechecked before
publication. Source and archive hashes are retained in [metadata](source-metadata.json).

All 38 supplied manifest entries matched. The C++ harness and all three
translation-unit variants were rebuilt and executed again, sequentially,
for publication. The official wheel output is retained from the supplied
audit; it was not rerun for this publication. Original audit files, the
installed wheel, and the existing MLX checkout were not edited.

## Reproduce

Use a compatible Apple-silicon Mac and the recorded CPU build:

```sh
export MLX_SOURCE_ROOT=/absolute/path/to/mlx-at-ce916db
export MLX_CPU_BUILD=/absolute/path/to/cpu-build
python3 build_and_test.py
# The baseline executables intentionally return 1; repaired selections return 0.
./native-before --strict-blocks
./native-block-only --strict-blocks
```

[BUILD.md](BUILD.md) documents preparation of an analogous CPU archive and
the limits of that recipe. [build-results.json](build-results.json) preserves
actual commands and exit codes with machine-specific paths replaced by
placeholders. The public runner changes filesystem references only; it uses
the saved baseline snapshot and environment-configured build directories.
Thread environment variables are set to one; no physical-core affinity is claimed.

To rerun the optional Python equality probe:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python probe.py
```

## Limits and attribution

Not validated: GPU, FP16/BF16, complex or non-finite inputs, every compiled
or vectorized execution path, huge duplicate-index counts, performance,
or real-model training. Passing these checks does not prove correctness for
all shapes and states. Maintainer acceptance, novelty of the strict-block
case and device/model impact are unestablished.

The public search did not identify an exact strict-block duplicate in its
recorded selection; it cannot exclude private or differently worded reports.
No security impact or entitlement to compensation is asserted.

Prepared with AI assistance. The numerical evidence comes from actual local
execution and explicit mathematical and finite-difference references. This
is independent GERO Research work, not an Apple-endorsed audit. Included MLX
source retains its MIT license.

#MLX #Autodiff #NumericalComputing #SoftwareTesting #OpenSource
