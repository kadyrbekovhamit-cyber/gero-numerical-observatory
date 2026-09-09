# nntrainer average pooling loses gradients with asymmetric padding

9 September 2026. **Reproduced in the native C++ implementation:**
`Pooling2DLayer` computes the correct forward values but skips output windows
during backward when ordinary `padding=same` produces asymmetric padding.

Tested version: [nntrainer main a7ea056](https://github.com/nntrainer/nntrainer/blob/a7ea056e79ab8e14447ea305c1b634e233343258/nntrainer/layers/pooling2d_layer.cpp#L214),
full SHA `a7ea056e79ab8e14447ea305c1b634e233343258`. The GitHub API returned
the same SHA for `main` during verification. Environment: macOS arm64, CPU,
FP32, NCHW.

## Minimal counterexample

Input `[[1,2],[3,4]]`, shape `[1,1,2,2]`, kernel `2x2`, stride `1x1`,
`pooling=average`, `padding=same`, with an upstream gradient of four ones.

| Quantity | Current source | Correct / after repair |
|---|---|---|
| Forward | `[2.5, 3, 3.5, 4]` | `[2.5, 3, 3.5, 4]` |
| Input gradient | `[0.25, 0.25, 0.25, 0.25]` | `[0.25, 0.75, 0.75, 2.25]` |
| Input-gradient sum | `1` | `4` |

The resolved padding is top=0, bottom=1, left=0, right=1. Forward evaluates
four windows containing 4, 2, 2, and 1 valid elements. Backward visits only
the first. The maximum absolute error in this example is 2, and only one
quarter of the required input-gradient sum is retained.

The reference matches nntrainer's own forward convention: padded positions
are excluded from the averaging divisor. The discrepancy therefore cannot be
explained by a different framework convention. Central finite differences of
the **same native C++ forward** with mixed upstream gradient `[1,-2,3,4]`
produce `[0.25,-0.75,1.75,4.75]`; the current backward again produces four
values of `0.25`.

## Root cause and minimal repair

The backward loop ends at `height - pool_height + top` and
`width - pool_width + left`. To cover the same windows as forward, those bounds
must use **bottom and right**. With multiple channels, skipping windows also
misaligns the pointer that reads the saved averaging divisors.

The [minimal source patch](patches/source.patch) changes only these two bounds
and performs the subtraction in signed integers. It does not change forward or
the averaging convention. The [regression patch](patches/tests.patch) exercises
the real layer and compares it with an independent window-membership oracle.

## What was tested

| Same selected test set | Total | Passed | Failed |
|---|---:|---:|---:|
| Current pooling source at the stated commit | 58 | 50 | 8 |
| After the minimal repair | 58 | 58 | 0 |

The 12 new tests include the hand-computed example, finite differences, and
ten shape/padding variants. Six matrix cases reproduce the defect: SAME with
an even kernel, bottom-only padding, right-only padding, stride 2, multiple
batches/channels, and explicit bottom/right padding. Four controls pass both
before and after: symmetric padding, valid padding, global average pooling,
and max pooling with SAME padding.

The matrix also checks linearity in the upstream gradient, zero gradients,
input immutability, upstream-gradient immutability, and gradient sums.
Linearity alone cannot validate an incorrectly transposed operator; an
independent Jacobian oracle is required. The other 46 tests are existing
pooling semantic/property tests, not 46 additional numerical examples.

Evidence: [before log](before-expanded.log), [before XML](before-expanded.xml),
[after log](after.log), [after XML](after.xml),
[machine-readable result](results.json), and [provenance](provenance.json).
The reproduction driver was also run against the repaired code; its output is
in [driver-verification/summary.json](driver-verification/summary.json).

The build used `ninja -j1`, `NNTR_NUM_THREADS=1`, `thread-backend=none`, and
BLAS disabled. A clean rebuild from scratch, other platforms, FP16, and NHWC
were not tested. The existing build contained unrelated earlier repairs, but
the pooling source was byte-identical to the stated commit before this change.
Both patches were applied independently to clean copies of their two target
files and produced byte-for-byte matches with the tested sources; see
[patch-validation.json](patch-validation.json). Clang-format 14 and
`git diff --check` passed on the source change.

## History and evidence boundary

An important novelty correction: old
[PR #1360](https://github.com/nntrainer/nntrainer/pull/1360) already contained
the correct bottom/right backward bounds. This report therefore does **not**
claim the formula or repair as a first discovery. The defect is reproducible
in the stated current source, and the performed public search found no exact
report containing this reproduction. Open NHWC PR #4085 does not repair the
tested NCHW bounds. See [DUPLICATES.md](DUPLICATES.md) for search details and
network limitations.

Reproduction instructions are in [BUILD.md](BUILD.md), the driver is
[reproduce.py](reproduce.py), and an English upstream-report draft is in
[UPSTREAM_REPORT.md](UPSTREAM_REPORT.md). This package documents a confirmed
training-correctness defect. It does not establish impact in a released
Samsung product, a security vulnerability, upstream acceptance, or eligibility
for a monetary reward.

## Separate unconfirmed candidate

`DropOutLayer::calcDerivative` reads the output gradient through constant index
zero while forward and finalize iterate over multiple inputs. This is a strong
candidate for incorrect multi-input gradient routing, but the layer/graph
contract and a native multi-input reproduction still need verification. It is
not included among the confirmed findings in this package.

`PreprocessL2NormLayer` explicitly rejects backward with an exception. That is
a contract limitation, not a newly identified defect.
