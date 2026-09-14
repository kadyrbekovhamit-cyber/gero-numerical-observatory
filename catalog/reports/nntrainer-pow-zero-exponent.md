> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/nntrainer-pow-zero-exponent.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# A constant forward, a NaN gradient: nntrainer PowLayer at exponent zero

Independent GERO research by Xamit Kadirbekov. 13 September 2026.
AI-assisted analysis, test preparation and writing. Synthetic inputs only.
**Public edition. The frozen evidence ZIP retains its original preparation-stage notes. No upstream submission or acceptance is claimed.**

`PowLayer` accepts a fixed zero exponent and returns `1` for the tested finite
inputs, including zero. Its backward pass nevertheless returns `NaN` at zero
and sufficiently small nonzero inputs. A fresh build of the actual nntrainer
C++ layer reproduces the mismatch with its own forward calculation.

Tested source: [nntrainer main, a7ea056e79ab8e14447ea305c1b634e233343258](https://github.com/nntrainer/nntrainer/tree/a7ea056e79ab8e14447ea305c1b634e233343258).
The public main reference still pointed to that commit when checked for this
audit. Meson identifies it as development version 0.6.0; no separate released
binary or Samsung device was tested.

## Minimal result

| Input | Exponent | Incoming gradient | Actual forward | Original backward | Candidate backward |
|---|---:|---:|---:|---:|---:|
| `0` | `0` | `3` | `1` | `NaN` | `0` |
| minimum positive FP32 subnormal, about `1.4013e-45` | `0` | `1` | `1` | `NaN` | `0` |
| `2` | `0` | `3` | `1` | `0` | `0` |
| `0` | `1` | `3` | `0` | `3` | `3` |

The harness constructs `PowLayer`, `InitLayerContext`, `RunLayerContext` and
`Var_Grad` objects, then calls the production `forwarding` and `calcDerivative`
methods. These are native library results, not a reimplementation of the
algorithm in Python. [Runtime harness](https://github.com/kadyrbekovhamit-cyber/gero-nntrainer-pow-audit/blob/main/frozen-evidence/evidence/pow_audit.cpp).

## The invariant comes from the implemented forward

For a fixed exponent of zero, the tested forward implements the constant
function `f(x)=1`, including its chosen value at `x=0`. The input derivative
must therefore be zero. With incoming gradient `g`, the vector-Jacobian
product is also zero.

This can be checked without importing another library's conventions. The
actual C++ forward at `h=1/32`, `-h` and zero returns `1`. Its weighted central
difference is exactly

```text
g * (f(h) - f(-h)) / (2*h) = 3 * (1 - 1) / (2/32) = 0.
```

Original backward at zero returns `NaN`. The finite difference and the
constant-function identity agree on zero. The nonzero-subnormal case provides
a second example that does not involve the value of a power at a zero base.

## Cause and candidate correction

The [current backward expression](https://github.com/nntrainer/nntrainer/blob/a7ea056e79ab8e14447ea305c1b634e233343258/nntrainer/layers/pow_layer.cpp#L34)
evaluates `g * exponent * x^(exponent-1)`. At exponent zero it still forms
`x^-1`. At zero, or where that reciprocal overflows FP32, the subsequent
zero-times-infinity operation produces `NaN`.

The candidate handles the constant case before evaluating the reciprocal:

```cpp
if (exp == 0.0f) {
  context.getOutgoingDerivative(SINGLE_INOUT_IDX).setZero();
  return;
}
```

This handles both signs of zero in the exponent. It uses the Tensor API to
overwrite the outgoing derivative buffer and leaves the existing expression
for nonzero exponents in place. Forward is unchanged.
[Source patch](https://github.com/kadyrbekovhamit-cyber/gero-nntrainer-pow-audit/blob/main/frozen-evidence/evidence/source.patch) · [source plus tests](https://github.com/kadyrbekovhamit-cyber/gero-nntrainer-pow-audit/blob/main/frozen-evidence/evidence/candidate.patch).

## What was actually verified

| Check | Original clean build | Candidate | Original expression restored |
|---|---:|---:|---:|
| 360 zero-exponent coordinate cases | 180 failures | 0 failures | 180 failures |
| 25 ordinary nonzero-exponent controls | 25 pass | 25 pass | 25 pass |
| All 22 existing Pow semantics tests + 3 new regressions | 23/25 pass | 25/25 pass | 23/25 pass |
| Forward finite difference at zero versus backward | `0` versus `NaN` | `0` versus `0` | `0` versus `NaN` |

The 360 zero-exponent cases cover 12 finite FP32 input values, two signed-zero
exponents, five incoming derivatives and three tensor shapes. Shapes are
`[1,1,1,12]`, `[2,1,2,3]` and `[3,2,1,2]`, with identical flattened inputs.
The shapes agree with each other both before and after the repair; this case
does not demonstrate a shape-dependent defect. They test that the repair
works across those layouts.

Inputs include both signs of zero, minimum subnormals, one quarter of the
minimum normal, minimum normals, ±2 and maximum finite values. The baseline's
180 failures are all `NaN` derivatives in the first six values of this grid.
Every forward value is `1`. The 25 ordinary controls use exact dyadic expected
values for identity, square, cube, square root and reciprocal cases.

Every vector invocation repeats backward after replacing the outgoing buffer
with a nonzero sentinel. It checks preservation of input and incoming-gradient
bytes, including signed zeros. Three fresh processes before and three after
the repair produced identical recorded results within each version.

For the mutation check, only the production PowLayer source was restored to
the original expression. After rebuilding, the same 180 coordinate failures
and the same two native test failures returned. The candidate was then
restored, rebuilt and validated again. Its final results match the earlier
candidate results. [Fixed validation protocol](https://github.com/kadyrbekovhamit-cyber/gero-nntrainer-pow-audit/blob/main/frozen-evidence/evidence/VALIDATION_PROTOCOL.md).

These are **385 coordinate cases and 25 native tests**, not 385 bugs or users.
The extra finite-difference experiment is reported separately. Repeated runs
and repeated backward calls are not added to the coordinate-case count.

## Build provenance and limits

The audit used a fresh isolated checkout of the pinned commit. Before the
candidate change, no production source differed from that commit; only the
three regression tests were added. This excludes the unrelated prior local
repairs present in the older exploratory build.

The measured configuration is macOS arm64, CPU FP32, contiguous NCHW tensors,
Apple clang 17, Meson 1.12.0, Ninja 1.13.2 and clang-format 14.0.6. BLAS and
the thread backend are disabled; NNTrainer has one numerical thread and Ninja
uses one build job. The two Darwin compatibility headers only supply platform
system includes. Source, library hashes, commands, complete build logs and
GoogleTest XML outputs are retained in the evidence directory.

No FP16, GPU, Android/Tizen device, non-contiguous tensor, multiple-input layer,
in-place inference, nonfinite upstream derivative, complete project test suite,
performance benchmark or production-model impact was evaluated. The test
establishes an invalid derivative at the layer boundary. It does not establish
how often deployed applications reach this configuration or what training
quality changes would result.

## Prior reports and duplicate screening

Five saved GitHub issue/PR queries returned 35 distinct records. No direct
duplicate was found in that bounded review. The introducing [PR #2801](https://github.com/nntrainer/nntrainer/pull/2801)
contains the same backward expression; its original tests use exponent 3.
The author's earlier [five-unary-layer issue #4326](https://github.com/nntrainer/nntrainer/issues/4326)
concerns discarded chain-rule operations in different layers. That report
does not describe this PowLayer zero-exponent case.

The existing GERO catalogue contains twelve nntrainer-related articles or
overviews. Those older results remain separate from this new report.
[Duplicate review](https://github.com/kadyrbekovhamit-cyber/gero-nntrainer-pow-audit/blob/main/frozen-evidence/evidence/DUPLICATE_REVIEW.md) ·
[Russian inventory and editorial notes](https://github.com/kadyrbekovhamit-cyber/gero-nntrainer-pow-audit/blob/main/frozen-evidence/NNTRAINER_REVIEW_RU.md).

The candidate has not been submitted or accepted upstream. Immediately before this publication, main and the four focused duplicate queries were rechecked at 2026-09-13 16:19 UTC: main and the audited PowLayer source are unchanged, and the query results contain no new direct match. See [prepublication check](https://github.com/kadyrbekovhamit-cyber/gero-nntrainer-pow-audit/blob/main/prepublication-recheck.json).

## Reproduce

Use the [build and execution recipe](https://github.com/kadyrbekovhamit-cyber/gero-nntrainer-pow-audit/blob/main/frozen-evidence/REPRODUCE.md). The package includes the
native harness, regression/source patches, frozen relevant source files,
actual before/after outputs, mutation results and public-search records.
Obtaining the full pinned project and its two required submodules requires
network access. Upstream code retains its Apache-2.0 license and attribution.


## Watch and download

[38-second English video](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/nntrainer-pow-zero-exponent.mp4) · [Evidence archive](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/gero-nntrainer-pow-zero-exponent-evidence-2026-09-13.zip) · [Video sources](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/nntrainer-pow-video-source.zip) · [GERO article](https://www.gero.uz/research/articles/nntrainer-pow-zero-exponent.html).
Original graphics and preset synthetic English narration. Original report and graphics: Xamit Kadirbekov / GERO, CC BY 4.0. Upstream source retains Apache-2.0.
