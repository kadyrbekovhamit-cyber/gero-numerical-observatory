# nntrainer: Dropout writes every input gradient to input zero

Local audit, 9 September 2026. **Reproduced and locally repaired.**
No report, PR or commercial proposal was sent during this work.

Tested revision: `a7ea056e79ab8e14447ea305c1b634e233343258` (development
version 0.6.0 reported by Meson). Platform: macOS 15.5 arm64, CPU FP32,
existing configured build. The baseline `dropout.cpp` matched this
revision byte-for-byte; unrelated earlier local repairs remain in the
build and are recorded in `provenance.json`.

## Reproduction

The actual `DropOutLayer` accepts two input dimensions and produces two
outputs in its `finalize`/`forwarding` path. With `dropout_rate=0`, forward
copies each input unchanged. Consequently, each output's incoming
gradient must be routed to the corresponding input.

| Tensor | Expected | Original runtime | Patched runtime |
|---|---|---|---|
| First input gradient | `[1,2,3]` | `[10,20,30]` | `[1,2,3]` |
| Second input gradient | `[10,20,30]` | `[0,0,0]` | `[10,20,30]` |

Input gradients were explicitly initialized to zero. Both forward
outputs were correct. Input values and incoming gradients remained
unchanged. Central differences of the real forward pass, using a mixed
signed objective and a step of `1/32`, independently confirmed the
correct derivatives for every coordinate of both inputs.

An additional three-input test uses the existing `reStoreData(true)`
path with different saved masks at rate `0.5`. This exercises the
masked backward branch deterministically. It is a replay-path test,
not a statistical test of mask generation.

## Cause and repair

Inside the loop over input index `i`, `calcDerivative` requests
`getOutgoingDerivative(SINGLE_INOUT_IDX)` where that constant is zero.
Later iterations overwrite the first input's gradient; subsequent
input gradients are not written. Change the destination to
`getOutgoingDerivative(i)` and remove the now-unused constant.

[Source patch](patches/source.patch) and [test patch](patches/tests.patch)
are limited to Dropout. They exclude the other local audit changes.

## Measured validation

| Run | Tests | Failures |
|---|---:|---:|
| Original source + new tests | 19 | 3 |
| Fixed source + identical tests | 19 | 0 |

Four new tests: a one-input control, two-input gradient routing, actual
runtime finite differences, and a three-input saved-mask replay. The
control passes before and after the repair; the other three fail before
and pass after it. The remaining 15 tests are existing Dropout semantics
and golden tests; all passed in both measured runs.

Evidence: [before.log](before.log), [after.log](after.log), machine-readable
`before.xml`, `after.xml`, and [results.json](results.json).
[BUILD.md](BUILD.md) describes reproduction. `reproduce.py` runs only
the four deterministic new tests; the 19-test runs above are separate
saved measurements.

After these measurements, clang-format changed only the helper's
signature wrapping. `tests-first-runs.patch` preserves the test source
used in the 19-test runs. The final `patches/tests.patch` was rebuilt and
verified by the four-test driver (`driver-verification/summary.json`).

## Boundaries

- Reproduced through real C++ layer contexts, not through a full training
  graph. In-place graph allocation and end-to-end effects have not been
  validated for this configuration. This is a layer-level correctness bug.
- No claim of FP16, GPU, other platforms, fresh clean-build validation,
  security impact, exploitation, bounty eligibility or accepted upstream fix.
- Existing golden tests cover a single input. The added tests use separately
  allocated input/output gradients and equal-shaped multiple inputs.
- Public duplicate checking is limited; see [DUPLICATES.md](DUPLICATES.md).
  Original discovery and absence of duplicate reports are not established.
- Build used `ninja -j1`, one NNTrainer thread, no BLAS and no thread backend.
