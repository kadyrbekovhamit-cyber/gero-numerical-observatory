# When Dropout sends two gradients to one input

9 September 2026 · Native C++ reproduction and proposed local repair

At a zero drop rate, nntrainer's `DropOutLayer` forwards both tested inputs
correctly but routes every input gradient to input zero. This is a bounded
layer-level correctness result at commit
`a7ea056e79ab8e14447ea305c1b634e233343258`.

| Input | Incoming output gradient | Original input gradient | Patched input gradient |
|---|---|---|---|
| 0 | `[1,2,3]` | `[10,20,30]` | `[1,2,3]` |
| 1 | `[10,20,30]` | `[0,0,0]` | `[10,20,30]` |

Both gradient destinations are initialized to zero. For zero-rate Dropout,
`y_i = x_i` and therefore `dx_i = dy_i`. Inside the loop over input `i`, the
source instead asks for `getOutgoingDerivative(SINGLE_INOUT_IDX)`, where the
constant is zero. Changing the destination to `getOutgoingDerivative(i)`
and removing the unused constant repairs the tested behavior.

## Evidence and measured results

- Original source plus four added tests: **16/19 pass, 3 fail**.
- Locally patched source: **19/19 pass**.
- The 15 pre-existing Dropout tests passed before and after; a new
  single-input control also passed in both versions.
- The three failing added checks cover two-input routing, central finite
  differences of the actual forward pass, and three-input saved-mask replay.
- Before publication, the final formatted test code was rerun through the
  existing native binary: four deterministic checks and the complete
  19-test selection passed. The existing training golden case is stochastic;
  the four added checks are deterministic.

[Self-contained build recipe](BUILD.md) ·
[Deterministic driver](evidence/reproduce.py) ·
[Source patch](evidence/patches/source.patch) ·
[Test patch](evidence/patches/tests.patch) ·
[Original measured results](evidence/results.json) ·
[Publication-day 19-test log](prepublication-check/full-runtime-authorized.log)

## Scope and provenance

CPU FP32, macOS 15.5 arm64, separately allocated native layer buffers.
The existing configured build includes earlier unrelated repairs. The
baseline Dropout file matches the pinned revision byte-for-byte; the
supplied patches change only Dropout source/tests. A fresh clean build,
in-place full-graph allocation, end-to-end training, FP16 and GPU were not
validated. Impact on complete models or Samsung devices is not established.

The constant-zero destination is already present in the public historical
[PR #1640](https://github.com/nntrainer/nntrainer/pull/1640). This case study
does not claim first discovery, absence of private duplicates, maintainer
confirmation, a merged fix, a security vulnerability or bounty eligibility.

`evidence/` preserves the original audit files and their checksums; its
"not submitted" notes describe the retained research state. Use the
self-contained `BUILD.md` above for the publication's complete recipe.
See [the recorded source/build provenance](evidence/provenance.json),
[duplicate-review limits](evidence/DUPLICATES.md) and
[publication-day summary](prepublication-check/full-summary.json).

The practical check is reusable: give independent inputs deliberately
different output gradients and verify each destination. Independent
reproduction and narrowly scoped numerical-correctness reviews are welcome.

Prepared with AI assistance. Numerical claims are grounded in retained
native executions; the finite-difference reference is algorithmically
independent, not an external laboratory review.

Tags: #MachineLearning #SoftwareTesting #Autodiff #NumericalComputing #nntrainer
