# Evidence contract — v1 and v2

## Current v2

- `freeze-v2.json` / `corpus-v2.json`: candidate/key-component hashes and 288
  new confirmation fixtures, fixed before viewing candidate outputs. The
  complete runner/helper pipeline was not included in the pre-view receipt;
  the final runner hash is retained in `benchmark-v2.json`.
- `benchmark-v2.json`: all confirmation rows, practical gates and comparisons.
  There are 202 nonzero reference prices: v2/Jaeckel closer/farther/equal counts
  146/30/26; v2/v0 counts 192/3/7. All registered practical gates passed.
- `development-v2-final.json`: all 316 old v1 cases are now development data;
  final candidate matches their previously measured price/log values exactly.
- `archive-v2-draft-01`: an unsuccessful development-only quadrature controller,
  retained before it was corrected. Not part of held-out confirmation.
- `python-timing-v2.json`: a small post-accuracy timing of the Python detailed
  APIs, at unequal achieved accuracy. No native C++ timing claim.

V2 uses positive binary64 quadrature; primary oracle is a multiprecision erfc
formula. The supplemental multiprecision integral shares the analytic formula
with v2, so it is not completely independent. New v2 tests are now consumed.
See `../TECHNICAL_NOTE_V2.md` and `../V2_MEASUREMENT_REVIEW.md` for limits.

## Preserved v1

- `corpus-v1.json`: 117 development and 199 newly frozen confirmation cases;
  exact binary64 hex inputs. This confirmation set is now consumed.
- `benchmark-v1.json`: generated rows, summaries, oracle convergence attempts,
  native build provenance, input-mapping errors and contemporaneous code hashes.
- `jackel-build-v1.json`: historical upstream commit, source hashes, compiler
  commands and floating-point environment probe.
- `diagnostics-v1.json`: four explicitly post-hoc checks at 260/360 digits and
  with a 120-digit positive integral. Not a second holdout.
- `archive-v0/`: original code and original evidence, with a manifest.
  The old decimal-input/error-measurement contract is superseded, not erased.
- `archive-v1-initial/`: preserved first run, whose log aggregate included 12
  wrong-side comparisons. The corrected run marks these separately and excludes
  them from same-leg error aggregates. All ordinary price results are unchanged.

The full v1 experiment passed all 316 oracle convergence checks and all 14
predetermined integral cross-checks. Eight development cases needed the
registered precision retry. These are empirical checks, not interval proofs.

There are 217 prices whose converged reference rounds to nonzero binary64,
including four subnormals; 99 round to zero. An exact price below the smallest
positive binary64 does not necessarily round to zero: half-minimum is the
round-to-nearest-even boundary. The reference converter handles this explicitly.

The candidate was not tuned during this experiment. The Jaeckel baseline is an
unchanged historical source from a pinned vollib commit, not verified latest
upstream. End-to-end price error is not isolated kernel error. No runtime,
Greeks, implied-volatility or financial-loss claims follow from this evidence.

See `../TECHNICAL_NOTE_V1.md` for the results and limitations. To reproduce,
choose a fresh output path with `python3 -m lab.run_benchmark_v1 --output ...`.
