# Frozen hybrid v3 experiment — 26 September 2026

This protocol precedes the first evaluation of the new confirmation inputs.
All 316 v1 and 288 v2 cases, plus unit fixtures, are development data. The first
v3 development run passes their cached-reference mixed gates; it is not new
confirmation evidence. Do not modify the candidate after reading confirmation
results. Preserve a failure as a failure and assign a later version to repairs.

## Hypothesis and scope

A binary64 hybrid can reduce the cost of the frozen v2 positive-payoff
integrator while preserving its practical OTM price/log gates on a new bounded
synthetic corpus. This does not mean equal ULP accuracy, universal correct
rounding, market-fit improvement, a faster native implementation, or novelty.

The candidate has exact-input ATM, small-v moment, tail-difference series,
conditioned direct CDF, and unchanged v2 fallback paths. Analytic Taylor
remainder arguments apply to exact-arithmetic truncation only. The direct
score is an engineering heuristic. No computed component score bounds total
floating-point error; no reference-oracle calls are in the candidate.

## Confirmation corpus and comparison

Seed 2026092603; 480 exact binary64 hex tuples, generated without pricing:
96 small-v, 80 negative-tail, 80 regular-with-carry, 96 switching-neighborhood,
80 monetary-range (including rounded-zero/subnormal thresholds), 48 exact-ATM.
Construction is specified by `lab/corpus_v3.py`. Deduplicate against both old
input corpora and within this corpus by nextafter(sigma,+inf), recording count.
These are stress strata, not an estimate of production input frequencies.

Primary reference is the unchanged erfc formula with exact-binary inputs at
100/180 digits and, if its convergence test requires, 260. Required relative
price stability 1e-60 and absolute log stability 1e-55. No failed reference is
silently removed. Fixed supplementary positive-integral checks at indices
0,47,95,135,175,215,255,303,351,391,431,479, 100 digits, absolute log agreement
1e-50. This integral is analytically related to candidate expansions/fallback;
it is not an independent mathematical representation of those paths.

Baselines: unchanged own v0, frozen v2, and unchanged historical native Jaeckel
(mirror commit 83ae882df8e19323798c7ebfb8898f94d2d92ade). Comparison to Jaeckel
is end-to-end through a binary64 forward/discount adapter. Record and audit
adapter distortion on nonzero-carry inputs. Latest official upstream has not
been verified. Preserve licensing/attribution. No native timing claim.

For each case, require correct OTM side and finite nonnegative call/put; no
unexpected zero when rounded reference is nonzero; OTM price absolute error
<= max(1e-10*reference, 1 ULP at rounded reference); OTM log absolute error
<= max(2e-11, 16 ULP at rounded reference log). These mixed tolerances do not
promise 1-ULP prices or 16-ULP logs. Record v2 and v3 gates and all comparisons,
including ties, losses, zero-rounding cases, exceptions, worst errors, branch
counts, and per-family results. Tests also check parity, monotonicity, scaling,
reciprocal symmetry, branch seams, and separately the tiny-volatility ATM path.

## Freeze and reproducibility

Before the first new pricing output, hash the corpus, all local lab Python
modules and tests, this protocol, native build receipt, all pinned native
sources and binary, old corpora consumed by deduplication, and the one-worker
wrapper. Record Python/mpmath/platform and thread environment. Hash the installed
mpmath .py tree as dependency provenance. The runner verifies the source/corpus
and dependency hashes. This is a local timestamped freeze, not external
preregistration; system libm/interpreter/platform binaries are identified by
environment rather than exhaustively hashed. Vendored source validates the
current native build receipt; a build reproduced elsewhere will differ.

## Timing, fixed before outputs

Only after saving accuracy evidence, time v0/v2/v3 detailed Python APIs on
all 480 frozen confirmation inputs, separately on each family, and separately
on the subset using v2 fallback. The fallback subset is descriptive selection
by the algorithm and must be reported as such. One warm-up, 7 alternating
method-order batches, 5 sequential loops per batch; report all raw batch times
and median ns/call, ratios of medians. No multiprocessing/GPU. No significance,
production-throughput, market-frequency or equal-ULP-accuracy claims. Report
fallback overhead even if it loses. Do not tune after timing. An all-case
speedup above 2x versus v2, together with all accuracy gates passing, is the
predeclared practical success criterion; every other result remains reportable.
