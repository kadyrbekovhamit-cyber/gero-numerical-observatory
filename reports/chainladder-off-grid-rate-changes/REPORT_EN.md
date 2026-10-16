## A rate change can disappear between monthly grid points

In chainladder's default monthly approximation, an incremental rate change dated in the middle of a month can be discarded completely. A +20% rate change on 15 July 2001 leaves every output factor at 1.0 in a small public-API example. For the wholly earlier year 2000, the independently expected factor is exactly 1.2: current rate 1.20 divided by old rate 1.00.

A focused fix and tests are submitted in [PR #1491](https://github.com/casact/chainladder-python/pull/1491), with [issue #1492](https://github.com/casact/chainladder-python/issues/1492). Acceptance and an official corrected release are not claimed.

## Reproduce through the public API

```python
import chainladder as cl
factors = cl.parallelogram_olf(
    values=[0.20], dates=["2001-07-15"],
    start_date="2000-01-01", end_date="2003-12-31", grain="Y",
)
print(factors)
# Original: the OLF column is 1.0 in every year.
# Proposed patch: the factor for 2000 is 1.2.
```

The test is pinned to main `90c7b3532499b662e57b5be2275a848ce90d5a70`. It contains synthetic data, not a live insurance portfolio.

The reference for 2000 needs no second implementation of the parallelogram algorithm. All the relevant premium was earned at the old rate. Bringing it to a rate 20% higher requires multiplying by 1.2. A factor of 1.0 fails that simple limiting case.

## Cause and proposed correction

[The incremental branch](https://github.com/casact/chainladder-python/blob/90c7b3532499b662e57b5be2275a848ce90d5a70/chainladder/utils/utility_functions.py) reindexes rate changes to a grid of month starts with exact-date matching. July 15 is absent from that grid, so the rate increase is replaced by zero rather than carried into the next grid point.

The patch sorts the actual effective dates, compounds incremental changes on those dates, then uses the last already-effective rate level at each grid point. The level before the first change is 1. This matches the as-of convention already used for cumulative histories.

The monthly approximation remains an approximation: in this example, July 15 first changes the August 1 grid value. The patch does **not** claim to make monthly and daily geometry identical. Use the daily mode where that timing resolution is required. The published reproduction includes both modes and both original/patched results.

For affected historical periods, dropping a positive increase understates the multiplier used to express old premiums at current rates. The effect on a real actuarial conclusion depends on the portfolio and model. No measured reserve error, customer loss or industry-wide exposure is established here.

## Verification

Two regression tests fail on the unchanged source. With the proposed patch, the selected suite has **126 passed and 1 skipped**. It comprises the new off-grid tests, the existing parallelogram tests and utility tests. The full project suite was not run; dependency warnings are present.

The added tests cover the exact 1.2 old-year reference, equality with the next monthly grid date, and agreement between equivalent incremental/cumulative histories through public `ParallelogramOLF` for monthly/daily grain and both vertical-line modes.

## Related work and contribution status

The 17 October 2026 public prior-art review included issue/PR searches, relevant comments and diffs, and recent file history. [PR #1181](https://github.com/casact/chainladder-python/pull/1181) introduced cumulative factors with as-of matching, while retaining exact reindexing in the incremental branch. #524 concerns origin-grain/leap-year handling; #686 concerns policy length. No exact matching incremental-date fix was found within this bounded search; global novelty is not guaranteed.

The upstream governance requires human review and a human-written PR form. This submission was sent at the owner's explicit request with AI assistance fully disclosed. Those human-review/authorship conditions have not been independently established, and the corresponding declarations were intentionally left unchecked. The maintainers may request a different process or decline the contribution. Filing the PR does not establish their acceptance.

## Attribution and reuse

Investigation and report: Xamit Kadirbekov and Daniyal Kadirbekov / GERO, with AI assistance in investigation, code and writing. Report CC BY 4.0; original reproduction MIT; chainladder source and submitted changes retain MPL-2.0 terms. Credit for chainladder itself belongs to the chainladder-python development team.
