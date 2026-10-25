# SciPy exact-test boundaries: floating-point ties can change p-values

Xamit Kadirbekov and Daniyal Kadirbekov · 25 October 2026 · GERO Research

**Status: reproduced in a bounded local environment; proposed fix in [SciPy PR #26496](https://github.com/scipy/scipy/pull/26496). The PR is open, not accepted, merged or released as of this report.**

## What we found

Two routines can omit outcomes whose test statistic is mathematically equal to the observed one because floating-point evaluation puts them on different sides of a strict comparison. In the examples below, this changes which side of a conventional 0.05 threshold the computed p-value falls on. It does not establish a real treatment effect, actual clinical harm, or the reliability of an entire research study.

| Synthetic example | Unpatched result | Patched result / reference |
|---|---|---|
| `poisson_means_test(7, n, 2, n, alternative="greater")`, `diff=0`, with n=1 then n=50 | approximately 0.051542 then 0.042278 | approximately 0.051542 at both scales |
| Two-sided `barnard_exact([[3,3],[0,6]], pooled=True)`, then reverse rows | approximately 0.044434 then 0.064941 | approximately 0.064941 in both orderings |
| Two-sided `barnard_exact([[5,0],[1,4]], pooled=False)` | 0.0078125 | 7/512 = 0.013671875 |

The Poisson common-exposure rescaling statement above is for the zero-difference null used in the example. We do not assert that rescaling exposures leaves every nonzero `diff` problem unchanged. For two-sided Barnard tests, reversing success/failure rows or the group columns should preserve the p-value.

## Mechanism and proposed correction

`poisson_means_test` computes the observed statistic and the enumerated outcome statistics along different arithmetic paths. Expressions such as `k/n**2` and `(k/n)/n` need not round identically. A tied outcome, even the observed outcome itself, can then be left out of the probability sum.

`barnard_exact` similarly constructs a rejection region with comparisons of floating-point Wald statistics. Exact ties can be excluded after rounding.

The proposed patch expands each comparison boundary by `100 * float64_epsilon * abs(observed_statistic)`. For a nonfinite Barnard observed statistic the tolerance is zero, avoiding an infinity-minus-infinity operation. The factor 100 is a practical numerical choice, **not a universal proved error bound**. Maintainer review may change this implementation.

This is not a claim of first discovery of the Barnard issue. [SciPy issue #18798](https://github.com/scipy/scipy/issues/18798) is relevant prior work, cited in the patch and PR. Our contribution here is the proposed change, additional regression cases, and separate reference calculations covering these two routines.

## Separate reference calculations

For the unpooled table `[[5,0],[1,4]]`, enumerate column-success counts `(a,b)` for column sizes 6 and 4 using exact rational arithmetic. The two-sided region contains `(0,4), (1,4), (5,0), (6,0)`. Its null probability is `q**4 * (1 + 10*q)`, with `q = pi*(1-pi) <= 1/4`. It is increasing in nonnegative q, so its maximum is `14/1024 = 7/512` at `pi=1/2`.

For the one-sided Poisson example, the zero-difference fitted means are both 4.5. A separate double sum with Decimal arithmetic at 80 and 110 digits gives:

```text
0.051541749805736265832452830801363160397838390275144090429507062032142577672060...
```

The sum enumerates counts 0 through 79 in each coordinate. The recorded upper bound on omitted probability is approximately `1.25912e-67`. Agreement at two precisions and the tail bound are numerical cross-checks; they are not an independent human review.

## Verification and limits

- Patch base: `b9be93fce75058df3e6848bfb79e24fd272a06b5`.
- Proposed head: `b2c2bc46d94e8f15f85fcdfe32da66586c6f94f9`.
- 18 targeted regression cases: **12 failed and 6 passed before; all 18 passed after**.
- The affected test module: **259 passed, 3 skipped, 1 failure**. The same `TestMannWhitneyU::test_gh24777[numpy]` failure was reproduced on the unpatched baseline.
- Runs reused an existing SciPy 1.18.1 binary installation with the target Python modules. There was **no fresh local native rebuild and no full SciPy suite run**.
- Scope: synthetic tables, exposure scales and the recorded module tests. No customer data, clinical records or measured financial losses were used.

The archive contains the patch, a small witness script, the separate oracle script and output, sanitized test logs, and hashes. Use an isolated SciPy development environment to apply the patch and run the upstream tests; this archive is evidence, not a replacement SciPy package or installation instruction for production.

## Reproduction

With SciPy available, run `python witness.py` to print the input examples and `python independent_oracles.py` for the reference calculations (the oracle needs only Python's standard library). The patch includes the 18 added/modified target cases. In an appropriately built SciPy checkout, they can be selected from `scipy/stats/tests/test_hypotests.py` using the test names in the patch and saved logs. The saved baseline failure must remain visible when reporting the broader module result.

## Credit, disclosure and reuse

Research/report attribution: **Xamit Kadirbekov and Daniyal Kadirbekov**. Claude and OpenAI Codex assisted the investigation, proposed implementation, regression tests, reference scripts and analysis. OpenAI Codex drafted this report and the PR description; the submitter approved the PR description for posting. This is not a claim of independent human code review or upstream endorsement.

Report: CC BY 4.0. Original reproduction scripts: MIT. The SciPy patch and upstream-derived code retain SciPy's BSD-3-Clause terms; the license is included. The archive contains no private findings from other projects.

## Sources

- [SciPy PR #26496 — proposed fix, tests and current discussion](https://github.com/scipy/scipy/pull/26496)
- [SciPy issue #18798 — prior Barnard exact-test report](https://github.com/scipy/scipy/issues/18798)
- [Pinned source before the patch](https://github.com/scipy/scipy/blob/b9be93fce75058df3e6848bfb79e24fd272a06b5/scipy/stats/_hypotests.py)
- [Pinned proposed source](https://github.com/scipy/scipy/blob/b2c2bc46d94e8f15f85fcdfe32da66586c6f94f9/scipy/stats/_hypotests.py)
