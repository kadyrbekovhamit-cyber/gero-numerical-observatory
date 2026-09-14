> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/dinero-js-safe-intermediate-overflow.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# Dinero.js misallocates safe integers after an unsafe intermediate product

Xamit Kadirbekov · Originally published 2026-09-06

All inputs remain JavaScript safe integers, yet binary64 rounding changes a 3-unit allocation from the exact [2, 1, 0] to [3, 0, 0].

[Original GERO article](https://www.gero.uz/research/articles/dinero-js-safe-intermediate-overflow.html) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/dinero-js-safe-intermediate-overflow) · [Original archived collection](https://doi.org/10.5281/zenodo.22683900) · [Individual Zenodo record](https://doi.org/10.5281/zenodo.22728991)

This page presents an existing document from the original 45-document archive. It is not a new finding or a new experiment. The archived corpus text below is preserved verbatim; the original article retains its layout, references and downloadable evidence. Publication dates, scientific claims, limitations and retraction status remain those of the source.

Text SHA-256: `a4e79808538c2af5f13dc86dca1789d3a6134808c68ca93be91cede6559b892c`

## Original archived text

<!-- BEGIN UNCHANGED CORPUS TEXT -->
Dinero.js misallocates safe integers after an unsafe intermediate product
https://www.gero.uz/research/articles/dinero-js-safe-intermediate-overflow.html

← Research index

INDEPENDENT NUMERICAL AUDIT

6 September 2026

Dinero.js misallocates safe integers after an unsafe intermediate product

All inputs remain JavaScript safe integers, yet binary64 rounding changes a 3-unit allocation from the exact [2, 1, 0] to [3, 0, 0].

Xamit Kadirbekov

Independent verification · GERO Research

Numerical audit

Dinero.js

Allocation

STATUS · VERIFIED ORDINARY CORRECTNESS DEFECT

Reproduced on Dinero.js 2.0.2 and current main; no production application impact is claimed.

UPSTREAM RECORD

Public reproducer on GitHub ↗

Finding

Dinero.js

2.0.2

can return a mathematically incorrect allocation with its default

number

calculator even when the amount, every ratio, the ratio sum, and all returned shares are JavaScript safe integers. The failure occurs when an intermediate multiplication crosses the safe-integer boundary and rounds across an integer-division boundary.

Minimal counterexample

amount = 3
ratios = [3002399751580333, 1000000000000000, 501199875790167]
ratio sum = 4503599627370500

actual:   [3, 0, 0]
expected: [2, 1, 0]

For

amount = -3

, the actual result is

[-3, 0, 0]

; the exact result is

[-2, -1, 0]

.

Exact oracle

For the largest ratio:

3 × 3002399751580333 = 9007199254740999
                    = 2 × 4503599627370500 − 1

The exact quotient is therefore

1

. JavaScript

number

rounds the intermediate product to

9007199254741000

, making it exactly twice the ratio sum. Dinero.js consequently begins with a share of

2

and then assigns the remaining unit to the same largest ratio, returning

3

.

The independent oracle performs the products and integer division with

BigInt

, followed by the same largest-ratio remainder policy.

Verification

Reproduced with the published package

dinero.js@2.0.2

.

Confirmed on upstream

main

commit

76b969e519dc44675d4af898d25629995d0b16f2

; its

distribute.ts

is identical to the release implementation.

Positive and negative counterexamples both reproduce.

Each input ratio and their sum passes

Number.isSafeInteger()

.

Targeted searches found no duplicate. Issues

#771

and

#776

cover different failure modes.

Proposed correction

The behavior-preserving solution is a combined exact integer multiply-divide operation. The default

number

calculator can promote safe integer operands to

BigInt

for

(amount × ratio) / total

, then convert the bounded quotient back to

number

. The

bigint

calculator can evaluate the expression directly.

A smaller defensive correction is to detect an unsafe intermediate product and throw a clear precision error that directs callers to

dinero.js/bigint

. Silently returning a different allocation should not be the fallback.

Upstream status

Reported as

dinerojs/dinero.js#892

on 9 September 2026. An issue was used instead of a pull request because a behavior-preserving fix requires extending the public calculator abstraction with an exact combined multiply-divide operation; the narrow alternative is a documented precision error directing number users to the bigint calculator.

Boundary

This is an ordinary numerical-correctness defect, not a security vulnerability. No affected production application or customer loss has been identified. The result demonstrates a library-level contract failure under valid safe-integer inputs.

<!-- END UNCHANGED CORPUS TEXT -->

Original report text: CC BY 4.0; code and third-party material retain their original licenses. AI-assisted archival presentation.
