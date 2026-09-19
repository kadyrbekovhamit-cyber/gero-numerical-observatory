> **Editorial update — 20 September 2026.** A dated consequences, practical-response and correction note follows the historical archive below. It corrects the PR #509 attribution and qualifies the historical CI and patch-scope statements. The marked original corpus and its original hash remain unchanged.

> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/moneyphp-largest-remainder-float-collapse.md). Claims, dates, authorship and licenses remain those of the original publication; the original archival import did not rerun or revalidate its numerical experiments. Later dated updates are identified separately.

# MoneyPHP loses largest-remainder ordering above float precision

Xamit Kadirbekov · Originally published 2026-09-05

An accepted 7-quadrillion-unit amount makes two exact residuals collapse to zero in binary64, assigning the final unit to the wrong share.

[Original GERO article](https://www.gero.uz/research/articles/moneyphp-largest-remainder-float-collapse.html) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/moneyphp-largest-remainder-float-collapse) · [Original archived collection](https://doi.org/10.5281/zenodo.22683900) · [Individual Zenodo record](https://doi.org/10.5281/zenodo.22729036)

This page presents an existing document from the original 45-document archive. It is not a new finding or a new experiment. The archived corpus text below is preserved verbatim; the original article retains its layout, references and downloadable evidence. Publication dates, scientific claims, limitations and retraction status remain those of the source.

Text SHA-256: `d6a021472e74d9f8967b1cfd9132dd2505ff1f111866b11d7a6bb3594096e5c2`

## Original archived text

<!-- BEGIN UNCHANGED CORPUS TEXT -->
MoneyPHP loses largest-remainder ordering above float precision
https://www.gero.uz/research/articles/moneyphp-largest-remainder-float-collapse.html

← Research index

INDEPENDENT NUMERICAL AUDIT

5 September 2026

MoneyPHP loses largest-remainder ordering above float precision

An accepted 7-quadrillion-unit amount makes two exact residuals collapse to zero in binary64, assigning the final unit to the wrong share.

Xamit Kadirbekov

Independent verification · GERO Research

Numerical audit

MoneyPHP

Allocation

STATUS · VERIFIED ORDINARY CORRECTNESS DEFECT

The total remains conserved; the defect changes which allocation receives the residual minor unit.

UPSTREAM RECORD

moneyphp/money#832 ↗

Finding

Money::allocate()

computes its initial integer shares with exact arithmetic, but converts the amount and ratios to binary

float

when ranking fractional remainders. At sufficiently large accepted amounts, distinct exact remainders collapse to the same floating-point value and the residual minor unit is given to the wrong share.

Minimal counterexample

$parts = (new Money\Money(
    '7000000000000000',
    new Money\Currency('USD')
))->allocate([1, 2]);

Observed amounts:

[2333333333333334, 4666666666666666]

Exact largest-remainder result:

[2333333333333333, 4666666666666667]

The exact fractional remainders are

1/3

and

2/3

. The single residual minor unit must therefore go to the second share. In the implementation both floating-point fractions become

0.0

, so the tie-breaking order gives it to the first share instead.

Reproducer

<?php
require 'vendor/autoload.php';

use Money\Currency;
use Money\Money;

$actual = array_map(
    static fn (Money $part): string => $part->getAmount(),
    (new Money('7000000000000000', new Currency('USD')))->allocate([1, 2]),
);

$expected = ['2333333333333333', '4666666666666667'];
var_export(['actual' => $actual, 'expected' => $expected]);
assert($actual === $expected);

Verification

Reproduced on PHP

8.5.10

with BCMath.

Reproduced at release

v4.9.0

, commit

d49ee625c6ba79b9d7a228ce153b02fc1032152b

.

Reproduced on

master

, commit

a5199443bfd20c3d3644b8445b09054a4c014191

.

The allocated parts still sum to the original amount; the defect is the proportional ranking of the residual unit.

For ratios

[1, 2]

, the first failing region begins when the largest share reaches the binary64 spacing boundary near

2^52

. A scanned failing amount starts at

6755399441055744

minor units; no mismatch was found in sampled windows around

10^14

and

4 * 10^15

.

The counterexample amount

7 * 10^15

is itself below

2^53

and is exactly representable as binary64. The loss occurs when ranking the fractional shares, not when storing the original amount.

Targeted issue and pull-request searches found no exact duplicate. Related issue

#506

and pull request

#509

introduced the largest-remainder policy, but do not report this loss of precision.

A common-scale integer-weight correction and portable regression test were submitted upstream as

moneyphp/money#832

.

Proposed correction

Normalize the ratios to common-scale integer weights, compute each exact residual numerator with MoneyPHP's configured

Calculator

, and compare those numeric strings directly. This removes binary

float

from the remainder ranking while preserving input order for exact ties.

Boundary

This is an ordinary correctness defect in a public library method. It is not evidence that a bank or payment processor transferred an incorrect amount, and no production system was tested. Cases with decimal ratios can also expose floating-point tie-breaking, but a true exact tie does not establish a unique wrong allocation and is intentionally excluded from this finding.

Upstream status

Correction submitted as

moneyphp/money#832

. Following maintainer review, the implementation was reduced from three passes to two and a compatibility regression was added for an ordinary

1:2

allocation. All CI, static-analysis, documentation and benchmark jobs pass. The exact remainder ordering intentionally changes only the large-value case in which binary64 had collapsed distinct

1/3

and

2/3

remainders into a false tie.

<!-- END UNCHANGED CORPUS TEXT -->

Original report text: CC BY 4.0; code and third-party material retain their original licenses. AI-assisted archival presentation.

## Update — 20 September 2026: consequences and practical response

The published counterexample was rechecked with MoneyPHP v4.9.0 (commit `d49ee625c6ba79b9d7a228ce153b02fc1032152b`), PHP 8.5.10 and BCMath. For `7000000000000000` minor units and ratios `[1, 2]`, the observed allocation was `[2333333333333334, 4666666666666666]`; exact largest-remainder allocation is `[2333333333333333, 4666666666666667]`. Both conserve the total. One share receives one minor unit too much and the other one too little. At `7000` units, the same library returned the expected `[2333, 4667]`.

### Engineering assessment: what this could affect

If an application uses an affected allocation to determine recipients' balances, the residual unit could go to the wrong recipient. Depending on the application, that could affect commission splits, revenue distribution or cost allocation. A downstream system using exact remainder ranking could then disagree with the application's per-recipient records, requiring reconciliation or explanation. These are conditional consequences inferred from the demonstrated calculation; they were not observed in a bank, payment processor or customer account.

A check that only verifies `sum(parts) == amount` will pass this example. That creates a testing blind spot: an apparently balanced result can still allocate value incorrectly. Repetition could accumulate recipient-level discrepancies if affected inputs recur, but this study has not measured occurrence rates, cumulative losses or production exposure.

The tested input is exceptionally large. This finding does not establish incorrect rounding of ordinary purchases, widespread customer losses, a security exploit, or failure of other versions and operations. It is a bounded correctness defect in the tested library behavior.

### Practical response

- Add a regression that checks each expected share, alongside conservation of the total.
- Cover ordinary amounts, large precision-boundary cases, reversed ratios and genuine exact ties.
- In the proposed correction, normalize ratios to common-scale integer weights, compute exact remainder numerators with the configured calculator, and preserve input order for exact ties. Avoid binary floating-point ranking of those remainders.
- Reproduce the relevant case in the application's own supported environment and review the proposed patch before adopting it. Do not infer that a submitted patch is already included in an installed release.

[Proposed fix and regression: MoneyPHP PR #832](https://github.com/moneyphp/money/pull/832). The live check at approximately 00:07 Asia/Tashkent on 20 September 2026 showed the PR open. No merged or released fix is claimed here.

Historical attribution correction: issue #506 and the unmerged proposal #509 predate the policy actually merged in [PR #526](https://github.com/moneyphp/money/pull/526). The earlier attribution to #509 as the introducing merge was imprecise. This correction does not change the reproduced numerical counterexample. The bounded recheck also does not establish that every other input is unaffected by the proposed patch.

Disclosure: GERO's own research; AI-assisted analysis and writing, checked against actual library output. Potential business consequences above are an engineering assessment, not measured customer harm.

Editorial status clarification — 20 September 2026: The earlier undated sentence saying that all CI, static-analysis, documentation and benchmark jobs pass is a historical statement, not a fresh verification. Those checks were not rerun for this update. The earlier statement that the proposed correction changes only the large-value case is not established by the bounded recheck and should not be read as a guarantee for every other input.
