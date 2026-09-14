> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/apache-fineract-nop-inverse-error.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# Apache Fineract nop does not invert pmt for non-zero rates

Xamit Kadirbekov · Originally published 2026-09-05

Fineract's public number-of-payments helper returns 7 when its adjacent payment function was given a 12-period loan.

[Original GERO article](https://www.gero.uz/research/articles/apache-fineract-nop-inverse-error.html) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/apache-fineract-nop-inverse-error) · [Original archived collection](https://doi.org/10.5281/zenodo.22683900) · [Individual Zenodo record](https://doi.org/10.5281/zenodo.22729038)

This page presents an existing document from the original 45-document archive. It is not a new finding or a new experiment. The archived corpus text below is preserved verbatim; the original article retains its layout, references and downloadable evidence. Publication dates, scientific claims, limitations and retraction status remain those of the source.

Text SHA-256: `7ffa77596c73d983e913927aa262c0f474c78dcd47f0faefcfe1d25df19e16a7`

## Original archived text

<!-- BEGIN UNCHANGED CORPUS TEXT -->
Apache Fineract nop does not invert pmt for non-zero rates
https://www.gero.uz/research/articles/apache-fineract-nop-inverse-error.html

← Research index

INDEPENDENT NUMERICAL AUDIT

5 September 2026

Apache Fineract nop does not invert pmt for non-zero rates

Fineract's public number-of-payments helper returns 7 when its adjacent payment function was given a 12-period loan.

Xamit Kadirbekov

Independent verification · GERO Research

Mathematical audit

Apache Fineract

Loan formulas

STATUS · FIXED UPSTREAM

Apache fixed the formula for Fineract 1.16.0 and credited Xamit Kadirbekov / GERO Research in the upstream commit.

UPSTREAM RECORD

apache/fineract e7bdad1 ↗

Finding

The public

FinanicalFunctions.nop()

method is intended to calculate a number of payments, but its non-zero-rate branch repeats a payment-style expression with

emiAmount

in the exponent. It is not the algebraic inverse of the adjacent

pmt()

method.

Minimal counterexample

For end-of-period payments:

rate = 0.01
principal = 1000
future value = 0
periods = 12

Fineract's own payment function returns:

pmt(0.01, 12, 1000, 0, false) = -88.848788678342

The inverse should therefore recover

12

. The independent closed form is:

N = log(PMT / (PMT + principal * rate)) / log1p(rate)

It returns

12.000000000

, while Fineract returns:

nop(0.01, -88.848788678342, 1000, 0, false) = 7

Verification matrix

Terms

6

,

12

,

24

, and

36

were tested at periodic rates

0

,

0.001

,

0.01

, and

0.05

.

Zero-rate branch:

4/4

round trips passed.

Non-zero-rate branch:

0/12

round trips passed.

Environment: OpenJDK

26.0.2

, macOS ARM64.

Source baseline: commit

74099701987ccb4755706d9a3de9fbbd3576ea1b

.

Proposed correction

Let

d = (type ? 1 + rate : 1)

and

A = emiAmount * d

. Solving the adjacent

pmt()

equation for the term gives:

growth = (A - rate * futureValue) / (A + rate * principal)
N = log(growth) / log1p(rate)

Because the existing API returns an integer number of payments, the computed term is rounded to the closest whole payment. A regression suite covering zero and non-zero rates, both payment timings, and a non-zero future value passes

33/33

round trips with the correction.

The focused implementation and regression tests are available in the public fork at commit

`6980e1377ce2e5a1eaf0586e628b27bb64dd8c26`

. The full Gradle target could not be run locally because this checkout requires JDK 25 while the available runtime is JDK 26; the corrected source compiled and the direct round-trip harness passed.

Duplicate and impact checks

Repository-wide search found no call site for

FinanicalFunctions.nop()

; only the declaration exists across 6,773 Java files. The sole external reference to the class is a call to

FinanicalFunctions.pmt()

from

LoanApplicationTerms.java

.

The repository uses ASF JIRA rather than GitHub issues. Targeted public searches found no matching report.

Upstream status

Apache registered the finding as

FINERACT-2809

and marked it fixed for Fineract 1.16.0 on 7 September 2026. The correction and regression tests are now in upstream

develop

as commit

`e7bdad1`

. The commit credits "Xamit Kadirbekov / GERO Research", and the maintainer's JIRA resolution comment states: "Fix belongs to Xamit Kadirbekov".

Boundary

This is a dormant public-library defect. Because the current repository has no

nop()

call site, it is not evidence of an incorrect real loan schedule, customer impact, data loss, or a security vulnerability.

<!-- END UNCHANGED CORPUS TEXT -->

Original report text: CC BY 4.0; code and third-party material retain their original licenses. AI-assisted archival presentation.
