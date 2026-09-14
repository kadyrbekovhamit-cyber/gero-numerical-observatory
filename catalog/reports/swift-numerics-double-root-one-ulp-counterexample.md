> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/swift-numerics-double-root-one-ulp-counterexample.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# An exact fifth root exposes a one-ULP accuracy gap in Swift Numerics

Xamit Kadirbekov · Originally published 2026-09-03

Double.root(3125, 5) returns 5.000000000000001 although the exact, representable result is 5.0.

[Original GERO article](https://www.gero.uz/research/articles/swift-numerics-double-root-one-ulp-counterexample.html) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/swift-numerics-double-root-one-ulp-counterexample) · [Original archived collection](https://doi.org/10.5281/zenodo.22683900) · [Individual Zenodo record](https://doi.org/10.5281/zenodo.22729051)

This page presents an existing document from the original 45-document archive. It is not a new finding or a new experiment. The archived corpus text below is preserved verbatim; the original article retains its layout, references and downloadable evidence. Publication dates, scientific claims, limitations and retraction status remain those of the source.

Text SHA-256: `c99ec4137c7b8b547cd61e80de6f394f6488db076740cee0afd9b9983a2d28d4`

## Original archived text

<!-- BEGIN UNCHANGED CORPUS TEXT -->
An exact fifth root exposes a one-ULP accuracy gap in Swift Numerics
https://www.gero.uz/research/articles/swift-numerics-double-root-one-ulp-counterexample.html

← Research index

INDEPENDENT NUMERICAL AUDIT

3 September 2026

An exact fifth root exposes a one-ULP accuracy gap in Swift Numerics

Double.root(3125, 5) returns 5.000000000000001 although the exact, representable result is 5.0.

Xamit Kadirbekov

Independent verification · GERO Research

Numerical audit

Swift

Floating point

STATUS · VERIFIED 1-ULP COUNTEREXAMPLE

The public implementation was built and executed at the cited commit. This is an accuracy finding, not a security finding.

REVIEWED SOURCE · APPLE SWIFT NUMERICS

Exact implementation and commit ↗

Finding

At source commit

899af71c0256d0ad181e3b7eb3453c1065d928a5

, the public call

Double.root(3125.0, 5)

returns

5.000000000000001

. The mathematically exact result is

5.0

, which is exactly representable as a binary

Double

.

Minimal reproducer

import RealModule

let result = Double.root(3125.0, 5)
print(result)
print(result == 5.0)
print(String(result.bitPattern, radix: 16))

Observed output:

5.000000000000001
false
4014000000000001

Exact certificate

5 × 5 × 5 × 5 × 5 = 3125

.

Therefore the exact fifth root of 3125 is 5.

The bit pattern of exact

5.0

is

0x4014000000000000

.

The returned bit pattern is

0x4014000000000001

.

The returned value is therefore one representable

Double

above the exact answer: a one-ULP gap.

⁵√3125 = 5

implementation → 5 + 1 ULP

Why it happens

The implementation reduces an integer root to a general power:

libm_pow(x.magnitude, 1 / Double(n))

The binary

Double

format cannot represent

1/5

exactly. That rounded exponent is then passed to

pow

. The source already contains a TODO noting that the implementation is “not quite correct” because either

n

or

1/n

may not be representable as

Double

. This audit supplies a small, concrete regression case for that acknowledged limitation.

Reproduction record

Repository:

apple/swift-numerics

.

Commit:

899af71c0256d0ad181e3b7eb3453c1065d928a5

.

Toolchain: Apple Swift 6.1.2.

Target: arm64 Apple macOS.

The

RealModule

target was built and the public

Double.root

method was called directly.

Question for upstream

Should

root(x, n)

guarantee a correctly rounded result when the exact root is representable, or is a small approximation error part of the intended contract? In either case,

root(3125, 5)

is a useful regression and documentation test because it makes the current accuracy boundary explicit.

Boundary

This reproduces a one-ULP numerical discrepancy in an open-source mathematical function. It is not a demonstrated security vulnerability, does not establish material impact in an Apple product, and does not imply that general-purpose

pow

implementations promise exact results. Upstream maintainers retain authority over the intended accuracy contract and any correction.

<!-- END UNCHANGED CORPUS TEXT -->

Original report text: CC BY 4.0; code and third-party material retain their original licenses. AI-assisted archival presentation.
