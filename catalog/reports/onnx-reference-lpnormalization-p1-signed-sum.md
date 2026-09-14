> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/onnx-reference-lpnormalization-p1-signed-sum.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# ONNX reference LpNormalization uses a signed sum for p=1

Xamit Kadirbekov · Originally published 2026-09-07

The input [1, -1] becomes [0, 0], while the exact L1-normalized result and ONNX Runtime both return [0.5, -0.5].

[Original GERO article](https://www.gero.uz/research/articles/onnx-reference-lpnormalization-p1-signed-sum.html) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/onnx-reference-lpnormalization-p1-signed-sum) · [Original archived collection](https://doi.org/10.5281/zenodo.22683900) · [Individual Zenodo record](https://doi.org/10.5281/zenodo.22728971)

This page presents an existing document from the original 45-document archive. It is not a new finding or a new experiment. The archived corpus text below is preserved verbatim; the original article retains its layout, references and downloadable evidence. Publication dates, scientific claims, limitations and retraction status remain those of the source.

Text SHA-256: `1bab855f658799ba7b4bd835fddda89c92e60b7709a4a7b641dfb480d3e93fb6`

## Original archived text

<!-- BEGIN UNCHANGED CORPUS TEXT -->
ONNX reference LpNormalization uses a signed sum for p=1
https://www.gero.uz/research/articles/onnx-reference-lpnormalization-p1-signed-sum.html

← Research index

INDEPENDENT NUMERICAL AUDIT

7 September 2026

ONNX reference LpNormalization uses a signed sum for p=1

The input [1, -1] becomes [0, 0], while the exact L1-normalized result and ONNX Runtime both return [0.5, -0.5].

Xamit Kadirbekov

Independent verification · GERO Research

Numerical audit

ONNX

Normalization

STATUS · VERIFIED ORDINARY CORRECTNESS DEFECT

Reproduced on ONNX 1.22.0 and current main; correction PR #8427 adds absolute values and a negative-input regression.

UPSTREAM RECORD

onnx/onnx#8427 ↗

Claim under test

ONNX defines

LpNormalization

as division by an Lp norm along a selected axis. For

p=1

, the divisor is the sum of absolute values:

||x||₁ = Σᵢ |xᵢ|

The sign of a component must not reduce the norm contributed by another component.

Minimal counterexample

For

axis=0

,

p=1

, and float32 input

[1, -1]

:

ONNX ReferenceEvaluator   [ 0.0,  0.0]
ONNX Runtime CPU          [ 0.5, -0.5]
exact x / Σ|x|            [ 0.5, -0.5]

The exact L1 norm is

|1| + |-1| = 2

. The reference implementation instead forms the signed sum

1 + (-1) = 0

; its zero-norm guard then replaces the result with a zero vector.

Two further cases isolate the same cause:

input [-1,-2]  reference [ 1/3,  2/3]  exact [-1/3, -2/3]
input [ 1,-3]  reference [-1/2,  3/2]  exact [ 1/4, -3/4]

The second reference output even has L1 norm

2

, so it is not normalized.

Cause

The current implementation computes:

norm = np.power(np.power(x, p).sum(axis=axis), 1.0 / p)

For

p=2

, squaring removes the sign. For

p=1

, it does not. The definition requires

np.abs(x)

before exponentiation.

The existing

p=1

backend examples use only non-negative inputs, so signed summation and absolute summation happen to agree in those tests.

Independent verification

Exact arithmetic gives

[1/2, -1/2]

for the minimal input.

NumPy

x / np.sum(np.abs(x))

gives

[0.5, -0.5]

in float32.

ONNX Runtime 1.29.0 returns the same result on the CPU provider.

ONNX ReferenceEvaluator 1.22.0 returns

[0, 0]

.

The formula remained present on ONNX

main

commit

1fdd92cf2f70f7a00c55d9153d243829d9349a96

, checked 2026-09-07.

Searches across open and closed ONNX issues and pull requests using four relevant phrasings found no duplicate.

The executable reproducer is published at

gero-onnx-reference-lpnormalization-audit

.

Impact and boundary

This defect affects the Python ReferenceEvaluator when

p=1

and the reduced axis contains negative values. It does not implicate ONNX Runtime, which matched the exact result in every tested case. The

p=2

path is not implicated.

A reference evaluator is used as an oracle by backend authors and conformance tests. Here, a correct runtime can disagree with the official reference, and a consumer of

onnx.reference

can receive a vector with wrong signs, magnitudes, or normalization.

This is an ordinary numerical-correctness defect, not a security finding. No deployed-system impact or loss is claimed.

Proposed correction

Compute the norm from absolute values:

norm = np.power(np.power(np.abs(x), p).sum(axis=axis), 1.0 / p)

A regression test must include a negative

p=1

input;

[1, -1]

distinguishes the two formulas with an exact result.

Upstream status

The one-line correction and a negative-input backend regression test were submitted as

ONNX pull request #8427

.

<!-- END UNCHANGED CORPUS TEXT -->

Original report text: CC BY 4.0; code and third-party material retain their original licenses. AI-assisted archival presentation.
