> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/onnx-reference-unique-unsorted-nonzero-axis.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# ONNX ReferenceEvaluator gathers unsorted Unique slices from the wrong axis

Xamit Kadirbekov · Originally published 2026-09-09

With sorted=0 and a nonzero axis, first-occurrence indices are always applied to axis 0, producing wrong data or an immediate out-of-bounds exception.

[Original GERO article](https://www.gero.uz/research/articles/onnx-reference-unique-unsorted-nonzero-axis.html) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/onnx-reference-unique-unsorted-nonzero-axis) · [Original archived collection](https://doi.org/10.5281/zenodo.22683900) · [Individual Zenodo record](https://doi.org/10.5281/zenodo.22695859)

This page presents an existing document from the original 45-document archive. It is not a new finding or a new experiment. The archived corpus text below is preserved verbatim; the original article retains its layout, references and downloadable evidence. Publication dates, scientific claims, limitations and retraction status remain those of the source.

Text SHA-256: `2174efe49c199122e5c33b643bd6c7bd4c42be9c9a1aa4502c9ba0302082e712`

## Original archived text

<!-- BEGIN UNCHANGED CORPUS TEXT -->
ONNX ReferenceEvaluator gathers unsorted Unique slices from the wrong axis
https://www.gero.uz/research/articles/onnx-reference-unique-unsorted-nonzero-axis.html

← Research index

INDEPENDENT NUMERICAL AUDIT

9 September 2026

ONNX ReferenceEvaluator gathers unsorted Unique slices from the wrong axis

With sorted=0 and a nonzero axis, first-occurrence indices are always applied to axis 0, producing wrong data or an immediate out-of-bounds exception.

Xamit Kadirbekov

Independent verification · GERO Research

Numerical audit

ONNX

Tensor indexing

STATUS · VERIFIED ORDINARY CORRECTNESS DEFECT

Reproduced on ONNX 1.22.0 and current main; correction PR #8443 passed 453 reference-evaluator tests and 13 differential axis cases.

UPSTREAM RECORD

onnx/onnx#8443 ↗

Finding

Unique(sorted=0)

returns unique values or subtensors in order of first occurrence. The released ONNX ReferenceEvaluator correctly computes the first-occurrence indices, but it always applies those indices to input axis 0. When the operator's

axis

is nonzero, the evaluator therefore selects from the wrong dimension.

For many shapes the result is an immediate exception. For a 2×4 input with

axis=1

, unique column index 3 is valid on the four-column dimension but out of range on the two-row dimension hard-coded by the evaluator.

Minimal counterexample

X = [[ 1,  2,  1,  3],
     [10, 20, 10, 30]]
axis = 1, sorted = 0

ReferenceEvaluator: IndexError: index 3 is out of bounds for axis 0 with size 2
specified Y / ORT:  [[1, 2, 3], [10, 20, 30]]
indices:             [0, 1, 3]
inverse_indices:     [0, 1, 0, 2]
counts:              [2, 1, 1]

Verification

Reproduced with ONNX

1.22.0

, ONNX Runtime

1.29.0

, NumPy

2.5.2

, Python 3.12 and the CPU execution provider.

Confirmed unchanged on ONNX

main

commit

c71cbb485c97aa2cb257626a5ac5b0f205c49a22

.

Regression tests cover positive

axis=1

and the equivalent negative

axis=-1

, checking all four outputs exactly.

Thirteen differential cases across flattened, positive and negative axes match ONNX Runtime after the correction.

The complete reference-evaluator test module passed: 453 tests, 4 skipped.

Restoring the hard-coded axis makes both regression cases fail with the original exception.

Ruff format/check and

git diff --check

pass.

Searches of ONNX issues and pull requests found no matching report.

Correction

Pass the operator's actual

axis

to

numpy.take

. The correction is one source line; the absent-axis case remains valid because NumPy gathers from the flattened input when

axis=None

.

The correction and regression tests are submitted in

ONNX PR #8443

. The public reproducer is in

gero-onnx-unique-unsorted-axis-audit

.

Boundary

This is a reference-implementation correctness defect, not a security issue. It affects unsorted

Unique

evaluation whenever a nonzero axis is supplied. ONNX Runtime produces the specified output for the counterexample.

<!-- END UNCHANGED CORPUS TEXT -->

Original report text: CC BY 4.0; code and third-party material retain their original licenses. AI-assisted archival presentation.
