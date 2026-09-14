> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/onnx-reference-gatherelements-rejects-subshapes.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# ONNX ReferenceEvaluator rejects valid GatherElements subshapes

Xamit Kadirbekov · Originally published 2026-09-09

Inputs of equal rank with smaller non-gather index dimensions are mathematically well-defined and accepted by ONNX Runtime, but the reference oracle requires exact shape equality.

[Original GERO article](https://www.gero.uz/research/articles/onnx-reference-gatherelements-rejects-subshapes.html) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/onnx-reference-gatherelements-rejects-subshapes) · [Original archived collection](https://doi.org/10.5281/zenodo.22683900) · [Individual Zenodo record](https://doi.org/10.5281/zenodo.22695938)

This page presents an existing document from the original 45-document archive. It is not a new finding or a new experiment. The archived corpus text below is preserved verbatim; the original article retains its layout, references and downloadable evidence. Publication dates, scientific claims, limitations and retraction status remain those of the source.

Text SHA-256: `ded68b4a92e0f9f96eb2d2259222ee7c8a823453c8c915d3ca80c768d281d086`

## Original archived text

<!-- BEGIN UNCHANGED CORPUS TEXT -->
ONNX ReferenceEvaluator rejects valid GatherElements subshapes
https://www.gero.uz/research/articles/onnx-reference-gatherelements-rejects-subshapes.html

← Research index

INDEPENDENT NUMERICAL AUDIT

9 September 2026

ONNX ReferenceEvaluator rejects valid GatherElements subshapes

Inputs of equal rank with smaller non-gather index dimensions are mathematically well-defined and accepted by ONNX Runtime, but the reference oracle requires exact shape equality.

Xamit Kadirbekov

Independent verification · GERO Research

Numerical audit

ONNX

Tensor indexing

STATUS · VERIFIED ORDINARY CORRECTNESS DEFECT

Reproduced on ONNX 1.22.0 and current main; correction PR #8444 passed 454 reference-evaluator tests and 36 differential shape/axis cases.

UPSTREAM RECORD

onnx/onnx#8444 ↗

Finding

GatherElements

requires

data

and

indices

to have equal rank, and defines the output shape as the shape of

indices

. Coordinates on each non-gather axis range over the corresponding index domain; those dimensions may therefore be smaller than the data dimensions.

The released ReferenceEvaluator imposes a stricter condition: every non-gather dimension must equal the corresponding data dimension. It rejects valid subshape inputs before performing any gather.

Minimal counterexample

data = [[10, 11, 12],
        [20, 21, 22]]              shape (2, 3)
indices = [[2, 0]]                 shape (1, 2)
axis = 1

ReferenceEvaluator: ValueError requiring equal non-axis dimensions
direct indexing:    [[12, 10]]
ONNX Runtime:       [[12, 10]]

The first output row refers only to the first data row, so the smaller leading dimension is well-defined and accepted by ONNX Runtime.

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

Three regression cases cover axes 0, 1 and -1 in two and three dimensions.

Thirty-six differential cases cover equal and smaller non-axis dimensions, positive and negative axes, and positive and negative indices; all corrected outputs match ONNX Runtime exactly.

The complete reference-evaluator test module passed: 454 tests, 4 skipped.

Restoring the released implementation makes all three new tests fail with the original shape-equality exception.

Ruff format/check and

git diff --check

pass.

Searches of ONNX issues and pull requests found no matching report.

Correction

Restrict data to the coordinate extent of the index domain on non-gather axes, then apply

numpy.take_along_axis

. This directly models the operator equations, preserves equal-shape behavior, and supports positive and negative indices.

The correction and regression tests are submitted in

ONNX PR #8444

. The public reproducer is in

gero-onnx-gatherelements-subshape-audit

.

Boundary

This is a reference-implementation correctness defect, not a security issue. It affects validation of otherwise valid

GatherElements

models. ONNX Runtime produces the specified output for the counterexample.

<!-- END UNCHANGED CORPUS TEXT -->

Original report text: CC BY 4.0; code and third-party material retain their original licenses. AI-assisted archival presentation.
