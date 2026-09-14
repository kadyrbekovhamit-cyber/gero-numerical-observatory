> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/mlx-muon-convolution-scaling.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# Same Gradients, Half the Step: MLX Muon and Convolution Shapes

Xamit Kadirbekov · Originally published 2026-09-09

A native CPU reproduction isolates a shape-dependent Muon update in MLX. Equivalent convolution and linear layers agree before the optimizer step; a minimal reorder passes 31 focused tests.

[Original GERO article](https://www.gero.uz/research/articles/mlx-muon-convolution-scaling.html) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/mlx-muon-convolution-scaling) · [Original archived collection](https://doi.org/10.5281/zenodo.22683900) · [Individual Zenodo record](https://doi.org/10.5281/zenodo.22694717)

This page presents an existing document from the original 45-document archive. It is not a new finding or a new experiment. The archived corpus text below is preserved verbatim; the original article retains its layout, references and downloadable evidence. Publication dates, scientific claims, limitations and retraction status remain those of the source.

Text SHA-256: `3dec385e521da9d1f6da555ef6554f397552974bc8e1bcc26303f3a8b55a88af`

## Original archived text

<!-- BEGIN UNCHANGED CORPUS TEXT -->
Same Gradients, Half the Step: MLX Muon and Convolution Shapes
https://www.gero.uz/research/articles/mlx-muon-convolution-scaling.html

← Research index

NUMERICAL CASE STUDY

9 September 2026

Same Gradients, Half the Step: MLX Muon and Convolution Shapes

Equivalent weight layouts agree on the calculation and its derivative. Then the optimizer makes their updates different.

Xamit Kadirbekov

Reproducible numerical experiments · GERO Research

MLX 0.32.2

Muon

CPU FP32

STATUS · LOCALLY REPRODUCED

Minimal patch: 31/31 focused tests pass; the original fails 19. These are manifestations of one defect. Novelty, maintainer acceptance and full-model effects are unestablished.

A 1×1 convolution and an equivalent linear layer can implement the same function. In this native MLX experiment, their outputs, losses and gradients are identical. Yet the convolution's Muon weight step is half the linear layer's step within FP32 rounding. The cause is the dimensions used to compute the optimizer's scale.

The 40-second explanation

Your browser does not support embedded video.

Download the video

.

Original explanatory diagrams; synthetic narration by the fictional Alex Vector using the macOS Daniel voice. AI-assisted preparation.

Remotion source, script and source ledger →

What the shape changes

In the

pinned MLX source

, Muon flattens trailing tensor dimensions before Newton–Schulz, then restores the original shape before calculating the learning-rate multiplier. For a convolution weight

(Cout,H,W,Cin)

, the resulting ratio is

W/Cin

.

The matrix-based multiplier instead uses

Cout/(H*W*Cin)

. This ordering is consistent with the

author's Muon source

, reviewed as an algorithm reference and not cross-executed in PyTorch.

scale = sqrt(max(1, rows / columns))

Convolution layout: (8, 1, 1, 2)
Flattened matrix:   (8, 2)

After restoring shape: sqrt(max(1, 1/2)) = 1
Using matrix shape:   sqrt(max(1, 8/2)) = 2

Holding the flattened update fixed therefore predicts a step ratio of one half. This is an algebraic prediction; the following values are separate native measurements.

Measured behavior

With zero parameters, gradient

(arange(N)+1)/16

, learning rate 0.01, momentum and weight decay zero, Nesterov disabled and five Newton–Schulz iterations:

Tensor / matrix shape

Tensor step norm

Matrix step norm

Ratio

(8,1,1,2) / (8,2)

0.011393075

0.022786150

0.5

(1,1,8,2) / (1,16)

0.013928725

0.006964363

2.0

(2,2,2,2) / (2,8)

0.009926165

0.009926165

1.0

Both released MLX 0.32.2 and the pinned Python module reproduce these comparisons. The third is a control where the two scale rules coincide.

The real convolution/linear check uses

mx.conv2d(x,w.reshape(8,1,1,2))

versus

x @ w.T

, and the mean square of the output as loss. Maximum output and gradient differences are zero; both losses are

0.0851593017578125

. Step norms are

0.026587212458252907

for linear weights and

0.013293605297803879

for convolution weights.

Exact inputs and reproduction →

The minimal patch

The patch

moves the existing scale calculation before the reshape. The API, Newton–Schulz, momentum and weight-decay logic are unchanged. The patch applies cleanly and produces the tested candidate byte-for-byte; the AST outside

Muon.apply_single

remains unchanged.

Variant

Pass

Fail

Execution errors

Original pinned Python

12 / 31

19

0

Locally patched Python

31 / 31

0

0

The selection covers 24 shape/configuration comparisons, four independent Python-math scale checks, two real convolution/linear comparisons and the unchanged body of an upstream Muon test in a CPU fixture. The scale-only checks use zero Newton–Schulz steps to isolate normalization; the other comparisons use five. Three-step comparisons also check momentum state, shape, dtype and unchanged inputs. This is not the entire upstream suite.

Scope, provenance and reproduction

For finite, consistent parameters and gradients represented as a tensor or its flattened matrix, the tested invariant is update agreement at

atol=2e-6

,

rtol=2e-5

. The domain is eight small shapes and three optimizer configurations. Passing these tests does not establish correctness for all states.

The source pin is

24c699ecee2f7c8b2040de8da1c8382c8bcf31c7

. Its Python code ran against the 0.32.2 native wheel on macOS 15.5 arm64, Python 3.12.14, CPU FP32. A fresh native build of main, GPU, FP16/BF16, distributed training, complete model training and device effects were not tested.

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python probe.py
.venv/bin/python regression.py

Run the scripts sequentially from the extracted evidence directory on an MLX-compatible Mac. They set CPU and one-thread environment limits before import. The regression runner records intentional baseline failures as well as patched results: inspect its JSON counts, not just the script exit code.

A publication-time search of Muon-related issues and pull requests found no exact duplicate in the inspected scope. Novelty and maintainer acceptance remain unconfirmed. The observed update discrepancy does not establish harm to a trained model or an Apple device.

Evidence and primary sources

Immutable GitHub report, native scripts, results and patch

Source ledger and duplicate-search limits

Original test log

·

Patched test log

Official MLX Muon documentation

Download the evidence package (ZIP) →

SHA-256:

9eb19d2debc3156406a6608427e57c42468b306977f21d5677605c3ec96386d6

Independent reproduction and scoped numerical-correctness reviews are welcome:

contact GERO

. Prepared with AI assistance; the evidence consists of actual native executions and an independently specified algebraic check, not a third-party audit.

#MachineLearning #Muon #MLX #NumericalComputing #SoftwareTesting

<!-- END UNCHANGED CORPUS TEXT -->

Original report text: CC BY 4.0; code and third-party material retain their original licenses. AI-assisted archival presentation.
