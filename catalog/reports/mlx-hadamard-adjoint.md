> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/mlx-hadamard-adjoint.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# Orthogonal Is Not Symmetric: MLX Hadamard Gradients

Xamit Kadirbekov · Originally published 2026-09-09

MLX uses the forward Hadamard transform in reverse differentiation for nonsymmetric factors 20 and 28. A local C++ repair passes 64 scenarios and 624 checks; measured energy steps confirm the correction.

[Original GERO article](https://www.gero.uz/research/articles/mlx-hadamard-adjoint.html) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/mlx-hadamard-adjoint) · [Original archived collection](https://doi.org/10.5281/zenodo.22683900) · [Individual Zenodo record](https://doi.org/10.5281/zenodo.22694126)

This page presents an existing document from the original 45-document archive. It is not a new finding or a new experiment. The archived corpus text below is preserved verbatim; the original article retains its layout, references and downloadable evidence. Publication dates, scientific claims, limitations and retraction status remain those of the source.

Text SHA-256: `73d758b4a4fdd7328a64f867902489cada29aca3a2ab4f0d7b5a78541034a443`

## Original archived text

<!-- BEGIN UNCHANGED CORPUS TEXT -->
Orthogonal Is Not Symmetric: MLX Hadamard Gradients
https://www.gero.uz/research/articles/mlx-hadamard-adjoint.html

← Research index

NUMERICAL CASE STUDY

9 September 2026

Orthogonal Is Not Symmetric: MLX Hadamard Gradients

A length-preserving transform still needs a transposed operator in its reverse derivative.

Xamit Kadirbekov

Reproducible numerical experiments · GERO Research

MLX 0.32.2

Hadamard adjoint

CPU

STATUS · LOCALLY REPRODUCED

Original: 186 failed checks out of 624. Local C++ patch: all 64 scenarios and 624 checks pass. Eight additional checks execute the actual energy updates. Novelty and maintainer acceptance are unestablished.

The Hadamard reverse derivative uses the forward matrix instead of its transpose. For the nonsymmetric factors 20 and 28, a simple quadratic loss receives an incorrect gradient and an optimization step increases the loss. The corrected step decreases it.

The one-minute explanation

Your browser does not support embedded video.

Watch on YouTube →

Original diagrams and synthetic narration by fictional Alex Vector using the macOS Daniel voice. AI-assisted preparation.

Remotion project and source ledger →

Minimal example

import mlx.core as mx
mx.set_default_device(mx.cpu)
x = mx.array([1.] + [0.] * 19)
loss = lambda x: 0.5 * mx.sum(mx.square(mx.hadamard_transform(x)))
g = mx.grad(loss)(x)
print(g[0].item())              # -0.5; expected approximately 1
print(loss(x - 0.125*g).item())  # 0.57031256; initial loss 0.5

The public

Python probe

selects CPU and sets computational thread environment variables to one before importing MLX. The

recorded results

include a direct-matrix adjoint reference, actual-forward finite differences and symmetric controls.

Why the correct gradient is known

Let

H

be the real Hadamard matrix and

Q=H/√N

. The source tables contain only +1 and −1. Recomputing their products in exact integer arithmetic establishes

HᵀH=N·I

. Therefore, in exact arithmetic:

QᵀQ = I
L(x) = 0.5 × ||Qx||² = 0.5 × ||x||²
∇L(x) = QᵀQx = x
∇²L(x) = I

For this real linear operation, the forward directional derivative is

Qv

, while the reverse derivative must be

Qᵀc

. Orthogonality alone does not imply

Q=Qᵀ

. The

exact certificate

records 0 asymmetric entries for H12, 300 for H20 and 588 for H28. Powers of two also provide symmetric controls.

The central domain is finite real inputs with a supported positive last-axis size

N=m·2ᵏ

, where

m∈{1,12,20,28}

, and a fixed real scale. The

official operation contract

describes supported sizes and normalization. Tests use CPU and the dtypes listed below. The exact matrix argument is separated from finite-precision implementation tolerances.

For arbitrary scale

s

, the corresponding energy gradient is

N·s²·x

. The native tests use this reference for normalized, positive, negative and zero scales. This is a smooth quadratic objective with no tie or subgradient convention to choose.

A measured optimization step raises the loss

For

x=e₀

and learning rate

η=1/8

, the original implementation returns approximately −0.5 for the first gradient component instead of 1. The ideal corrected step is

(1−η)e₀

, yielding loss

0.5×(7/8)²=0.3828125

.

The original audit gave that corrected loss analytically.

For publication, a separate native program also executed the actual forward after both the original and patched updates.

Its

source

,

runner

and logs preserve the distinction.

N

Original first gradient

Original loss after step

Patched first gradient

Patched loss after step

20

−0.5

0.5703125596

1

0.3828125000

28

−0.5

0.5703125000

0.9999999404

0.3828124702

40

−0.4999999702

0.5703125596

1.000000119

0.3828125000

56

−0.5000000596

0.5703125000

0.9999996424

0.3828125298

Initial measured losses are approximately 0.5; their small deviations and the patched values above are FP32 rounding. The sign reversal and loss increase are much larger than that rounding. First-coordinate finite differences of the official wheel's actual forward give approximately 1.

Original update log

·

Patched update log

. The supplemental selection passes 0/8 checks before and 8/8 after. These eight checks are separate from the 624-check primary suite. No real LLM or complete training pipeline was studied.

Source defect and proposed repair

At

pinned

Hadamard::vjp

, around line 6295

, the reverse rule delegates to JVP:

return jvp(primals, cotangents, argnums);

That is correct for the symmetric families. For factors 20 and 28 it applies

sH

instead of

sHᵀ

. With normalized scale, the erroneous energy gradient is

Q²x

instead of

QᵀQx

.

The

local C++ patch

uses the source factorization

H_N=H_m⊗H_(2ᵏ)

. For factors 20 and 28, it applies the transposed small factor through a matrix multiplication and reuses the existing symmetric power-of-two transform. Scale is applied once. Factors 1 and 12 retain their existing VJP path.

The additional constant matrix is at most 28×28; the proposed implementation does not materialize an N×N matrix. It changes only the reverse rule plus the required header include. Forward, JVP and primitive serialization state remain unchanged.

This is a correctness-oriented implementation. Large-array performance, memory behavior and GPU execution were not benchmarked. Maintainer review, a complete build and the upstream suite remain necessary before production acceptance.

Actual validation results

Selection

Original

Patched

Primary scenarios passed

34/64

64/64

Primary checks passed

438/624

624/624

Primary failed checks

186

0

Supplemental e₀ update checks passed

0/8

8/8

The

native harness

covers forward, JVP and VJP against an explicit matrix product; cotangent linearity and zero cotangents; quadratic-energy gradients; Hessian-vector products through forward and reverse differentiation; zero third derivatives; selected actual-forward finite differences;

grad(vmap)

,

vmap(grad)

and a last-axis batch layout.

FP32 single-vector sizes: 1, 2, 4, 12, 24, 48, 20, 40, 80, 28, 56 and 112, each with normalized, +0.25, −0.125 and zero scale.

FP32 batches of two: sizes 4, 12, 20, 28, 40 and 56.

FP16 and BF16 batches of two: sizes 12, 20, 28, 40 and 56, using moderate representable inputs and normalized scale.

Low-precision checks use broader tolerances. They do not establish numerical stability over the complete FP16/BF16 domain. The exact tolerances and deterministic vectors are in the harness. The 186 failures are manifestations of one reverse-rule defect, not 186 independent findings.

Primary baseline log

·

Primary patched log

·

Build commands and exit codes

.

Versions and reproduction boundary

Official wheel: MLX 0.32.2, CPU.

Inspected main:

24c699ecee2f7c8b2040de8da1c8382c8bcf31c7

.

Compatible native base:

ce916dbbcaa88e433b6fd1e60a17f766d49c27fe

.

Hadamard VJP/JVP bodies and constant matrix tables match between the inspected main and compatible base.

Apple clang 17.0.0, C++20, macOS arm64. Test and original/patched translation units compiled sequentially with

-O0

and linked ahead of an existing CPU-only

libmlx.a

.

Reused archive SHA-256:

7826e14e2d1ee526a16352829d460ae39c048e331c339be0b6f8d67d74e68a06

.

This is a partial native rebuild on a compatible base, not a clean full build of main.

The official wheel and original checkout were not modified. Computational thread environment variables were set to one; no GPU numerical execution was used for this report.

All 33 supplied artifact hashes matched before copying. Publication verification repeated the wheel probe, primary native compilation/execution and exact integer certificate checks. The Hadamard-only artifact validator passed 45 checks. Related FFT/pad validation from the supplied validator is outside this publication and was excluded from that 45-check selection.

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python probe.py

export MLX_SOURCE_ROOT=/absolute/path/to/mlx-at-ce916db
export MLX_CPU_BUILD=/absolute/path/to/cpu-build
python3 build_and_test.py
python3 run_e0.py
python3 validate_artifacts.py

BUILD.md

describes the required archive layout and the unexecuted fresh-build setup recipe. Source and result hashes are recorded in

source-metadata.json

and

SHA256SUMS.json

.

Prior work and limits

The original

PR #1249

introduced this transform and its derivative rules. The existing

gradient test

exercises powers of two, which are symmetric controls.

Issue #4049

and

PR #4054

concern Metal kernel launch behavior for the non-power-of-two factors. This report's CPU reverse-derivative failure is a different mechanism and also reproduces at sizes 40 and 56.

Four recorded repository searches returned 1, 2, 6 and 17 results for Hadamard-related queries. No exact matching adjoint report was identified in those results. The search is not exhaustive and does not establish priority.

Public-history review

.

Only the listed sizes, scales, layouts and finite input samples were executed. Larger shapes, Metal/CUDA, compiled graphs and complete-model effects remain untested. Passing a targeted suite is not proof over all possible states. This report does not present previously known FFT or pad observations as new findings.

Prepared with AI assistance. Numerical evidence is actual execution, independent algebra, exact integer products and finite differences. Independent GERO Research; no Apple endorsement. Included MLX source retains its

MIT license

.

Full English report, tests and patch →

Download the evidence ZIP

SHA-256: fe33e65681e9d34116c977dc1d6fbde16e8be8709ddafe57848588b57cd7cb29

Source ledger

·

File checksums

#MLX #Autodiff #HadamardTransform #NumericalComputing #SoftwareTesting

<!-- END UNCHANGED CORPUS TEXT -->

Original report text: CC BY 4.0; code and third-party material retain their original licenses. AI-assisted archival presentation.
