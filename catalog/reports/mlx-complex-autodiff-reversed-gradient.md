> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/mlx-complex-autodiff-reversed-gradient.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# A Real Loss, a Reversed Gradient: Complex Autodiff in MLX

Xamit Kadirbekov · Originally published 2026-09-09

A real loss increases after a gradient-descent step in MLX. Native tests isolate missing complex conjugation and an arccosh branch error; the local C++ patch passes 131 scenarios.

[Original GERO article](https://www.gero.uz/research/articles/mlx-complex-autodiff-reversed-gradient.html) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/mlx-complex-autodiff-reversed-gradient) · [Original archived collection](https://doi.org/10.5281/zenodo.22683900) · [Individual Zenodo record](https://doi.org/10.5281/zenodo.22694763)

This page presents an existing document from the original 45-document archive. It is not a new finding or a new experiment. The archived corpus text below is preserved verbatim; the original article retains its layout, references and downloadable evidence. Publication dates, scientific claims, limitations and retraction status remain those of the source.

Text SHA-256: `8aa0c836569077fe4c7794fcce317df8320727f32d72859fc574b59afb36568f`

## Original archived text

<!-- BEGIN UNCHANGED CORPUS TEXT -->
A Real Loss, a Reversed Gradient: Complex Autodiff in MLX
https://www.gero.uz/research/articles/mlx-complex-autodiff-reversed-gradient.html

← Research index

NUMERICAL CASE STUDY

9 September 2026

A Real Loss, a Reversed Gradient: Complex Autodiff in MLX

A real objective turns a subtle complex-derivative rule into a visible wrong-way update.

Xamit Kadirbekov

Reproducible numerical experiments · GERO Research

MLX 0.32.2

Complex autodiff

CPU

STATUS · LOCALLY REPRODUCED

Two mechanisms. Original: 85 failed scenarios out of 131. Local C++ patch: 131/131 pass. The conjugation class is known; first discovery and full-model impact are not claimed.

At

x=1

, the real loss

L(x)=Re(cos(i*x))=cosh(x)

has derivative

+sinh(1)

. MLX 0.32.2 returns a negative derivative. A gradient-descent step with learning rate 0.01 raises the loss from 1.54308 to 1.55700. After the local C++ repair, it falls to 1.52938.

The parameter and objective are real. This example therefore distinguishes an incorrect update from a difference in how complex gradients are named. It is a synthetic scalar experiment, not a full-model training measurement.

The 45-second explanation

Your browser does not support embedded video.

Download the video

.

Original explanatory diagrams; synthetic narration by the fictional Alex Vector using the macOS Daniel voice. AI-assisted preparation.

Remotion source, script and source ledger →

1. A backward rule needs conjugation

For a holomorphic scalar function away from branch cuts and singularities, and the real inner product

Re(conj(a)*b)

, the forward derivative is

JVP(z,t)=f′(z)*t

. Its adjoint is

VJP(z,c)=conj(f′(z))*c

.

Seven implementations in the

pinned source

delegate their backward rule to the unmodified forward rule:

cos

,

arcsin

,

arccos

,

arctan

,

arcsinh

,

arccosh

and

arctanh

. This omits the conjugate. Existing

exp

,

log

and

sin

rules are controls in the test selection.

The patch follows the existing conjugate-input/conjugate-output pattern. Seven affected functions are instances of one known class, not seven independent discovery claims.

2. Principal square roots do not distribute

Complex

arccosh

has a separate derivative issue. The code uses

1/sqrt(z*z−1)

. The principal-branch derivative, away from the cut, is

1/(sqrt(z−1)*sqrt(z+1))

. These expressions can have opposite signs.

z = −0.25 + 0.5i
tangent = 0.5 + 0.25i

Original JVP:  −0.17871645 + 0.47494581i
Analytic JVP:  +0.17871646 − 0.47494581i
Patched JVP:   +0.17871648 − 0.47494581i

A JVP does not require conjugation; repairing the backward rule alone cannot fix this sign error. The candidate computes the two square roots separately for complex64 and preserves the previous real-input path. Finite differences of the actual forward agree with the analytic reference at the tested points.

Measured before and after

Measurement

Original

Patched

Real-loss derivative at x=1

−1.1752011776

+1.1752011776

Loss after step from 1.5430806875

1.5569984913

1.5293759108

Scenarios passing

46 / 131

131 / 131

Numerical assertions passing

446 / 622

622 / 622

The selection comprises 120 complex scenarios (ten functions, four points ±0.25±0.5i and three cotangents), ten real-input controls and the real-loss example. Checks compare forward, JVP, VJP, the adjoint relation and finite differences. The original fails 84 VJP assertions, 84 adjoint assertions, six arccosh JVP assertions and two real-loss assertions.

Tolerance is

3e−6 + 3e−5*abs(reference)

; central finite differences use

h=2^-10

and absolute tolerance

3e−4

. The references specify the mathematics independently of the patched implementation. Passing this finite selection does not prove correctness for all inputs.

What was actually rebuilt

The wheel probe uses MLX 0.32.2 on CPU, complex64/float32. The inspected main is

24c699ecee2f7c8b2040de8da1c8382c8bcf31c7

. Native before/after executions compile the relevant C++ translation unit on compatible base

ce916dbbcaa88e433b6fd1e60a17f766d49c27fe

and link it against an existing CPU-only static archive.

This is a partial native rebuild, not a clean build of current main.

All 20 tested JVP/VJP method bodies are byte-identical between the base and inspected main. The patch also applies cleanly to main. The archive hash, compiler flags, raw logs and source snapshots are retained. The compiler is Apple clang 17.0.0, C++20, on macOS 15.5 arm64. Compilation and experiments ran sequentially with thread environment limits set to one.

Prior work and limits

MLX #3766 by obchain

already addresses missing conjugation in other unary functions; the inspected source history contains its change. This report adds tested functions and a separately motivated arccosh branch repair. The originality of the complete selection and maintainer acceptance remain unestablished.

The test does not cover full models, the complete upstream suite, GPU/CUDA, FP16/BF16, branch-cut boundaries, extreme magnitudes or all possible complex inputs. No Apple-device impact is inferred. API availability for the prior PR differed from cached web pages, so no current PR status is inferred from a cached badge.

Reproduce and inspect the evidence

Immutable report, scripts, source snapshots and patch

Native C++ regression harness

·

Candidate patch

Original log

·

Patched log

Build setup and reproduction boundary

·

Primary sources and prior-art ledger

# Compatible Mac, from the extracted evidence directory
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python probe.py

# With a compatible CPU build of the recorded base:
export MLX_SOURCE_ROOT=/absolute/path/to/mlx-at-ce916db
export MLX_CPU_BUILD=/absolute/path/to/cpu-build
.venv/bin/python build_and_test.py

Download the evidence package (ZIP) →

SHA-256:

438c61bfacd0872b78115b4cfc096f8e6eaa0377f8f4e717b67430cbcb80c2df

Independent reproduction and scoped numerical-correctness reviews are welcome:

contact GERO

. Prepared with AI assistance. Numerical claims come from actual local runs and separately specified references; this is not a third-party audit or an Apple-endorsed report.

#MachineLearning #MLX #Autodiff #ComplexNumbers #SoftwareTesting

<!-- END UNCHANGED CORPUS TEXT -->

Original report text: CC BY 4.0; code and third-party material retain their original licenses. AI-assisted archival presentation.
