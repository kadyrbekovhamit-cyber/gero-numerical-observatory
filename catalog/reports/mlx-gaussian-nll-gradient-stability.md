> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/mlx-gaussian-nll-gradient-stability.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# Finite Loss, Wrong Gradient: Gaussian NLL in MLX

Xamit Kadirbekov · Originally published 2026-09-09

Native MLX 0.32.2 CPU reproduction: finite Gaussian NLL with a variance gradient of the wrong sign. Inputs, references, regression tests and limits of a mitigation.

[Original GERO article](https://www.gero.uz/research/articles/mlx-gaussian-nll-gradient-stability.html) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/mlx-gaussian-nll-gradient-stability) · [Original archived collection](https://doi.org/10.5281/zenodo.22683900) · [Individual Zenodo record](https://doi.org/10.5281/zenodo.22694686)

This page presents an existing document from the original 45-document archive. It is not a new finding or a new experiment. The archived corpus text below is preserved verbatim; the original article retains its layout, references and downloadable evidence. Publication dates, scientific claims, limitations and retraction status remain those of the source.

Text SHA-256: `a3fb70d4aef47c79923397a61abbb30140c753ffc14597907b4db70bc183299f`

## Original archived text

<!-- BEGIN UNCHANGED CORPUS TEXT -->
Finite Loss, Wrong Gradient: Gaussian NLL in MLX
https://www.gero.uz/research/articles/mlx-gaussian-nll-gradient-stability.html

← Research index

NUMERICAL CASE STUDY

9 September 2026

Finite Loss, Wrong Gradient: Gaussian NLL in MLX

A finite loss can still point the optimizer in the wrong direction. Three scalar inputs expose an unstable intermediate in autodiff.

Xamit Kadirbekov

Reproducible numerical experiments · GERO Research

MLX 0.32.2

Gaussian NLL

CPU FP16 / FP32

STATUS · LOCALLY REPRODUCED

Numerical-stability result on native MLX 0.32.2 CPU. Novelty and full-model impact are unestablished. The proposed mitigation has a retained counterexample.

With FP16

mean=20

,

target=0

and

variance=300

, Gaussian negative log-likelihood in MLX returns a finite loss, but its variance gradient has the wrong sign. Central differences of the actual forward function independently confirm the negative slope.

The derivative that should be returned

The

API contract

uses variance, not standard deviation. For one element, finite inputs, variance above epsilon and

full=False

:

r = mean - target
L = 0.5 * (log(v) + r*r/v)
dL/dv = (v-r*r)/(2*v*v)

At r=20, v=300:
dL/dv = (300-400)/180000 = -1/1800

The native result is

+0.001667022705078125

, versus a mathematical derivative of approximately

−0.000555555556

. This is a sign reversal, not a last-bit rounding difference. Negative Gaussian NLL values themselves are valid for a density and are not counted as an error.

Observed values

Type; mean, target, variance

Native loss

Reference loss

Native dL/dv

Reference dL/dv

FP16; 300, 0, 300

Inf

152.851891

NaN

−0.498333

FP16; 20, 0, 300

3.517578125

3.518557904

+0.00166702

−0.00055556

FP16; 0, 0, 0.0001

−4.60546875

−4.605087223

NaN

+4999.170441

FP32; 1e20, 0, 1e20

Inf

5.0000001002e19

NaN

−0.5

References use 80-digit Decimal arithmetic from the actual rounded inputs. FP16

0.0001

, for example, is

0.00010001659393310547

. The

full machine-readable results

preserve exact inputs, controls, intermediates and additional FP32 cases. All 14 inspected scenarios agree between the pinned Python function and the installed public API.

Why an algebraically correct formula fails

The loss squares the residual before division. FP16 cannot represent

300²=90000

: its largest finite value is 65504. Forward therefore overflows before the division, even though the final loss of about 152.85 would fit.

Backward contains a different dangerous square. The denominator contribution of

Divide::vjp

is evaluated as

−(cotangent*numerator)/square(denominator)

. With residual 20 and variance 300, the residual square 400 is finite, but the variance square overflows. The negative gradient contribution disappears, leaving approximately

1/(2*300)

.

For a sufficiently small variance, its square instead rounds to zero, creating Inf or NaN in backward. Direct tests of

a/b

reproduce the same mechanism without the loss function. At FP16

a=b=300

, forward is 1 but the derivative with respect to b is −0 rather than −1/300. These cases are related manifestations of numerical instability, not separate discoveries for every failing input.

At the finite-loss example, central differences of the native forward are negative for steps 2, 8, 16 and 32. Their values range from −0.00048828125 to −0.000579833984375. The steps accommodate FP16 quantization; this is a local sign check, not a proof for all inputs.

A mitigation that passes 12 tests—and still has a limit

A local candidate promotes inputs to FP32 before subtraction and squares the normalized residual

r/sqrt(v)

. The original passes 4 of 12 regression methods; the candidate passes 12/12. The exact public minimal script and both selections were rerun before publication.

The candidate is not a universal repair.

It changes the FP16 loss dtype and epsilon precision. At FP32

mean=variance≈3e38

, it still returns

dL/dv=−Inf

instead of −0.5. This counterexample is retained and excluded explicitly from the passing 12-test subset. Simply reordering the forward term to

r*(r/v)

does not repair the gradient either. A production solution needs a stable backward rule and an explicit supported domain.

Candidate code →

·

Regression assertions →

·

Publication-day rerun →

Minimal native reproduction

"""Public MLX API reproduction; small synthetic arrays on CPU."""
import mlx.core as mx
from mlx.nn.losses import gaussian_nll_loss

mx.set_default_device(mx.cpu)
for mean, variance in [(300., 300.), (20., 300.), (0., .0001)]:
    x = mx.array([mean], dtype=mx.float16)
    y = mx.array([0.], dtype=mx.float16)
    v = mx.array([variance], dtype=mx.float16)
    loss = gaussian_nll_loss(x, y, v)
    derivative = mx.grad(lambda z: gaussian_nll_loss(x, y, z))(v)
    print('mean=', x.item(), 'variance=', v.item(),
          'loss=', loss.item(), 'dL/dvariance=', derivative.item())

Measured environment: macOS 15.5 arm64, Python 3.12.14, official MLX 0.32.2 wheel, CPU. The installed native core hash matches its wheel RECORD. The installed Gaussian function structurally matches the source pinned at

ce916dbbcaa88e433b6fd1e60a17f766d49c27fe

; this is not a fresh native-core build of that commit.

Scope and useful negative results

GPU/Metal execution, complete model training, device effects, every input range and the latest upstream HEAD were not tested. The measured mean/target gradients remain correct in the original 300/300 example; it would be inaccurate to say all gradients are wrong.

Separately, nntrainer nearest/bilinear Upsample passed the original 18 tests and 72 additional cases against an independent interpolation matrix. No new defect was established in that bounded CPU FP32 area. The code and measured results are included in the evidence appendix.

Two publication-day GitHub searches returned documentation, control-flow, attention and memory-related reports; no exact Gaussian gradient-sign match was identified in that scope. That limited search does not prove novelty. Maintainer confirmation, upstream acceptance, security impact and bounty eligibility are not established.

Evidence and primary sources

Immutable GitHub report, runnable code, tests and logs

Pinned Gaussian loss source

Pinned division backward implementation

Source ledger and duplicate-search limits

Download the evidence package (ZIP) →

SHA-256:

b6f066befd016ce54c162332a5c83bc8f7a00d4be97e6b9ff1fc62df3d717232

Independent reproduction and scoped numerical-correctness reviews are welcome:

contact GERO

. Prepared with AI assistance; the evidence consists of native executions and algorithmically independent references, not a third-party laboratory review.

#MachineLearning #NumericalStability #Autodiff #MLX #SoftwareTesting

<!-- END UNCHANGED CORPUS TEXT -->

Original report text: CC BY 4.0; code and third-party material retain their original licenses. AI-assisted archival presentation.
