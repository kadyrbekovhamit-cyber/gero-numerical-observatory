# Finite Loss, Wrong Gradient: Gaussian NLL in MLX

9 September 2026 · Native CPU measurements · MLX 0.32.2

For FP16 inputs `mean=20`, `target=0`, `variance=300`, MLX returns a finite
Gaussian negative log-likelihood but a variance gradient with the opposite
sign to the mathematical derivative. The actual forward function's finite
differences also have the opposite sign to autograd.

This is a reproducible numerical-stability case study. It is not a claim of
first discovery, security impact, a production-ready patch or upstream acceptance.

## The contract

The [MLX API documentation](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.gaussian_nll_loss.html)
defines `vars` as variance, not standard deviation. For finite scalar inputs,
`v > eps`, `full=False`, and a single element:

```text
r = mean - target
L = 0.5 * (log(v) + r*r/v)
dL/dmean = r/v
dL/dtarget = -r/v
dL/dv = (v-r*r)/(2*v*v)
```

`full=True` adds a constant and does not change these derivatives. A mean
reduction over N elements divides their gradients by N. Negative Gaussian
NLL values are valid for a density. The derivative at `v=eps` is excluded;
the clamped branch below epsilon is checked separately.

Reference values use 80-digit Decimal arithmetic starting from the actual
FP16/FP32-rounded inputs. Intermediate expressions and final representability
are distinguished: a genuinely unrepresentable final answer is not counted
as spurious overflow.

## Measured cases

| Type; mean, target, variance | Native loss | Reference loss | Native dL/dv | Reference dL/dv |
|---|---:|---:|---:|---:|
| FP16; 300, 0, 300 | Inf | 152.8518912373 | NaN | -0.4983333333 |
| FP16; 20, 0, 300 | 3.517578125 | 3.51855790399 | **+0.001667022705** | **-0.000555555556** |
| FP16; 10, 0, 300 | 3.017578125 | 3.01855790399 | +0.001667022705 | +0.001111111111 |
| FP16; 0, 0, 0.0001 | -4.60546875 | -4.60508722321 | NaN | +4999.170441 |
| FP16; 0.02, 0, 0.0001 | -2.60546875 | -2.60456460513 | -Inf | -15002.73664 |
| FP32; 1e20, 0, 1e20 | Inf | 5.0000001002e19 | NaN | -0.5 |
| FP32; 2e10, 0, 1e20 | 25.0258502960 | 25.0258508999 | +4.999999841e-21 | -1.499999930e-20 |

For example, FP16 `0.0001` is actually `0.00010001659393310547`.
[Full measurements, rounded inputs and references](evidence/mlx-review-results.json)
include controls and counterexamples. All 14 scenarios gave identical results
through the pinned Python implementation and the ordinary installed public API.

## Two unstable intermediates

[Pinned losses.py:330](https://github.com/ml-explore/mlx/blob/ce916dbbcaa88e433b6fd1e60a17f766d49c27fe/python/mlx/nn/losses.py#L330)
computes the residual square before dividing by variance:

```python
loss = 0.5 * (mx.log(vars) + mx.square(targets - inputs) / vars)
```

FP16's largest finite value is 65504. Thus `300*300=90000` overflows before
division by 300, even though the final loss is representable. By contrast,
`mean=400,target=0,v=1` has loss 80000, which truly cannot fit in FP16.

The backward problem remains even when the residual square is finite.
[Divide::vjp at the source pin](https://github.com/ml-explore/mlx/blob/ce916dbbcaa88e433b6fd1e60a17f766d49c27fe/mlx/primitives.cpp#L1806)
calculates the denominator contribution in this order:

```text
bar_b = -(cotangent * numerator) / square(denominator)
```

For residual 20 and variance 300, the forward residual square is 400, but
the backward variance square overflows. The negative contribution vanishes,
leaving approximately `1/(2*300)`, although the correct gradient is:

```text
(300-400)/(2*300*300) = -1/1800 = -0.000555555555...
```

The algebraic formula is correct; its finite-precision evaluation is unstable.
For small variance the square instead underflows to zero. At residual zero,
the resulting `0/0` contaminates the gradient with NaN. For a nonzero residual,
the quotient can become infinite although the true derivative fits in FP16.

Direct primitive checks reproduce the same mechanism without Gaussian NLL:

| Type; a=b | a/b | Native derivative with respect to b | Reference |
|---|---:|---:|---:|
| FP16; 300 | 1 | -0 | -0.003333333333 |
| FP16; rounded 0.0001 | 1 | -Inf | -9998.340882 |
| FP32; rounded 1e20 | 1 | -0 | -9.999999800e-21 |

The FP16 control `a=2,b=4` returns the correct derivative -0.125. These are
related manifestations, not a claim of a separate defect for every failed input.

## Independent forward check

At `mean=20,target=0,v=300`, central differences of the actual FP16 forward
returned -0.00048828125, -0.00048828125, -0.00054931640625 and
-0.000579833984375 for steps 2, 8, 16 and 32 respectively. All are negative;
autograd is positive. Steps are large enough to observe the quantized forward
and are not presented as an exact derivative of a continuous FP16 function.

## A bounded mitigation, with its counterexample

[gaussian_candidate.py](gaussian_candidate.py) promotes the inputs to FP32
before subtraction and computes the squared normalized residual `r/sqrt(v)`.
The original passes 4 of 12 regression methods; this candidate passes 12/12.
The tests cover large and small values, gradient signs, controls, reductions,
the constant term, shapes and the below-epsilon branch.

For FP16 `300,0,300`, the candidate gives loss 152.8518829346 and variance
gradient -0.498291015625 (the gradient is rounded back to FP16). For
`20,0,300`, it gives -0.0005555152893.

**This is not a universal fix.** The FP16 loss becomes FP32 and epsilon has
FP32 precision, changing the dtype/epsilon contract. At FP32
`mean=variance≈3e38,target=0`, the candidate still returns variance gradient
`-Inf` instead of -0.5, although its loss is finite. This counterexample is
retained in the full measurements and lies outside the 12-test passing subset.
Simply reordering the forward term to `r*(r/v)` also fails to repair backward.
A robust upstream solution needs an explicit supported domain, a stable
backward rule and wider testing. No upstream-ready patch is claimed here.

## Reproduce

Measured on macOS 15.5 arm64, Python 3.12.14, native MLX 0.32.2, CPU.
Use a supported Apple Silicon MLX environment and a Python 3.12 virtualenv:

```sh
python -m pip install mlx==0.32.2
python minimal_mlx_repro.py
python test_mlx_regressions.py --variant original
# Expected: 4 passed / 8 failed; exit code 1.
python test_mlx_regressions.py --variant candidate
# Expected: 12 passed; exit code 0, within the stated subset.
```

The regression script loads the included pinned Python source against the
installed native core. The minimal script separately uses the ordinary
installed API. For the extended measurement probe, set `MLX_REPO` to a local
MLX checkout at `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`, then run
`python review_mlx.py`. Its source pin is asserted. The public probe changes
only checkout path configuration from the retained local probe. Its output
files are written next to the script.

The native core checksum matches its installed wheel RECORD; the installed
Gaussian function's AST matches the pinned function. This is not a clean
native-core build of that commit. GPU/Metal kernels, complete training runs,
other devices, all input ranges and the latest upstream HEAD were not tested.
No claim is made that all mean/target gradients are wrong: they are correct
in the originally reported 300/300 example.

The exact publication minimal script and both 12-test selections were rerun
before publishing: original 4/12, candidate 12/12.
[Publication rerun results](evidence/publication-rerun.json).

## Related negative result: nntrainer Upsample

A separate CPU FP32 check at nntrainer `a7ea056e79ab8e14447ea305c1b634e233343258`
passed the original 18 tests and 72 additional nearest/bilinear cases against
an independent dense interpolation matrix and its transpose. Maximum observed
absolute forward/backward errors were 5.4241e-7 and 1.2398e-6, within the stated
tolerance `5e-5 + 3e-6*abs(reference)`. No new defect was established in that bounded area. The
[C++ probe](evidence/review_upsample.cpp) and
[matrix results](evidence/upsample-matrix-results.jsonl) are retained as an
appendix; this reused an existing configured build with unrelated prior repairs.
The interpolation source matched its pin. Fractional scales, NHWC, FP16,
multiple inputs, GPU and full models were not evaluated.

## Publication boundaries and provenance

[SOURCES.md](SOURCES.md) records the public-source and duplicate-review scope.
The two publication-day GitHub searches are limited searches, not proof of
novelty. There is no maintainer confirmation, accepted upstream fix, measured
full-model damage, device-safety conclusion or established bounty eligibility.

Evidence files replace workstation-specific absolute paths with labeled
placeholders. Numerical values are unchanged. Original local file hashes are
recorded in [evidence-manifest.json](evidence-manifest.json); published bytes
have their own [SHA256SUMS](SHA256SUMS). Included MLX source is under the
[original MIT license](MLX-LICENSE.txt).

Prepared with AI assistance. The checks are algorithmically independent
references and actual native executions, not a third-party laboratory review.
Independent reproduction and scoped numerical-correctness review are welcome.

#MachineLearning #NumericalStability #Autodiff #MLX #SoftwareTesting
