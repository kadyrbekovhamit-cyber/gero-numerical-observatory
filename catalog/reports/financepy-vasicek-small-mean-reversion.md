# FinancePy Vasicek pricing loses the small-mean-reversion limit

Xamit Kadirbekov · GERO Research · 16 September 2026

The executed FinancePy `models.vasicek_mc.zero_price` returns infinity for a
finite zero-coupon bond price of approximately **0.6167242197654975**. The input
mean-reversion parameter is positive: `a=1e-8`. At `a=1e-7`, the same component
returns **1.3630600530447776** instead of **0.6167242683325149**.

The other inputs are `r0=0.05`, `b=0.03`, `sigma=0.01`, and `t=10` years. These
are synthetic stress conditions, not an observed bank calibration or customer
portfolio. Very slow mean reversion is a numerical boundary regime; its
frequency in production has not been measured.

## Executed implementations

- Executed official master snapshot: [`2b9227fea9d832c4033421d6cd53a54316414fca`](https://github.com/domokane/FinancePy/blob/2b9227fea9d832c4033421d6cd53a54316414fca/financepy/models/vasicek_mc.py).
- Official PyPI **1.1.2** wheel, verified against the SHA256 in current PyPI
  metadata. The target module is byte-identical to master. The whole source tree
  and wheel are separately fingerprinted; they are not claimed identical.
- Actual Numba CPU implementation, preserving the existing `fastmath=True`
  decorator. Python 3.12.14, NumPy 2.3.5, Numba 0.62.1, mpmath 1.3.0,
  macOS 15.5 on arm64; one configured numerical thread.

Exact wrong outputs can depend on the compiler and math library. The reference
price and the need for finite output in these stated cases do not depend on the
observed direction of floating-point cancellation.

## Independent mathematical check

For the rate dynamics documented in the module,

`dr = a (b-r) dt + sigma dW`,

the integrated rate is Gaussian. Let

`B(s) = (1 - exp(-a*s)) / a`, continuously extended by `B(s)=s` at `a=0`.

Its integrated mean is `b*t + (r0-b)*B(t)` and integrated variance is
`sigma^2 * integral_0^t B(s)^2 ds`. The bond price is therefore

`exp(-integrated_mean + integrated_variance / 2)`.

The primary oracle evaluates these moments at 80 and 120 decimal digits, using
the exact binary values of the input floats. All rounded float64 reference
values agree at both precisions. **48 additional numerical quadrature checks**
evaluate the variance integral independently at both precisions and give the
same rounded results. The oracle is in `oracle.py`; it does not call FinancePy.

At zero mean reversion the limit is `exp(-r0*t + sigma^2*t^3/6)`, independent of
`b`. At zero volatility and `r0=b`, it is `exp(-r0*t)`. At zero maturity it is 1.

A Vasicek discount factor is **not a probability**. Gaussian rates can become
negative, so a price above 1 is not in itself a defect. The discrepancies here
are established against the mathematical price for the same inputs.

## Measured results

The primary grid contains **1,350 distinct scenarios with strictly positive
mean reversion**. An additional **135 zero-mean-reversion limit cases** are
reported separately. The fixed acceptance threshold was
`2e-11 * max(1, abs(reference))`, selected before the four-variant matrix.

| Implementation | Positive-a failures / 1,350 | Zero-a failures / 135 |
|---|---:|---:|
| Original pinned source | 527 | 135 |
| Local candidate correction | 0 | 0 |
| Restored original source | 527 | 135 |
| Official PyPI 1.1.2 | 527 | 135 |

The 527 positive-a failures include **72 non-finite outputs** and **45 zero
outputs** where the reference price is positive and finite. These are subsets
of the 527, not additional independent defects.

All **823 previously passing scenarios** remain within the fixed tolerance;
**495** retain their exact binary output. The other 328 change by small amounts
within tolerance. The corrected series branch intentionally changes the
evaluation formula, so unchanged behavior is not claimed bit for bit everywhere.

The original, restored and released variants produce identical raw rows on the
tested environment. Original source fingerprints remain unchanged; only the
candidate target module differs.

## Candidate correction and regression checks

For `abs(a*t) < 0.5`, the candidate evaluates `B` using `expm1` and the
integrated variance with a 20-term polynomial in `a*t`. The coefficients come
from integrating `(1-exp(-x))^2`; they are not fitted to the test data. The
series has a continuous value at `a=0`. The conventional branch outside this
region is retained.

For `|a*t| <= 0.5`, the exact-arithmetic omitted series tail is less than
`2e-22` for the dimensionless variance factor. Floating-point evaluation error
is checked by the executed oracle matrix; this tail bound alone is not a
floating-point error proof.

One existing Vasicek test plus 18 new regressions: **19 pass** on the candidate.
The original and restored versions give **11 failures and 8 passes**. Tests
cover small positive mean reversion, the zero-reversion limit, deterministic
constant rates and both sides of the branch boundary. The existing test also
contains its original seeded Monte Carlo comparisons; the Monte Carlo routines
were not modified or independently audited in this investigation.

`git apply` to a clean source copy produces the exact candidate bytes. Final
pytest entry points explicitly assert the imported source path. Exploratory
`*-initial-import-run.log` files record an earlier test-launcher path mistake;
they are retained for provenance and excluded from the final regression result.

## Duplicate review and limits

Checked the current target source, official release, **258 upstream issue/PR
title-and-body records**, three focused GitHub issue searches, 12 target-file
history summaries, a focused web search and the current **99-report** GERO
catalog. No exact report was found in that bounded review. The already reported
[CIR zero-price case #264](https://github.com/domokane/FinancePy/issues/264)
concerns a different model and implementation and is explicitly excluded.
Numerical cancellation and stable exponential evaluation are established
techniques; this report makes no claim to discover those mathematical methods.

No full FinancePy suite, calibration study, performance benchmark, production
bank impact or customer loss was measured. The scope is this analytic pricing
function. Other functions in the Vasicek module have not been declared fixed.
The candidate is a local proposal, not an upstream-accepted change.

Investigation and preparation were AI-assisted. Upstream FinancePy code retains
its original GPLv3 license and attribution. No private records are used.

## Reproduction

Use a fresh environment with `financepy==1.1.2`, `mpmath==1.3.0` and compatible
Numba/NumPy. Run `python minimal_repro.py` against the installed package, or use
`--variant baseline`, `candidate`, `mutation`, or `release` with this research
directory. The selected environment is in `evidence/environment.json`.

To replay the complete study, run `python reproduce.py --output /path/to/new-directory`.
The output directory must not exist. The runner makes a separate working copy,
regenerates the 80/120-digit oracle cases and 48 quadrature controls, executes
four variants, checks archived output hashes, runs the 19 tests and verifies
a clean patch application. It never rewrites the frozen evidence. Dependencies
are listed in `requirements-reproduction.txt`; Python 3.12 was tested.
The original source, extracted wheel and candidate/restored copies are included.
All numerical work runs sequentially with one configured thread.

## Source refresh before archival packaging

On 16 September 2026, master advanced to `87779f5e1bd99b1a0d2eae241d2c453488b0c76d`. The target
`financepy/models/vasicek_mc.py` remains byte-identical to the executed source.
A fresh review of all 258 upstream issue/PR title-and-body records found only
the distinct CIR issue #264 among the recorded numerical-stability terms.
This is a source and duplicate-status refresh, not a full execution of the
new repository revision. See `evidence/upstream-review-2026-09-16.json`.

A separate portable working copy regenerated all 1,485 oracle rows and 48
quadrature controls, reproduced all four archived raw result files byte for
byte, and repeated the 19-test original/candidate/restored sequence. All five
minimal variant entry points and clean patch application succeeded. See
`evidence/portable-full-replay.json`. No dependencies were downloaded during
this replay.

## Publication status update - 16 September 2026

Maintainer report: https://github.com/domokane/FinancePy/issues/269 . The issue is open with no comments at the prepublication check; no acceptance is claimed. Current master remains87779f5 and the official Vasicek target has not changed. The frozen ZIP and its experimental evidence remain unchanged.

[Download the frozen reproduction archive](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/artifacts/gero-financepy-vasicek-small-mean-reversion-research-2026-09-16.zip). SHA-256: `73ae765799146c338fd253ded1c385375d4abf6362188625abda0a2836f101f5`.

Original GERO explanatory text: CC BY 4.0. FinancePy and derivative code retain GPLv3 and their original notices.

## Publication links

- [Github](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/financepy-vasicek-small-mean-reversion.md)
- [Gero](https://www.gero.uz/research/articles/financepy-vasicek-small-mean-reversion.html)
- [Huggingface](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/financepy-vasicek-small-mean-reversion.md)
- [Zenodo](https://zenodo.org/records/22801595)
- [Linkedin](https://www.linkedin.com/feed/update/urn:li:share:7506071696911609857/)
- [Youtube](https://youtube.com/shorts/k-VGcx7o_LM)

Zenodo DOI: **10.5281/zenodo.22801595**. The public34.09-second overview uses synthetic JennyNeural narration and actual speech-timed English captions. Publication links do not represent a new numerical experiment or upstream acceptance.
