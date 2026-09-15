# FinancePy CIR zero-coupon pricing loses its finite range and small-volatility limit

Independent GERO research, 15 September 2026. Author: Xamit Kadirbekov. AI-assisted investigation and preparation. This report documents executed local tests of the real implementation. Verified publication links are listed below.

## Result

The actual Numba-compiled `financepy.models.cir_montecarlo.zero_price` returns **2.2407257971155513e+96** for a unit zero-coupon payment whose model price is **0.6882687528140472**. The inputs are `r0=.03, a=.1, b=.05, sigma=1e-10, t=10`. The initial and long-run rates are nonnegative, so a unit payment's discounted value cannot exceed one.

The same result was observed in current pinned source and the separately executed PyPI 1.1.2 wheel. An algebraic reformulation removes **1,168 → 0** failing prices from a predeclared grid of **4,536 distinct parameter vectors**. Restoring the original source restores the same **1,168** failures. The failure count describes this synthetic grid, not a frequency in financial users' workloads.

## Executed versions

- Current upstream `master`: [`2b9227fea9d832c4033421d6cd53a54316414fca`](https://github.com/domokane/FinancePy/commit/2b9227fea9d832c4033421d6cd53a54316414fca), checked again on 15 September. The source package prints a historical 1.1.0 banner; the exact commit is the source identity.
- Released package: PyPI **1.1.2**, separately extracted and imported. Its 219 package files match the official wheel, whose SHA-256 is `3c32578b81f338741ac135bb05ff9aa9164d75f6aa89c4d5e7f6a96c8b6f37d9`.
- Target: [`cir_montecarlo.py`, function `zero_price`](https://github.com/domokane/FinancePy/blob/2b9227fea9d832c4033421d6cd53a54316414fca/financepy/models/cir_montecarlo.py). The current raw file and released target are byte-identical to the executed baseline.
- macOS 15.5 arm64, Python 3.12.14, NumPy 2.3.5, Numba 0.62.1, SciPy 1.16.3, mpmath 1.3.0. Numba recorded a native nopython signature with five float64 inputs and a float64 result. This was not a rewrite of the implementation in an oracle script.
- One configured numerical worker, sequential processes, no GPU or audio playback.

## Mathematical convention and independent checks

The risk-neutral short rate follows `dr = a(b-r)dt + sigma*sqrt(r)dW`, with `a>0`, `r0,b,sigma,t>=0`. The price of one unit at maturity is `P(t)=E[exp(-integral_0^t r(s)ds)]`. Thus `0<=P<=1`, `P(0)=1`, and the absorbing case `r0=b=0` has price one.

At zero volatility the deterministic solution gives

`P0(t) = exp(-b*t - (r0-b)*(1-exp(-a*t))/a)`.

For each exact stored binary64 input vector, `oracle.py` evaluates the direct affine closed form at **80** and **120 decimal digits**. All 4,536 pairs agree within `1e-55` absolute error before rounding to float64. No FinancePy output is used as an expected value.

As a separate check, 48 parameter vectors were evaluated by integrating the affine Riccati equations `B'=1-aB-sigma^2*B^2/2`, `(log A)'=-abB`, with initial values zero, using DOP853. The maximum price difference from the high-precision oracle was **7.8826e-15**. That solver does not use closed-form affine coefficients. These 48 checks validate the oracle through a different route; they are not added to the 4,536 grid count.

The fixed price tolerance, declared before candidate execution, is `2e-12 + 2e-12*abs(reference)`. Bounds use `[-2e-15,1+2e-15]`. No tolerance was relaxed.

## Why the evaluation fails

Three related numerical regimes occur in the same pricing function:

1. At small positive volatility, a base close to one is raised to a power proportional to `1/sigma^2`. Floating-point error in the base is greatly amplified. The headline case contains no extreme rates or maturity, but its volatility is a deliberately small stress input.
2. At large `sqrt(a^2+2*sigma^2)*t`, intermediate positive exponentials overflow although the final price is finite and representable.
3. The separate zero-volatility branch subtracts `exp(-a*t)` from one, losing accuracy at small `a*t`.

These are numerical evaluation defects in a correct analytical pricing model. They are presented as one component report, not three independent discoveries of a new financial formula.

| Inputs `(r0,a,b,sigma,t)` | Original / release | 120-digit oracle rounded to float64 | Candidate |
|---|---:|---:|---:|
| `(.03,.1,.05,1e-10,10)` | `2.2407257971155513e96` | `0.6882687528140472` | `0.6882687528140472` |
| `(.03,10,.05,.1,100)` | `NaN` | `0.006753121072037893` | `0.0067531210720379` |
| `(0,1e-8,.2,0,10)` | `0.9999998990272212` | `0.9999999000000084` | `0.9999999000000080` |

## Candidate correction

Let `h=sqrt(a^2+2*sigma^2)`, `u=1-exp(-h*t)` and `x=sigma^2*u/[h(h+a)]`. Evaluate `h` with `hypot` and `u` with `expm1`. Algebraically,

`B = (u/h)/(1-x)`

`log(A) = [2ab/(h+a)] * [(u/h)*(-log(1-x)/x)-t]`.

The ratio `-log(1-x)/x` has limit one at zero, handled explicitly. `log1p` evaluates its numerator. This form has no positive exponential of `h*t` and no division by `sigma^2`; it also extends to `sigma=0`. The implementation evaluates `x` as a product of ratios to avoid forming an unnecessary squared volatility.

The candidate does not clip prices or replace small nonzero volatility with zero. It changes only `zero_price` in one source file and retains the existing parameter validation. It has been validated on the stated domain/grid; this is not an accuracy guarantee for every possible finite float64 argument.

## Verification

| Check | Original source | Candidate | Restored source | PyPI 1.1.2 |
|---|---:|---:|---:|---:|
| Price errors / 4,536 vectors | 1,168 | 0 | 1,168 | 1,168 |
| Range failures, including nonfinite output | 224 | 0 | 224 | 224 |
| NaN / infinity | 192 / 3 | 0 / 0 | 192 / 3 | 192 / 3 |

Categories overlap and must not be added together. The remaining 29 original range violations are finite prices above one. Maximum candidate absolute error was **2.9976e-15**.

All **3,368** previously passing grid prices still pass the unchanged tolerance. Of those, **1,165** changed binary value, so a claim that all ordinary outputs were byte-identical would be false. Forty upstream tests from the two CIR test files pass on both original and candidate. Those tests include their own limited Monte Carlo checks; this study does not establish Monte Carlo accuracy or real-world financial impact.

A second complete run from independently copied sources and fresh Numba caches reproduced all four 4,536-row result JSON files **byte-for-byte**. All **230 baseline source files** and **219 released package files** remained unchanged. Candidate/restored-source comparisons confirm that the only candidate source change is the intended pricing function. See `evidence/paired-verification.json`, source manifests, and fresh-run receipts.

## Duplicate review and limitations

The bounded review covered 253 upstream issue/PR title-body records in the earlier recorded review, four focused searches, the relevant returned discussions (#23 and #167), 12 target-file history entries and the live 97-record GERO catalog. No exact duplicate was found. The related search hits concern a tree feature request and equity finite differences. The search is documented in `DUPLICATE_REVIEW.md`; it is not a worldwide priority guarantee, and unrelated issue comments were not exhaustively reviewed.

There was no full FinancePy suite, calibration, Greek, portfolio, performance or production-bank evaluation. No claim is made about customer losses, deployed bank systems, or the frequency of the stress inputs. The correction is a local candidate, without upstream acceptance. Upstream acceptance is not claimed; submission links, when verified, are listed below.

Reproduction instructions: `REPRODUCE.md`. Preserve the original baseline, raw results and immutable archive when preparing an external report.

## Immutable archive and review timing

The frozen research ZIP retains its preparation-time statement that publication was pending. That is historical metadata, preserved with the original evidence. Publication status is established by the external links below. A later focused duplicate check found no exact match; live issue pagination returned incomplete subsets, so it is not represented as a new exhaustive review.

## Publication links

[GERO](https://www.gero.uz/research/articles/financepy-cir-zero-price-stability.html) · [GitHub](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/financepy-cir-zero-price-stability.md) · [Hugging Face](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/financepy-cir-zero-price-stability.md) · [LinkedIn](https://www.linkedin.com/feed/update/urn:li:share:7505637773324976128/) · [YouTube](https://www.youtube.com/shorts/yyDsUwfKW70) · [Maintainer issue](https://github.com/domokane/FinancePy/issues/264)

[Immutable evidence archive](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/artifacts/gero-financepy-cir-zero-price-research-2026-09-15.zip)

SHA-256: `1500cda3ca46ddfc02e19099b8b57c12b5c674936e219a27bc4e9f06de5bb3db`.
