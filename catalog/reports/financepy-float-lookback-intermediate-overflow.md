# FinancePy floating lookback calls: a finite price becomes unavailable

20 September 2026 · Independent GERO research by Xamit Kadirbekov · AI-assisted investigation and preparation.

A valid floating-strike lookback call worth approximately **54.031435058324617** fails in FinancePy's public equity and FX pricing functions. An intermediate power overflows before it is multiplied by a tiny normal probability. A narrow local correction restores the finite price; restoring the original source restores the failure.

This is **one implementation defect shared by two products**, not 52 independent bugs. [Developer issue #272](https://github.com/domokane/FinancePy/issues/272) was submitted before this publication. No developer acknowledgment or accepted correction is claimed.

## Exact example and executed consequence

The one-year input is spot `S=100`, historical minimum `M=40`, continuously compounded domestic/risk-free rate `r=0.02`, dividend/foreign rate `q=0.07`, and volatility `sigma=0.01`. Dates are 1 January 2026 to 1 January 2027, using ACT/365F.

| Implementation | Public equity call | Public FX call |
|---|---|---|
| Current master | `OverflowError` | `OverflowError` |
| Official PyPI 1.1.2 wheel | `OverflowError` | `NaN` with warnings |
| Current source plus local candidate | `54.03143505832461` | `54.03143505832461` |
| Restored current source | `OverflowError` | `OverflowError` |

The independently calculated value is about `54.0314350583246172724…`, using the exact binary64 inputs and effective curve rates recorded in the evidence. Scaling spot/minimum to `1 / 0.4` also reproduces the failure.

The measured chain is:

**Overflowing intermediate power → the actual public pricing function raises or returns NaN → a synthetic quote document records UNAVAILABLE instead of READY.**

The quote adapter is explicit GERO demonstration code consuming the actual native result. It records `UNAVAILABLE → READY → UNAVAILABLE` for original, candidate and restored variants. It is not a FinancePy order-management subsystem or a real bank quote. No trade, invoice, client exposure or monetary loss was measured.

## Versions and environment

Executed source: [`ee5749b9693f3b7c33f71e091a027402c3d3b303`](https://github.com/domokane/FinancePy/tree/ee5749b9693f3b7c33f71e091a027402c3d3b303). Public APIs are `EquityFloatLookbackOption.value()` and `FXFloatLookbackOption.value()`.

The official [FinancePy 1.1.2 wheel](https://pypi.org/project/financepy/1.1.2/) has SHA-256 `3c32578b81f338741ac135bb05ff9aa9164d75f6aa89c4d5e7f6a96c8b6f37d9` and was executed, not merely inspected.

The release was tested with its supported Numba 0.62.1 environment. Current source declares Numba 0.67.x, so a separate current/candidate/restored run uses **Numba 0.67.0 and llvmlite 0.49.0**, satisfying that requirement. Both use Python 3.12.14, NumPy 2.3.5 and SciPy 1.16.3 on macOS ARM64. Current source has the same failure counts and candidate preservation results in both environments. Small cross-environment differences in otherwise finite results are retained, rather than described as byte-identical. Full version receipts are included.

One configured CPU worker was used; no GPU or audio playback. These controls do not impose an OS-wide one-core limit.

## Why the price must remain finite

For the floating call, the payoff is `S_T - min(M, inf S_t)`, with `M <= S_0`. It is nonnegative and bounded above by `S_T`, so its discounted expectation is finite under the tested geometric Brownian model.

The implementation forms `(S/M)^(-w) * Phi(z)`, where `w = 2(r-q)/sigma²`. Here `w` is close to `-1000`: `2.5^1000` overflows binary64, although the combined term is tiny. The preceding limiting branch checks `S < M`, while public validation requires `M <= S`. That guard therefore cannot handle a valid floating-call input.

The candidate replaces this unreachable branch with a log-space product when `log_weight = -w*log(S/M) > 500`:

```python
z = -a1 + 2*b*sqrt(T)/sigma
weighted_cdf = exp(log_weight + scipy.special.log_ndtr(z))
term = weighted_cdf - exp(b*T)*Phi(-a1)
```

The cutoff selects a safe numerical evaluation path; it does not discard the probability term. Other branches remain unchanged. This is a local candidate, not a complete stability correction for all lookback cases.

## Independent mathematical reference

Write `Y_t = log(S_t/S_0) = mu*t + sigma*W_t`, with `mu = r-q-sigma²/2`, and `a = log(S_0/M)`. The Brownian reflection formula gives the probability of reaching level `-b` by expiry:

```text
H(b) = Phi((-b-mu*T)/(sigma*sqrt(T)))
     + exp(-2*mu*b/sigma²)*Phi((-b+mu*T)/(sigma*sqrt(T))).
```

Integrating the minimum's hitting distribution gives a reference independent of FinancePy's lookback expression:

```text
price = S_0*exp(-q*T) - M*exp(-r*T)
      + S_0*exp(-r*T)*integral[a,infinity](exp(-b)*H(b) db).
```

The oracle uses 65-digit mpmath quadrature, with a monotonic hitting-probability bound on each omitted tail. Seven 100-digit repeats agree. For the main example, the nonnegative extremum correction is about `1.52e-1634`; a simpler independent bound places it below `1.4e-1630`. These are numerical convergence and bound checks, not a formal floating-point proof.

## Executed verification

- **600 native public calls per variant:** 420 floating-call oracle comparisons and 180 put preservation controls.
- **52 / 0 / 52 / 52 failures** for current / candidate / restored / official release. These are 26 input combinations exercised through two products.
- All **548 previously finite rows remain byte-identical within each tested current/candidate environment**. Original and restored outputs are identical within each environment. Put rows are preservation controls, not a separate complete put-pricing oracle.
- The same fixed tolerance, `2e-6*max(1,S,M)`, applies to every call variant, respecting the documented six-decimal normal-CDF approximation. No candidate call exceeds that tolerance on the grid.
- Four new finite-price regressions fail on original/restored/released code and pass with the candidate. Four existing upstream **fixed-strike** lookback tests pass on current and candidate; they are neighboring-product controls, not upstream floating-lookback regression coverage.
- Clean patch application yields the exact tested candidate files. A fresh source/cache replay reproduced the original four 600-row native files byte-for-byte. A separate dependency-compatible current replay retains its own outputs and checks.

The grid covers spots 1 and 100; minimum fractions 0.1, 0.25, 0.4, 0.5, 0.8, 0.99 and 1; volatilities 0.005, 0.01, 0.02, 0.05 and 0.2; three nonzero-carry rate pairs; and one-year expiry.

## Prior art and limitations

The bounded review inspected 260 issue/PR bodies, 118 fully paginated PR file lists, 51 relevant source-history snapshots, plausible matching diffs, the changelog and GERO catalogues. A fresh prepublication check found only our issue #272 for the focused lookback query and no earlier GERO lookback report. No exact prior public report or proposed correction was found in that review; this is not a global priority claim.

The existing source TODO about tightening the `w=100` cutoff is acknowledged. We do not claim that general numerical concerns were previously unknown. The specific finding is a validated floating-call input reaching an overflowing power through an unreachable guard.

Existing put cutoffs, near-equal-rate perturbation, zero-volatility behavior, ordinary CDF approximation, extreme parameter ranges and complete lookback stability are outside scope. No full upstream test suite, performance benchmark, real portfolio, bank integration or production customer impact was tested. See the README, source ledger, exact environments, raw outputs and candidate patch in the accompanying archive.

Original GERO report text: CC BY 4.0. Research scripts and FinancePy-derived candidate code: GPL-3.0-or-later. Original upstream notices and licenses are retained.

## Public records and delivery status

- [github](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/financepy-float-lookback-intermediate-overflow.md)
- [gero](https://www.gero.uz/research/articles/financepy-float-lookback-intermediate-overflow.html)
- [zenodo](https://zenodo.org/records/22858372)
- [linkedin](https://www.linkedin.com/feed/update/urn:li:share:7507454926776815616/)
- [huggingface](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/financepy-float-lookback-intermediate-overflow.md)

[Complete evidence ZIP](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/raw/refs/heads/main/reports/financepy-float-lookback-intermediate-overflow/gero-financepy-float-lookback-evidence-2026-09-20.zip). SHA-256: `fbe433c8f98cec540e507acb1d1a7f82bb46ad3b2eda360f93773d19663e53b4`.

Pending distribution: youtube. The verified developer report is [FinancePy #272](https://github.com/domokane/FinancePy/issues/272).
