# FinancePy Merton jump pricing: a zero starting probability can erase an option value

Independent GERO research by Xamit Kadirbekov, 23 September 2026. AI-assisted preparation and actual public-source execution.

## Result

At Poisson exposure `lambda*T=800`, the current FinancePy `MertonJumpDiffusion.value` returns **0.0** for a call whose independent Black–Scholes value is approximately **23.50586700**. All price inputs are finite. Setting the jump size to zero makes every jump multiplier one, so changing the event rate must not change the underlying process or its option value.

The investigation is pinned to current-source commit `7a68a8a8b3be507da6829793861d36865c98c547`. The latest PyPI **1.1.2 wheel does not contain this module**. Its version banner must not be mistaken for evidence that this defect is present in that release.

| Same call, zero-size jumps | Current source | Local candidate | Independent reference |
| --- | ---: | ---: | ---: |
| Exposure 0 | 23.50585896 | 23.50585896 | 23.50586700 |
| Exposure 740 | 23.50770001 | 23.50585896 | 23.50586700 |
| Exposure 745 | 23.95368337 | 23.50585896 | 23.50586700 |
| Exposure 800 | 0.00000000 | 23.50585896 | 23.50586700 |

Inputs: stock 120, strike 100, expiry one year, interest 0.03, dividend yield 0.01, diffusion volatility 0.2, zero jump mean and volatility. `max_jumps=4000` deliberately provides ample terms; default-budget truncation is a separate question. Small differences between the candidate and reference are the existing normal-CDF approximation, not the error counted here.

Nonzero jumps also reproduce the loss. With exposure 800, jump volatility 0.01 and log-jump mean −0.00005, the current call is 0.0, the 70-digit reference is approximately 28.14771759, and the candidate is 28.14773001.

## Cause and candidate

The algorithm initializes the Poisson mass with `exp(-lambda*T)` and generates the next mass by multiplication. At exposure 800, the initial mass rounds to zero in double precision. Every subsequent mass consequently remains zero, including near the distribution's mode. Before complete underflow, a subnormal initial mass can already distort the entire distribution.

The local candidate evaluates subsequent masses from their log PMF, using `gammaln(n+2)` for the factorial term. This allows representable probabilities near the mode to recover even when the first mass is unrepresentable. `candidate.patch` changes this probability computation only. It is a proposed correction, not an accepted upstream fix.

## Executed checks

- **112 observations:** 84 zero-jump-size call/put scenarios and 28 scenarios with nonzero jump volatility.
- **52 → 0 → 52** discrepancies for current source, candidate and restoration of the original source. Restored outputs match baseline byte for byte.
- Both baseline and candidate pass all **12 functions invoked by the existing Merton jump-diffusion regression script**. The full FinancePy suite was not run.
- Formal absolute tolerance: `2e-7 * max(S, K, 1)`. The exploratory probe's stricter `1e-6` flag also flagged the normal-CDF approximation and is not used for the final defect count.
- Python 3.12.14, NumPy 2.3.5, SciPy 1.16.3, Numba 0.67.0, llvmlite 0.49.0. One configured CPU worker; no GPU.

The independent oracle imports neither FinancePy nor SciPy. It uses mpmath at 70 decimal digits: direct Black–Scholes for zero-size jumps and a conditional expectation with independently accumulated high-precision Poisson weights for nonzero jumps. Chernoff bounds put omitted price tails below `1e-25` in those checks. This does not certify all parameters or replace a full model validation.

## Measured chain: implementation → document → decision

A GERO synthetic worksheet multiplies the exposure-800 call price by a synthetic quantity of 100 and compares it with an explicitly chosen review threshold of 2000:

| Variant | Worksheet value, rounded | Worksheet decision |
| --- | ---: | --- |
| Current | 0.00 | Below threshold |
| Independent reference | 2350.59 | Review |
| Candidate | 2350.59 | Review |
| Restored source | 0.00 | Below threshold |

The model price comes from actual FinancePy code. The CSV, quantity, threshold and review rule are demonstration logic, not FinancePy features or a real institution's policy. No customer positions, losses, deployment or regulatory violations were established.

## History, disclosure and limits

The source history examined lists the module's introduction in commit `bb10c3936e078a5212694d017746453175151c09` on 16 September 2026. A bounded review of current all-state issue searches, the recent 100 pull requests, source history and the 120-entry GERO catalog found no exact existing report or correction. FinancePy issue 263 concerns Merton **firm calibration and monetary scaling**, a different model and root cause. This search is not an exhaustive originality guarantee.

The developer report was submitted and its exact body verified at [FinancePy issue 273](https://github.com/domokane/FinancePy/issues/273). No acknowledgement or acceptance is claimed. The latest released wheel's absence of this module was checked against the official PyPI hash.

The tested inputs include deliberately high Poisson exposures. The candidate does not fix or certify payoff-weighted stopping criteria, an insufficient `max_jumps` budget, arbitrary extreme conditional prices, Greeks, calibration across all inputs, or deployed systems. These results establish a reproducible current-source numerical defect and a measured synthetic consequence.

## Sources and reproduction

- [Pinned model source](https://github.com/domokane/FinancePy/blob/7a68a8a8b3be507da6829793861d36865c98c547/financepy/models/merton_jump_diffusion.py)
- [Existing upstream regression script](https://github.com/domokane/FinancePy/blob/7a68a8a8b3be507da6829793861d36865c98c547/tests/regression/TestFinModelMertonJumpDiffusion.py)
- [Official package releases](https://pypi.org/project/financepy/#history)

Run `run_checks.py` for current/candidate/restoration and the independent oracle, `upstream_checks.py` for the 12 upstream tests, and `measure_downstream.py` for the synthetic worksheet. Raw evidence and exact dependency versions accompany the archive. Platform publication receipts are separate; a prepared file is not a completed upload.


Frozen evidence: [gero-financepy-merton-jump-poisson-underflow-evidence-2026-09-23-v1.0.zip](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/reports/financepy-merton-jump-poisson-underflow/gero-financepy-merton-jump-poisson-underflow-evidence-2026-09-23-v1.0.zip). SHA256 `6b1611caee952ec8f0ed05db46c682f49de7f7f6e35149479f84fea5c57f25df`.

Article: [GERO Research](https://www.gero.uz/research/articles/financepy-merton-jump-poisson-underflow.html).

### Verified distribution

- [Zenodo archive · DOI 10.5281/zenodo.22916584](https://zenodo.org/records/22916584)
- [Hugging Face report and evidence](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/financepy-merton-jump-poisson-underflow.md)
- [LinkedIn summary](https://www.linkedin.com/feed/update/urn:li:share:7508485824913723392/)

Video for this case has not yet been produced or published.
