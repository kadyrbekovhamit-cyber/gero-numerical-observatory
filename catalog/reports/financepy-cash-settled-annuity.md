> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/financepy-cash-settled-annuity.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# FinancePy cash-settled swaption annuity counts N−1 payments instead of N

[Zenodo DOI 10.5281/zenodo.22740980](https://zenodo.org/records/22740980) · [Maintainer correspondence: issue 262](https://github.com/domokane/FinancePy/issues/262) (submitted; no acceptance claimed).

[GERO article](https://www.gero.uz/research/articles/financepy-cash-settled-annuity.html) · [GitHub code and reproducer](https://github.com/kadyrbekovhamit-cyber/gero-financepy-cash-annuity-audit) · [LinkedIn video post](https://www.linkedin.com/feed/update/urn:li:ugcPost:7505114181160046592/)

[40-second English video](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/financepy-cash-settled-annuity.mp4) · [Captions](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/financepy-cash-settled-annuity.en.vtt) · [Video source and voice provenance](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/financepy-cash-annuity-video-source.zip) · [Frozen evidence ZIP](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/gero-financepy-cash-annuity-evidence-2026-09-14.zip).

Narration: Microsoft en-US-JennyNeural through edge-tts, rate −3%, using the same client as the Ingenium reference film. Original GERO graphics. Archive SHA-256: `32d15e39b54cbf07a88c3ceb744c97e34f4acb9bea5dd05a2376775a79e9d7c6`.


Independent GERO investigation by Xamit Kadirbekov, 14 September 2026. Synthetic inputs; AI-assisted analysis and verification. Local candidate correction, not an accepted upstream fix. Public edition prepared 14 September 2026. The original frozen evidence archive preserves its preparation-stage labels.

## Finding

`IborSwap.cash_settled_pv01()` returns zero for a swap with one strictly future fixed payment when valuation is on or before the swap's effective date. Its `payment_dts` list already contains payment dates only, but the method sets `start_index = 1` as though the effective date were an extra list entry. With N payments it therefore sums N−1 discount terms. Because the remaining terms are reindexed from the first period, the numerical result is A(N−1), not simply A(N) minus the first payment's present value.

This reproduces in the **FinancePy 1.1.2 PyPI wheel** and a fresh checkout of upstream `master`, **`2b9227fea9d832c4033421d6cd53a54316414fca`**. All 522 observed numerical outputs agree exactly between the release and source baseline. The affected function's parsed syntax tree is identical in both. The repository project metadata says 1.1.2; its import banner still says 1.1.0, so the commit and wheel hash identify the tested code more precisely than that banner.

Relevant pinned source:

- [The annuity calculation](https://github.com/domokane/FinancePy/blob/2b9227fea9d832c4033421d6cd53a54316414fca/financepy/products/rates/ibor_swap.py#L316).
- [Fixed-leg payment generation](https://github.com/domokane/FinancePy/blob/2b9227fea9d832c4033421d6cd53a54316414fca/financepy/products/rates/swap_fixed_leg.py).
- [The public cash-settled swaption pricing method](https://github.com/domokane/FinancePy/blob/2b9227fea9d832c4033421d6cd53a54316414fca/financepy/products/rates/ibor_swaption.py#L287).

## Mathematical contract

For N regular payments at frequency m per year and a flat swap rate s, the conventional forward cash annuity is

`A(N,m,s) = Σ[j=1..N] (1/m) / (1+s/m)^j`.

At zero rate this is exactly `N/m`. For one annual payment at 4%, the answer is `1/1.04 = 0.961538461538…`, while the tested baseline returns **0**. No date adjustment, ex-coupon entitlement, tiny rate or floating-point overflow is needed to trigger this failure.

Here `pv01` means an annuity factor. The maintainer explains this convention in [issue 227](https://github.com/domokane/FinancePy/issues/227#issuecomment-3175185832). The finding is not the separate question of scaling that factor into dollar PV01. The [OpenGamma Strata documentation](https://strata.opengamma.io/apidocs/com/opengamma/strata/pricer/swap/DiscountingSwapLegPricer.html#annuityCash-int-int-double-) also describes a conventional cash annuity parameterized by payment frequency, number of periods and yield. Strata was consulted as documentation, not executed as a numerical reference.

`minimal_repro.py` calls the real FinancePy API and prints its generated payment dates. The independent annuity oracle in `reproduce.py` sums discounted payments with 80-digit Decimal arithmetic.

## Measured effect on an option price

The downstream test calls the real `IborSwaption.cash_settled_value()` using its Black model. Assumptions shared by both rows: valuation/constructor settlement 15 January 2026; exercise 15 January 2027; forward swap rate and strike both 4%; volatility 25%; flat continuously compounded discount rate 3%; synthetic notional 1,000,000; no business-day adjustment.

| Underlying swap after exercise | Released/source baseline | Candidate | Independent annuity + erf-based Black |
|---|---:|---:|---:|
| One annual payment, maturity 15 January 2028 | 0.000000 | 3,712.944454 | 3,712.941395 |
| Ten semiannual payments, maturity 15 January 2032 | 15,759.084387 | 17,342.956375 | 17,342.942085 |

Values are in the currency units of the supplied notional; the API does not select a currency. This is a demonstrated difference in a synthetic public pricing API, not an observed trading loss, bank exposure or client transaction. Both payer and receiver options were exercised in the grid.

## Validation results and the remaining approximation residual

| Check family | Observations | Baseline failures | Candidate failures |
|---|---:|---:|---:|
| Annuity, valuation before effective date | 120 | 120 | 0 |
| Annuity, valuation on effective date | 120 | 120 | 0 |
| Annuity, one day after effective date, before first payment: control | 120 | 0 | 0 |
| Swaption against independent annuity, exponential discounting and erf-based Black | 162 | 162 | **7** |

The primary 360-case grid covers annual, semiannual, quarterly and monthly payments; 1, 2, 3, 10 and 60 payments; negative, zero, tiny positive and ordinary positive flat rates. Candidate maximum absolute annuity error is `1.9185e-13`.

The seven downstream residual failures are retained, not hidden by loosening the tolerance. The initial comparison used absolute tolerance 0.02 or relative tolerance 1e-6. After the annuity correction, the largest residual is **0.04389523 per 1,000,000 notional**. It is explained by the existing FinancePy normal-CDF approximation in the Black kernel. This is not presented as a new CDF defect.

An added diagnostic holds that Black kernel fixed and composes it with the independent, complete annuity and exponential discount factor. **All 162 composition checks pass**, with a maximum absolute residual of `2.2410e-9`. This isolates removal of the annuity error, but it is not 162 wholly independent Black-model verifications. The original failed run, original harness and protocol amendment are retained. There is no claim that all 522 strict independent-oracle observations pass after this patch.

Further checks:

- The complete configured upstream unit suite passes: **972 tests**, comprising 958 existing tests and 14 new regressions, no failures/skips, four numerical-fit warnings. Upstream configuration excludes the legacy directory.
- Restoring the original production file makes **all 14 added regressions fail by assertion**, and returns exactly the original 522 numerical outputs, including the 402 failed strict-oracle comparisons. Restoring the candidate returns the 14 regressions to passing.
- The candidate changes only the incorrect pre-start payment skip in one production file and adds the regression test file. The patch applies cleanly to the pinned baseline.
- Execution used the actual Python package and its normal Numba-backed models, with no mock implementation or monkeypatch of the pricing API. An existing dependency environment was reused; it was not a newly provisioned machine. Source paths and versions are recorded.

## Duplicate review

On 14 September 2026, screening covered **251 public issue/PR title-and-body records**, open and closed, plus focused GitHub queries for `cash`, `cash_settled_pv01`, `swaption` and `annuity`. Query result counts were respectively **10, 0, 13 and 2**; GitHub did not flag incomplete results. Selected related comments and the affected file's 28 returned commit summaries were reviewed.

Nearby reports have different causes: #227 concerns annuity units; #69 concerns end-of-month schedule generation; #252 concerns Hull–White payer/receiver mapping; #157 concerns ownership of coupons on settlement day. This reproduction uses strictly future coupon dates. Previous local FinancePy and Fineract/PyXIRR findings were excluded from the new-candidate list.

**No direct duplicate was identified within that bounded search.** This is not proof of first discovery or a search of private/deleted reports. See `DUPLICATE_REVIEW.json`; full raw search responses remain in the local work directory.

## Reproduce

Use Python 3.12 and install `requirements.txt` in an isolated environment. For the short released-package reproduction, install `financepy==1.1.2` and run `python minimal_repro.py`. For the complete evidence package:

```sh
python reproduce.py --source source-baseline --output results/my-baseline.json --expect affected
```

Make a separate copy of `source-baseline`, apply `candidate.patch` from within that copy using `git apply`, and run:

```sh
python reproduce.py --source source-corrected --output results/my-candidate.json --expect corrected
cd source-corrected
python -m pytest -q unit_tests
```

The `--expect corrected` gate requires all independent annuity and additional composition checks to pass; it still records the stricter independent Black residuals. Read both the `groups` and `composition` fields in the JSON. Dependencies, platform, source hashes, XML results and the original failed comparison are included.

## Boundaries

The tested scope is regular fixed schedules and this flat-rate cash-annuity/Black cash-settlement path. No claim is made about physical settlement, every market convention, irregular stubs, matured swaps, options valued after expiry, changing notionals, calibrated market data, production systems or actual customers. The candidate intentionally does not resolve other methods' settlement-day or expiry behavior. No upstream acceptance or financial-loss claim is implied.

Upstream source and the compatible patch/tests retain GPL-3.0-or-later licensing. Reports and original audit scripts are also offered under GPL-3.0-or-later for this evidence package. No affiliation with FinancePy or OpenGamma is implied.
