# actuarialmath: a missing power erases endowment uncertainty

Independent GERO research by Xamit Kadirbekov. Runtime confirmation: 24 September 2026. AI-assisted investigation and writing; results below were obtained by executing the public Python package. This completes an internal candidate first recorded on 19 September, not a newly introduced regression.

Vendor report: [terence-lim/actuarialmath issue 8](https://github.com/terence-lim/actuarialmath/issues/8), submitted and verified on 24 September. No maintainer acknowledgment or accepted fix is claimed.

## Result

`Insurance.A_x(..., endowment=2, moment=2)` returns **1.5 instead of 3** for a one-year pure endowment with survival probability 3/4 and zero interest. The mean is correctly 1.5. Consequently, the returned moments imply the impossible variance `1.5 - 1.5**2 = -0.75`.

Current main `7d18f11ad304898f177b7922b3c53f70e4c2b4f4` and the separately downloaded, hash-verified PyPI **1.1.0** wheel behave identically in the tested cases. The repository's project metadata says 1.0.1; the commit and released wheel identify the two tested distributions separately.

```python
from actuarialmath import LifeTable
life = LifeTable().set_table(q={40: 0.25, 41: 1}).set_interest(i=0)
for m in (1, 2):
    print(life.A_x(40, t=1, benefit=lambda x, t: 0,
                   endowment=2, moment=m))
# current/release: 1.5, 1.5
# independent expectation: 1.5, 3.0
```

The payout Z is 2 with probability 3/4 and 0 otherwise. Thus E[Z] = 3/2, E[Z²] = 3, and Var(Z) = 3/4. In general, scaling a payout by c must scale its second raw moment by c².

## Cause and narrow candidate

The death-benefit term already raises the discounted benefit to the requested moment. The endowment term instead multiplies a unit-benefit moment by the endowment amount just once:

```diff
- E = self.E_x(x, s=s, t=t+u, moment=moment) * endowment
+ E = self.E_x(x, s=s, t=t+u, moment=moment) * endowment**moment
```

This change remains in the caller, because subclasses such as LifeTable do not accept an `endowment` argument in their `E_x` override. It leaves the first moment unchanged. It is a local candidate; upstream acceptance is not established.

This is distinct from [issue 6](https://github.com/terence-lim/actuarialmath/issues/6), where the separate `endowment_insurance(..., moment=VARIANCE)` convenience method applies an extra b² to already scaled moments. The present reproduction calls `A_x` directly and never invokes that variance branch. The known whole-life, ConstantForce, density and selected-duration cases are also excluded.

## Executed validation

Each source variant executes **2,880 public-method calls**:

| Variant | Disagreements |
| --- | ---: |
| Current source | 720 |
| Official release 1.1.0 | 720 |
| Local candidate | 0 |
| Restored original | 720 |

All **2,160 previously passing outputs remain bit-identical**. Release and restored numerical rows exactly match the original rows. The 720 failing observations describe one root cause, not 720 separate defects.

The 2,400 discrete calls use an independent exact `Fraction` sum over mutually exclusive death and survival payouts. Annual death probabilities are 0, 1/4, 1/2, 3/4 and 1; interest rates 0, 1/20 and 1/4; terms 1 and 3; deferrals 0 and 2; death benefits 0, 1, 2 and 10; survival benefits 0, 1/2, 1, 2 and 100; moments 1 and 2. There are 576 discrete disagreements before the change.

The 480 continuous calls use the full `Insurance` class with explicitly supplied exponential survival and density functions. Independent 80-digit Decimal integration in closed form checks mortality forces .02/.2, interest forces 0/.03, terms 1/5, deferrals 0/2, death benefits 0/1/10 and the same five survival benefits. All 480 oracles were repeated at 120 digits with the same rounded expectations. There are 144 continuous disagreements before the change.

The predeclared comparison is `abs(actual-expected) <= 2e-11 * max(1, abs(expected))`. Decimal/rational parameter literals are compared to the API's binary64 representations; this is not a correctly-rounded binary64 claim. Terms remain within the supplied survival support.

A fresh directory replay applied the patch with `git apply`, reversed it for the restoration variant, verified the complete source manifest, and reproduced all four grids and generated documents. The repository's `tests/test_changes.py` example completed for both original and candidate with identical stdout. No full upstream-suite result is claimed.

Environment: macOS arm64, Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6, matplotlib 3.10.8, IPython 9.17.1. One configured numerical worker; no GPU, paid calls or media playback.

## Measured chain: moments → worksheet → review decision

An explicitly synthetic example uses 100 independent one-year policies, each paying 100 on survival with probability .75 and zero on death, at zero interest. The real APIs are composed as `A_x` moments → `insurance_variance` → `Life.portfolio_percentile`. GERO supplies the wiring, JSON worksheet, rounding and example review threshold.

| Quantity | Original | Candidate / independent moment expectation |
| --- | ---: | ---: |
| Mean per policy | 75 | 75 |
| Second raw moment | 75 | 7,500 |
| Raw variance from moments | −5,550 | 1,875 |
| Library helper's variance after clamping | 0 | 1,875 |
| Library normal-approximation 95th percentile, 100 policies | 7,500.00 | 8,212.24 |
| Consumer reads worksheet against a synthetic 8,000 threshold | BELOW_THRESHOLD | REVIEW |

The generated worksheet understates this normal-approximation estimate by **712.24** before the candidate. An independent exact-rational binomial CDF gives an exact discrete 95th percentile of **8,200**. The candidate preserves the library's normal approximation; it does not make that approximation exact. Restoring the original line restores the original worksheet and decision.

These are synthetic expected-payout and risk-summary calculations. No real insurer document, premium, regulatory capital requirement, commercial quote, customer loss, deployed use or security impact was established.

## Prior-art review and scope

The bounded review read all seven open/closed upstream issue/PR bodies, their available comments, PR 2's diff, the guide repository's empty issue list, and all nine target-file history diffs. The faulty expression was already in the earliest inspected file history. The current GERO catalog and the prior whole-life, endowment-variance, ConstantForce, Beta and annuity reports were compared. Focused public searches did not find an exact earlier report or correction. This is not an exhaustive proof of worldwide novelty.

The earlier internal I02 probe was not a published confirmed case; searches inside the frozen endowment-variance and annuity-selection archives found no I02/probe inclusion. This report completes that retained candidate. It does not republish issue 6 under another name.

Validation is limited to the specified first and second positive moments, finite positive terms, nonnegative payouts and rates, and supplied survival laws. Whole-life limits, other subclasses, zero-term special behavior, negative payouts, arbitrary time-varying benefits and all other actuarial methods are outside the candidate's certification.

## Reproduction and evidence

Use `reproduce.py` after installing `requirements-repro.txt`. It sets numerical thread counts to one and runs the full four-variant grid. `portable_replay.py` performs the separate patch-and-reverse replay in a new directory. Source bytes, the official wheel, independent oracles, raw results, generated worksheets and receipts are retained. See `SOURCE_LEDGER.json`, `SOURCE_MANIFEST.json`, `GRID_RECEIPT.json` and `PORTABLE_REPLAY_RECEIPT.json`.

Primary sources: [pinned implementation](https://github.com/terence-lim/actuarialmath/blob/7d18f11ad304898f177b7922b3c53f70e4c2b4f4/src/actuarialmath/insurance.py), [official API guide](https://actuarialmath-guide.readthedocs.io/en/latest/insurance.html), [official release](https://pypi.org/project/actuarialmath/1.1.0/).


Frozen evidence: [gero-actuarialmath-direct-endowment-second-moment-2026-09-24.zip](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/reports/actuarialmath-direct-endowment-second-moment/gero-actuarialmath-direct-endowment-second-moment-2026-09-24.zip). SHA256 `ecc98650dc611bccd882731c926e41c8ba57792263677980a7d51845ef6470e5`.
