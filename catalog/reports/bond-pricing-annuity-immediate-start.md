# bond_pricing 0.7.3 ignores immediate-start annuity payments

Xamit Kadirbekov · GERO Research · 23 September 2026

The released `bond_pricing.present_value.annuity_instalment` accepts `immediate_start=True` but calculates the same payment as for an ordinary, end-of-period annuity. At ordinary inputs, this changes a payment of approximately **110 into 121**. The result does not satisfy the requested present value when checked with the package's existing `annuity_pv` API.

This is one reproducible implementation defect on synthetic cash flows. No deployed bank system, customer loss or bounty entitlement is claimed. The correction below is proposed and has not been accepted upstream.

## Minimal example

```python
from bond_pricing import annuity_instalment, annuity_pv

p = annuity_instalment(rate=0.1, n_periods=2, pv=210,
                      immediate_start=True)
print(p)  # 120.99999999999994; expected approximately 110

value = annuity_pv(rate=0.1, n_periods=2, instalment=p,
                   immediate_start=True)
print(value)  # 231.00000000000003; requested PV was 210
```

For two payments at times 0 and 1 and an effective per-period rate of 10%, the cash-flow equation is `payment + payment / 1.1 = 210`. Its solution is 110. A payment of 121 is appropriate at times 1 and 2, not at the requested times 0 and 1. The 10% difference here is a property of this example, not an estimate of real-world financial losses.

## Source and cause

The [documented API](https://bond-pricing.readthedocs.io/en/latest/#bond_pricing.present_value.annuity_instalment) defines the flag as selecting whether cash flows start immediately. In the [pinned implementation](https://github.com/jrvarma/bond_pricing/blob/8e04fd03c2594422e457e2f08b460017dc6b4993/bond_pricing/present_value.py), `annuity_instalment` accepts the flag but does not use it in the calculation.

The [PyPI 0.7.3 wheel](https://pypi.org/project/bond-pricing/0.7.3/) was verified against PyPI's SHA-256. Its `present_value.py` is byte-identical to master commit `8e04fd03c2594422e457e2f08b460017dc6b4993`, fetched on 23 September 2026. This is a dated source comparison; later upstream changes require a fresh check.

## Correction and executed checks

`fix.patch` incorporates the timing flag into the required annuity present value while preserving the existing ordinary-annuity branch. For terminal payments, it follows the timing convention already used by this package's PV/FV functions. It does not address unrelated small-rate cancellation or array/default-value behavior.

The accompanying executable replay imports the real wheel, constructs the local correction separately, and compares both against a direct 85-digit Decimal sum of the discounted cash flows. Rates are interpreted as their actual binary64 inputs; the oracle does not call the package's annuity-factor formula.

| Check | Released implementation | Proposed correction |
|---|---:|---:|
| Payment mismatches, 720 scenarios | 288 | 0 |
| Present-value mismatches of returned streams, same scenarios | 288 | 0 |
| Six additional regression checks | Not a separate before/after corpus | 6 passed |

All 360 end-of-period scenarios remained bit-for-bit unchanged. The additional checks cover default-FV roundtrips, timing-vector broadcasting and a positive-rate perpetuity. Rates were −5%, 0%, 1%, 10% and 50%; frequency pairs were (1,1), (12,12) and (2,12); the finite corpus used 1, 2, 12 or 60 payments and terminal payments of 0 or 25.

A second replay through GERO Audit Lab's ordinary queue used the same 256 inputs for the release and correction (218 unique inputs): **105 payment mismatches before, zero after**. These inputs partly overlap the first corpus. Neither repeated scenarios nor two observables should be counted as distinct defects.

Checks ran sequentially on CPU with numerical threads limited to one, without a GPU. The source-level correction and reference were prepared in the same AI-assisted research workflow; this is not an independent third-party audit or proof of correctness outside the tested domain.

## Prior-report and bounty review

Before disclosure, all seven public repository issues, 15 issue comments, the empty pull-request list, all three commits returned for this file, release/advisory listings and focused public searches were reviewed. Issues #6 and #7 concern frequency conventions and do not describe this ignored flag. The current repository tree contained no security, disclosure, contribution or bounty policy file. No exact prior report or fix was found in the reviewed material.

This establishes a documented public-source search, not worldwide novelty: private reports and unindexed discussions remain unknowable. No applicable publicly advertised bounty program was found. Similar search results for **Bond Protocol** concern a different project and were excluded. Full review scope is recorded in `NOVELTY_AND_BOUNTY.json`.

## Reproducibility and licensing

The evidence package contains the pinned wheel, upstream source/license, proposed patch, recorded results, dependency pins and `reproduce.py`. The replay uses an existing Python environment with the listed dependencies; it does not silently install packages or contact external services. It writes its results and patched copy inside its own directory. Python 3.12.13, NumPy 2.3.5, pandas 2.3.3 and SciPy 1.16.3 were used for the original run.

The upstream package declares GPLv3. The derived patch and the accompanying replay code are supplied under GPLv3. Permission for later GPL versions was not established. The separate proprietary Audit Lab product is not included in this publication. Text and figures by GERO Research are CC BY 4.0; upstream material retains its own license.

Publication status, maintainer receipt and verified distribution links are recorded separately. A submitted report is not an acknowledgment or an accepted fix.

## Verified publication links

- [github](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/bond-pricing-annuity-immediate-start.md)
- [zenodo](https://zenodo.org/records/22919559)
- [gero](https://www.gero.uz/research/articles/bond-pricing-annuity-immediate-start.html)
- [huggingface](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/bond-pricing-annuity-immediate-start.md)
- [linkedin](https://www.linkedin.com/feed/update/urn:li:share:7508531622317719552/)
- [youtube](https://www.youtube.com/shorts/E-sES3VgLOA)
- [instagram](https://www.instagram.com/gero.math.tech/reel/Ddok5Onzp-J/)

Maintainer issue: https://github.com/jrvarma/bond_pricing/issues/8 . Proposed correction only; no upstream acceptance or reward is claimed.
