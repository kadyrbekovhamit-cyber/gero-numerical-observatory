# One missing digit in NeqSim's ISO 6976:2016 table

NUMERICAL AUDIT · 8 NOVEMBER 2026

NeqSim's ISO 6976:2016 constants give n-decane a molar mass of 42.28168 instead of 142.28168 g/mol, so with 0.1 mol% n-decane the relative density comes out about 0.55% low and the Wobbe index about 0.28% high.

Xamit Kadirbekov, Daniyal Kadirbekov · AI-assisted investigation and writing

**Checked 8 November 2026: PR #4920 is open and not merged. It awaits maintainer review; no review has been posted yet. No upstream acceptance or released fix is claimed.**

[Upstream pull request #4920](https://github.com/equinor/neqsim/pull/4920) · [GitHub catalog](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/neqsim-iso6976-2016-decane-molar-mass.md)

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO Numerical Assurance). Published 8 November 2026.

## Published proposal

- [equinor/neqsim PR #4920](https://github.com/equinor/neqsim/pull/4920) — head commit `5b0e37fa2a383b1188be01e1dcf6ab5ced1acfd4` on master `2f6ea70`; open at submission on 8 November 2026 (Tashkent time).

## Context

We cross-checked open gas-measurement code — AGA8 / GERG-2008 equations of state, ISO 5167 flow metering and ISO 6976 calorific values — against the NIST reference code and the standards themselves. Most of it agreed with the references to 13–14 significant digits. This is one of the few places where it did not.

## What the table is used for

ISO 6976 gives the rules for computing a natural gas's calorific value, density, relative density and Wobbe index from its composition. NeqSim, Equinor's open-source thermodynamics library, implements the 2016 edition in `Standard_ISO6976_2016`, which reads the per-component constants from `src/main/resources/data/ISO6976constants2016.csv`.

## The discrepancy

The n-decane row of that file (component `nC10`) has `MolarMass` 42.28168 g/mol. The correct value is **142.28168 g/mol** (ISO 6976:2016, Table A.2; 10 × 12.0107 + 22 × 1.00794). NeqSim's own 1995-edition table already has 142.285. The first digit was lost.

```diff
- 19,"nC10",42.281680000,0.434000000,0.584000000,...
+ 19,"nC10",142.281680000,0.434000000,0.584000000,...
```

## Effect

The molar mass enters the mixture molar mass, and through it the relative density, the density, the Wobbe index and every value given per unit mass. Volumetric calorific values are not affected.

| n-decane in the gas | Relative density | Wobbe index |
|---|---|---|
| 0.1 mol% | about −0.55% | about +0.28% |
| 0.01 mol% | about −0.056% | about +0.028% |

Regression example: methane 0.999, n-decane 0.001. NeqSim returns a mixture molar mass of 16.068699 g/mol; the expected value is 0.999 × 16.04246 + 0.001 × 142.28168 = 16.168699 g/mol.

## Who is affected

The impact is narrow. Only `Standard_ISO6976_2016` reads this table; the main NeqSim calculation paths use the 1995 table, which is correct. Only gases that contain n-decane are affected, and the error grows with the n-decane fraction.

## Fix and test

- Change the n-decane molar mass in `ISO6976constants2016.csv` from 42.28168 to 142.28168.

- New test `testMolarMassWithDecane` in `Standard_ISO6976_2016_Test.java`: the methane / n-decane mixture above must give 16.168699 g/mol (tolerance 1e-6). Before the fix it gives 16.068699.

## Upstream status

As of 8 November 2026, PR #4920 is open and requires review; no review has been posted. Nothing has been merged or released.

## Limits

The full NeqSim test suite was not run locally; the effect was checked with a standalone class against the released NeqSim 3.28.0 jar and with an independent implementation of the ISO 6976:2016 formulas. The percentages above are for the example compositions; other gases give other values. We make no claim about any operator's measured volumes or invoices.

## Attribution and license

AI tools (Claude) assisted with the analysis, the reference calculations, the patch and test code, and the text of this report; the results were re-run before publication. Xamit Kadirbekov and Daniyal Kadirbekov are the report authors; upstream software remains credited to its developers. Report: CC BY 4.0. Upstream patches retain their upstream licenses.

GERO page: [https://www.gero.uz/research/articles/neqsim-iso6976-2016-decane-molar-mass.html](https://www.gero.uz/research/articles/neqsim-iso6976-2016-decane-molar-mass.html)

_Report CC BY 4.0. Upstream code retains its own licence. Synthetic compositions; no measured custody-transfer or customer-loss claim._
