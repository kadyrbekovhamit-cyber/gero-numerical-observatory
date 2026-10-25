# Tax-Calculator: Schedule D and AMT rule alignment

**Authors:** Xamit Kadirbekov and Daniyal Kadirbekov (GERO). 25 October 2026.

**Status:** Open upstream PRs; not merged or released. Scoped checks and limitations documented.

Five synthetic regressions and 490 passing checks support four proposed calculation corrections.

## Source and proposed changes

Base: `f6ddd4ca74d6f2bf5a8e785b975f8681f030be9b` in PSLmodels/Tax-Calculator.

- [taxcalc PR #3196](https://github.com/PSLmodels/Tax-Calculator/pull/3196) — proposed commit `5ea15d1ceb8a944d5933ee3813d13300e76ae224`.

## Technical evidence

This aligns four current-law calculation details with the Schedule D Tax Worksheet, Form 6251 instructions and IRC 55(d). Five new single-household regressions fail on master f6ddd4ca74d6 and pass with this patch.

- **Schedule D:** include worksheet lines 19–21 when stacking 25%/28%-rate gains, so gains that fit below the top of the 24% ordinary bracket use the ordinary rate. In the 2025 single-filer fixture (taxable income 140,000; ordinary income 60,000; special-rate gain 30,000), regular tax should be 22,214. Before the change it is 23,114 or 24,014 depending on gain type. The 2024 MFJ fixture changes 35,682 to 34,482. [IRS worksheet](https://www.irs.gov/instructions/i1040sd).
- **MFS AMTI:** restore the Form 6251 line-4 addition instead of only zeroing the exemption. In the 2025 fixture with pre-addition AMTI 1,200,000, the added amount is 68,500, so AMTI is 1,268,500 and tentative minimum tax is 352,789. This differs from the recently rewritten AMT function's test expectation, which is updated with the full calculation. [IRS line-4 instructions](https://www.irs.gov/instructions/i6251).
- **2026 threshold:** correct AMT_em_pe from 639,200 to 640,200 in current law, the OBBBA reform and its update source. [Rev. Proc. 2025-32, section 3.10](https://www.irs.gov/pub/irs-drop/rp-25-32.pdf).
- **Child AMT cap:** set the age parameter to zero from 2018 to implement IRC 55(d)(4)(A)(iii), retaining the calculation for historical/counterfactual reforms. In the 2025 dependent fixture (age 17, wages 10,000, interest 30,000), AMT changes from 2,009.50 to zero. This is distinct from #3180's joint-return eligibility fix. **Source discrepancy to review:** Rev. Proc. 2025-32 section 3.11 still describes a child cap; that appears inconsistent with section 55(d)(4)(A)(iii) and the 2025 Form 6251 exemption worksheet. The proposal follows the statute and form, not that inflation paragraph. Please review this interpretation before merging. [Current IRC 55(d)](https://www.law.cornell.edu/uscode/text/26/55).

Validation: 490 passed across test_calcfunctions.py, test_policy.py, test_parameters.py and the five new regressions. Separate Fraction-based worksheet calculations and hand-calculated MFS witnesses match the patched implementation; they differ on the original. Existing Numba/JIT environment was reused, single numerical thread. Full population/CPS/PUF/TMD aggregate-output tests, full lint and documentation regeneration have not been run. Aggregate baselines will need deliberate review; they were not blindly replaced. The separate 2026 charitable-deduction question is excluded.

## Reproduction and boundaries

The archive contains exact public commit patches, upstream licenses, and selected validation metadata. Check out the pinned base, apply each proposal independently, install the project’s declared test dependencies, and run the affected tests described above. The metadata records the actual narrower local environment; a full clean build or unlisted tests are not implied. Existing local paths in logs are descriptive, not required paths.

Novelty checking covered related open/closed GitHub issues and PRs and relevant source history on 25 October 2026. Related reports are acknowledged above; an exhaustive search of every community channel or first-discovery priority is not claimed. No real customer records were used, no deployed product was tested, and no monetary or clinical loss was measured.

## Attribution and licensing

Claude assisted the initial investigation and patch preparation. OpenAI Codex independently repeated the scoped before/after checks, checked the stated reference calculations, and helped prepare the proposed changes and this report. Research/report authors are Xamit Kadirbekov and Daniyal Kadirbekov. Independent human code or legal review is not claimed. Upstream projects retain credit for their software. Report: CC BY 4.0; original reproduction code: MIT; patches retain the upstream licenses.
