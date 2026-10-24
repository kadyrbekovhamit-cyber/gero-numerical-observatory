# Two reported tax-model discrepancies corrected in PolicyEngine-US 2.60.3

**Authors: Xamit Kadirbekov and Daniyal Kadirbekov / GERO.** Published 24 October 2026.

**Status: upstream fix merged and present in the published package source.** PolicyEngine's Max Ghenis implemented both corrections in [PR #10393](https://github.com/PolicyEngine/policyengine-us/pull/10393), merged on 23 October 2026. The public PR explicitly connects the work to the outside report sent on 5 October. Our contribution was reporting and reproducing the discrepancies; the merged implementation and its expanded tests belong to PolicyEngine.

## What we reported

Our saved experiment used PolicyEngine-US **2.24.1**, synthetic households, tax year 2025 and explicitly supplied adjusted gross income. It calculated deduction variables, not complete tax liabilities.

1. **Married filing separately.** For a married couple in separate tax units, the model allowed a $25,000 tips deduction and a $12,500 overtime deduction for one spouse with supplied AGI of $80,000. Both should be zero under the joint-filing requirement. The same positive values for eligible single and joint filers served as controls.
2. **Phase-out boundaries.** For an eligible single filer at MAGI $150,999, the model reduced the deductions continuously, returning approximately $24,900.10 and $12,400.10. The form instead retains $25,000 and $12,500 until the next complete $1,000 step. At $151,001 the reductions should remain $100 each, rather than $100.10 each.

The reference is the [IRS 2025 Schedule 1-A](https://www.irs.gov/pub/irs-prior/f1040s1a--2025.pdf): Parts II/III require joint filing for married claimants; lines 11/19 round the number of $1,000 increments down before the $100 reduction is applied. These are errors in eligibility and in translating a discrete rule into code, rather than merely decimal rounding noise.

## What changed upstream

The merged code excludes the SEPARATE filing status from both deductions. It also uses a floor operation to count complete phase-out increments. PR #10393 includes regression and property tests and documents its test results and remaining limitations. Those test results are upstream evidence, not tests independently rerun by GERO.

On 24 October we checked the public PR's merged status, inspected its two formula changes, and retrieved the **2.60.3** wheel from [PyPI](https://pypi.org/project/policyengine-us/2.60.3/). The wheel was uploaded on 23 October at 07:07 UTC. Its SHA-256 matched PyPI's published digest; inspection confirmed the eligibility mask and floor operation in both shipped formulas. The evidence manifest records the package and extracted-file hashes.

**We did not install or execute release 2.60.3 in this publication pass.** Source inspection establishes that the corrections shipped; it is not an independent end-to-end regression test or validation of the entire tax model.

## Attribution and limits

The separate-filer requirement had already appeared in [issue #6165](https://github.com/PolicyEngine/policyengine-us/issues/6165) during discussion of earlier legislative drafts. We credit that predecessor and make no first-discovery claim. Our report supplied a reproducible current-model example against the final IRS form and also raised the phase-out-step discrepancy.

The examples measure deduction amounts. We do not claim a measured $6,000 tax difference, actual taxpayer losses, or independently verified national fiscal impact. This is an open-source modeling result, not an error discovered in the IRS's own production system. No private correspondence, taxpayer data or contact details are included in the evidence archive.

The archive contains the historical reproducer and sanitized saved output, a release-source inspection manifest and public reference links. The historical script deliberately asserts the old behavior and is labelled accordingly; it is not a passing test for the corrected release.

AI assistance was used for source review, investigation and writing. The report is CC BY 4.0; the original reproduction script is MIT licensed. Thank you to Max Ghenis and the PolicyEngine team for investigating and correcting both reported points.
