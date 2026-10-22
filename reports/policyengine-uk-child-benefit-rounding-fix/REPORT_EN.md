# PolicyEngine UK: Child Benefit charge rounding now corrected upstream

Xamit Kadirbekov and Daniyal Kadirbekov · GERO · 22 October 2026

**Status:** Upstream PR 2323 merged 12 October; helper checks pass; full simulation not rerun

The earlier implementation treated the High Income Child Benefit Charge as a smooth proportional taper. [Section 681C](https://www.legislation.gov.uk/ukpga/2003/1/section/681C) instead specifies whole-percentage and whole-pound rounding. In the 2024+ band, an adjusted net income of £60,199 yields no completed £200 step; £60,399 yields one percentage point. For annual benefit £1,354.60, the corresponding charges are £0 and £13.

MaxGhenis opened issue 2321 and PR 2323, which merged on 12 October at `0820203e381fcf8acb7a18ab0eb691e9d39029f6`. The public explanation attributes the work to a PolicyBench UK internal audit. Our earlier private report was sent 5 October, but timing does **not** establish that it caused this fix. We credit the upstream implementation to its actual author.

On 22 October, current head `9d81a5dfa8bd679f2ade4acbcd215a69f03b3cc4` contains the correction. We imported its unchanged helper module and checked 33 income/benefit combinations, including just-below/exact staircase boundaries, against a separate Decimal construction. All 33 agree. We also inspected the tax formula's calls to these helpers. This is not a full PolicyEngine Simulation run or independent repetition of the maintainer's whole test suite.

The fix distinguishes statutory tax rounding from the model's continuous behavioural take-up approximation. The example therefore concerns the tax-charge variable, not a blanket claim that every continuous model component is wrong. This case is now a **fixed-source retrospective**, not an unanswered current defect.

## Evidence and limits

The attached archive contains the selected source, reproduction and observed output. AI-assisted investigation and writing; the stated executions were actually performed locally. Numerical jobs ran sequentially with one numerical worker and no GPU. No production account, real funds or actual customer losses were tested. Bounded public-source and issue review does not establish global priority. Private correspondence and contact details are excluded.

Report: CC BY 4.0. Original reproduction: MIT. Bundled upstream source retains its own licence; see the archived notices.
