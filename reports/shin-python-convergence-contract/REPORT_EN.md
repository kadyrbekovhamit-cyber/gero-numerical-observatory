# shin: a default probability result can hide non-convergence

Xamit Kadirbekov and Daniyal Kadirbekov · GERO · 22 October 2026

**Status:** Reproduced Python failure-reporting gap; no reply found in the reviewed channel by 22 October

At `ae460853fabeca7d512bbf7de8aaa55dc17f3485`, the Python API with `force_python_optimiser=True` accepts odds [2.1, 3.6, 4.6]. After 1,000 iterations, it returns [0.4607771237, 0.2518222697, 0.1887618806], summing to 0.9013612740. `full_output=True` reports a delta of 0.1805021215. The default result omits those diagnostics.

This is an **underround** input: inverse odds sum to about 0.97136. We do not claim that the Shin model supplies valid betting probabilities in that domain, or that normalising this failed vector is a correct repair. Rejecting an unsupported domain or returning an explicit convergence failure are both reasonable API choices. The documented `full_output` option is an existing workaround, not a discovery.

A normal overround control [2.7, 2.3, 4.4] converges in 437 iterations and sums to approximately one. A two-outcome example follows a different analytic branch; the report does not generalise the failure to it. We executed the unchanged Python API; a deliberately unusable native-extension placeholder allowed import without invoking Rust. The Rust optimiser and complete package suite were not tested.

The [repository documentation](https://github.com/mberk/shin) describes the model and output options. A private report was sent 5 October with a no-earlier-than-20-October publication date. Fresh source and targeted mailbox review on 22 October found no substantive response or corrective change in the reviewed scope.

## Evidence and limits

The attached archive contains the selected source, reproduction and observed output. AI-assisted investigation and writing; the stated executions were actually performed locally. Numerical jobs ran sequentially with one numerical worker and no GPU. No production account, real funds or actual customer losses were tested. Bounded public-source and issue review does not establish global priority. Private correspondence and contact details are excluded.

Report: CC BY 4.0. Original reproduction: MIT. Bundled upstream source retains its own licence; see the archived notices.
