# penaltyblog: positive expected log growth at a ruin boundary

Xamit Kadirbekov and Daniyal Kadirbekov · GERO · 22 October 2026

**Status:** Reproduced in current modules; extra report unanswered in reviewed channel; no validated repair

At `6870f0e762e9cba35fa722dc14a8daeaf8bc6134`, `kelly_criterion(2.0, 0.8, fraction=2.0)` caps the proposed fraction at 1. The losing outcome has probability 0.2 and leaves zero wealth. Consequently expected log wealth is negative infinity. The result instead exposes `expected_growth` about +0.5545177444, while its own risk diagnostics report negative-infinite Kelly growth and 0.2 ruin probability. The output fields disagree about the same boundary. Returning negative infinity, rejecting the stake, or applying a documented strictly-below-one cap would make the contract coherent; no repair was validated here.

A separate diagnostic concerns `arbitrage_hedge`. With existing stakes [50,0,0], old odds [3.5,3.4,2.9] and new odds [2.8,3.8,3.1], its LP reports unbounded and exposes `lp_success=False`. A fallback proposes stakes with a negative guarantee near -50.90. Exact arithmetic verifies that the feasible alternative [0,40,50] produces net payoffs [35,12,15]. This establishes a limitation of that fallback; it does not show that its reported negative payoff was summed incorrectly. An explicit budget constraint or failure without an advisory stake vector would avoid conflating an unbounded model with the absence of a profitable feasible allocation.

We imported the real unchanged betting modules, bypassing unrelated package initialisation, and used SciPy with one numerical thread. No live betting, account or recommendation was tested. A separate cap in the multi-bet API may be a deliberate risk policy and is not counted as a mathematical defect.

The extra private report was sent 5 October; no response was found in the targeted 22 October review. **Issue 50 is a different, earlier partial-hedge case:** its maintainer reports a fix in v1.13.1. We do not describe that earlier case as unanswered or assert that the new examples invalidate its fix.

## Evidence and limits

The attached archive contains the selected source, reproduction and observed output. AI-assisted investigation and writing; the stated executions were actually performed locally. Numerical jobs ran sequentially with one numerical worker and no GPU. No production account, real funds or actual customer losses were tested. Bounded public-source and issue review does not establish global priority. Private correspondence and contact details are excluded.

Report: CC BY 4.0. Original reproduction: MIT. Bundled upstream source retains its own licence; see the archived notices.
