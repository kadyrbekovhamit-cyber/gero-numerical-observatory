# Independent review: actuarialmath endowment variance

Review finalized: 2026-09-19 09:19:39 UTC / 14:19:39 Asia/Tashkent.
Reviewer: independent delegated mathematical/source review.
Case: /Users/khamit/Documents/Codex/gero-continuous-research/research/actuarialmath-endowment-variance-2026-09-19
Pinned source named by the evidence: 7d18f11ad304898f177b7922b3c53f70e4c2b4f4; official release 1.1.0.

## Decision

**No blocking mathematical or API-contract objection found to the narrow candidate correction or the single vendor issue.** The finite-support oracle is independent of the implementation's insurance recurrences; the elementary one-year anchor is unambiguous. The issue/report appropriately distinguish the normal portfolio approximation from the exact binomial quantile and restrict the decision demonstration to synthetic inputs and GERO's own adapter.

This review is source/document inspection and mathematical reasoning only. I did not execute any numerical code, import the package, build, install, render, or run the supplied tests. Numerical counts and replay success below are conclusions recorded by the root's executions, not an independent second run. Novelty depends on the root's separate bounded current-source/issues/history/GERO review.

## Inspected artifacts

- WORK_STATUS.md; candidate.patch.
- check_endowment.py; run_grid.py; GRID_RECEIPT.json.
- check_chain.py; DOWNSTREAM_RECEIPT.json.
- prepare_variants.py; test_endowment_variance.py; TEST_RECEIPT.json.
- PORTABLE_REPLAY_RECEIPT.json.
- source/baseline/src/actuarialmath/insurance.py: E_x, A_x, insurance_variance, whole_life_insurance, term_insurance, endowment_insurance.
- source/baseline/src/actuarialmath/life.py: portfolio_percentile and neighboring portfolio code.
- REPORT_EN.md and disclosure/ISSUE.md, inspected before issue submission.

The official [Insurance guide](https://actuarialmath-guide.readthedocs.io/en/latest/insurance.html) was also opened during this review. Its Variances section states the second-moment-minus-squared-mean identity and includes the benefit to the moment's power; its endowment section combines term and survival benefits. These support the contract under review. Its subsequent covariance expansion contains separate problematic signs/factors and should not be adopted as an oracle. The present finite-payout oracle does not depend on that expansion.

## Cause and narrow correction

The non-variance endowment path returns death-benefit moments from term_insurance, plus survival probability/discount moments multiplied by the effective endowment raised to the requested moment. For moment 1 and moment 2, these are already the actual payout's first and second moments.

The variance branch recursively requests those moments while retaining b and endowment, then calls insurance_variance(A2, A1, b=b). The helper multiplies the moment difference by b squared. That factor is appropriate when inputs are moments of a unit-benefit random variable, but not after both benefit amounts have been included.

Changing only this call to b=1 leaves the shared helper's unit-moment behavior intact. It implements variance as max(0, E[Z²]−E[Z]²) for the already-scaled moments. It also handles zero death benefit with a positive survival benefit: multiplying by b squared currently discards a nonzero survival variance. The patch does not promise to improve cancellation or the helper's existing clamp.

For one year, with discounted payouts b*v on death and e*v on survival, the exact variance is q*(1−q)*(b−e)²*v². In the binary-exact anchor q=1/4, b=2, e=1, i=0, mean=5/4, second moment=7/4 and variance=3/16. The current extra factor four produces 3/4. No approximation or decimal conversion is needed for this anchor.

## Oracle independence and tolerances

The grid constructs rational death/survival probabilities directly, enumerates each discounted death payout plus the survival payout, and checks total probability one. It obtains the variance from the centered payout sum and checks the equivalent raw-moment identity. It calls no second actuarial implementation to produce expected answers. The independence is adequate for the specified finite discrete contract.

The grid covers 13 mortality-profile/term combinations, four interest rates, five death-benefit values and five endowment arguments. Default endowment maps to the death benefit, matching the implementation. Zero/certain mortality, deterministic payouts, differing benefits, zero death benefit, multiple terms, and monetary rescaling are useful controls.

The oracle uses exact decimal rational inputs, while the library receives floats. For non-binary decimals these are slightly different numerical inputs. This is a limitation, not a source of the large discrepancy: the binary-exact anchor avoids it entirely, and the grid uses explicit absolute allowances. A short disclosure sentence could make the decimal-to-float distinction more explicit.

The preselected absolute variance tolerance is 256*epsilon*max(1,b²,e²). It is conservative relative to the payout scale, which accommodates ordinary floating-point cancellation. Therefore “zero mismatches” means zero outside this declared tolerance, not exact rounding, relative accuracy near zero, or absence of all cancellation errors. The report states the tolerance. It should retain that qualification.

The support-range bound is an independent necessary control, not a complete characterization of variance correctness. Using only positive-probability outcomes for its range is correct. The primary direct variance comparison supplies the stronger oracle. The unit-conversion check correctly scales both death and survival payouts, so true variance scales quadratically; the erroneous extra b² yields quartic scaling in nonzero examples.

## Reported grid and regression evidence

The receipts record 1,300 scenarios and 483 original/release/restored variance mismatches, falling to zero with the candidate; 375 support-bound violations also disappear. First and second moments pass and remain equal across variants. Applying only the previously reported whole-life correction leaves the endowment discrepancies.

The evidence preserves 794 scenarios with all three recorded outputs equal. It also discloses 23 previously passing scenarios whose variance changes within tolerance, with maximum absolute change approximately 4e-15. Those must not be silently grouped as bitwise-unchanged controls.

**Small wording correction advised:** run_grid.py checks Python list/float equality after JSON parsing. It does not compare IEEE-754 byte representations. In REPORT_EN.md and disclosure/ISSUE.md replace “first/second moments remain byte-identical” with “first/second moments remain exactly equal in the recorded outputs,” unless a separate bitwise check is actually added by the root. “All three reported outputs remain identical in 794 scenarios” is supported in the ordinary numeric-output sense. Complete JSON artifacts are not byte-identical because provenance paths and metadata differ; do not claim they are.

The six regression methods are meaningful: non-unit payout, zero death payout, deterministic equal payout, discounted two-period payouts, quadratic rescaling, and preservation of the shared helper's contract. Four failing original methods can produce six failure entries because the rescaling test has three failing subtests. This is correctly separated from the method count. No full upstream-suite coverage is established.

PORTABLE_REPLAY_RECEIPT.json reports successful replay in a fresh package copy: five direct grid variants, four downstream variant results, and targeted test outcomes. I inspected this receipt, but did not independently reproduce it.

## Downstream validity and limits

The chain calls two actual upstream public APIs: endowment_insurance for moments and portfolio_percentile for an aggregate percentile. The latter uses a normal approximation, mean*N + ndtri(prob)*sqrt(variance*N). The independent normal reference uses the standard-library quantile implementation. Agreement with that reference establishes propagation of the corrected input variance through the same approximation; it is not proof that a normal approximation equals the exact discrete quantile.

The separate exact aggregate reference enumerates the binomial death count using rational masses, maps each count to total payout, groups equal payout values, sorts the support by payout, and takes the first cumulative mass at least 0.95. Sorting by payout correctly handles the reverse-benefit case b<e; grouping handles the deterministic case b=e. The construction assumes 100 independent identically distributed policies. That is explicitly synthetic and need not describe an insurer's actual portfolio.

The reported main example, b=100 and e=50, changes the normal-approximation amount from 41,862.125661 to 6,606.121257, compared with the exact binomial 6,600. The remaining approximately 6.12 is approximation error, not residual evidence that the variance patch failed. The synthetic budget 7,000 therefore changes the adapter's decision. In the zero-death-benefit case, the amount changes in the opposite direction, from 7,500 to 8,212.242513, with exact 8,200 and budget 7,900. Two of four selected decisions change.

The JSON round-trip and subsequent threshold decision are implemented in GERO's adapter. They are a legitimate concrete propagation demonstration, but not a claim that the library contains an insurer's document workflow, policy rules, or capital engine. The chosen budgets demonstrate possible consequences under those assumptions; they do not estimate frequency, customer loss, actual premiums/reserves, or regulatory decisions.

A normal approximation can sometimes exceed a distribution's finite support even with correct moments. Thus the original amount exceeding the maximum possible payout is useful context for this selected example, but must not be used alone to allege a separate bug in portfolio_percentile. The exact variance mismatch and corrected-versus-original normal calculation establish this case.

## Distinction from the previous whole-life case

The previously reported whole-life problem squares its first moment before passing it to a helper that squares it again. Its recursive calls use unit benefits, making the helper's b² factor appropriate in that path.

The present endowment path instead includes actual benefits in both recursive moments and applies b² again. It calls term/whole-life only for positive moments 1 or 2, not the defective whole-life VARIANCE branch. Consequently this is a separate implementation cause in a separate caller, though it belongs to the same variance family. prepare_variants.py changes the earlier whole-life line only for a negative attribution control; the recorded grid remains discrepant. That control is relevant and avoids treating the same fix as a new finding.

No claim is made here that the issue is newly introduced or unprecedented. The report dates the inspected pattern to historical source and qualifies its duplicate search. The root owns the fresh novelty review and any submission.

## Final scope and next action

No observed mathematical/contract blocker prevents one narrowly scoped developer report after the root's duplicate check. Apply the minor equality wording correction above. Preserve the declared tolerance, normal-versus-exact distinction, 23 rounding-only changes, and synthetic decision attribution in all derived publication text.

The established scope is discrete finite-support LifeTable endowment examples, the inspected current/release source, and the selected aggregate examples. Continuous-time paths, unbounded/extreme inputs, subclasses with different overrides, all package features, full upstream regressions, performance, and real insurer exposure remain outside this review. No external message or publication was made by this reviewer.
