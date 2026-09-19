# Strata downstream review: Black priceAdjoint2 at zero volatility

Reviewed 19 September 2026, Asia/Tashkent (18 September UTC). Read-only child review; the root agent owns all numerical execution. Current source pin: `987932ee95bf53e2baaff9a6b8e738a00f558b10`.

The observed effect is a real library integration failure through the documented custom-provider overload of `SabrExtrapolationRightFunction`. The fixture that supplies constant zero volatility is synthetic. The evidence does not establish a failure in the default Hagan/CMS path or any deployed bank.

## Native call path and contract

- [Custom factory and constructor](https://github.com/OpenGamma/Strata/blob/987932ee95bf53e2baaff9a6b8e738a00f558b10/modules/pricer/src/main/java/com/opengamma/strata/pricer/impl/option/SabrExtrapolationRightFunction.java#L130): the six-argument overload accepts `VolatilityFunctionProvider<SabrFormulaData>`. Argument order is `of(forward, sabrData, cutOffStrike, timeToExpiry, mu, provider)`. The ordinary five-argument overload has a different order. Neither the inspected factory, constructor nor provider contract requires strictly positive volatility.
- [Provider contract](https://github.com/OpenGamma/Strata/blob/987932ee95bf53e2baaff9a6b8e738a00f558b10/modules/pricer/src/main/java/com/opengamma/strata/pricer/impl/volatility/smile/VolatilityFunctionProvider.java#L19): extend this abstract class. Implement scalar volatility and second adjoint; override the concrete first adjoint if an exact constant fixture is wanted. Six first derivatives and the 2-by-2 forward/strike Hessian are zero for a constant function.
- `SabrExtrapolationRightFunction` and `SabrHaganVolatilityFunctionProvider` are final. The former has a private constructor, so use its public factory. The latter cannot be subclassed.
- [Fitting chain](https://github.com/OpenGamma/Strata/blob/987932ee95bf53e2baaff9a6b8e738a00f558b10/modules/pricer/src/main/java/com/opengamma/strata/pricer/impl/option/SabrExtrapolationRightFunction.java#L373): provider second adjoint feeds `BlackFormulaRepository.priceAdjoint2`; line 380 combines its Hessian into call-price strike curvature. Zero volatility with an ordinary non-ATM cutoff makes the old Hessian NaN. The all-small-price guard at lines 381-384 consequently fails and root bracketing receives NaN.
- [BracketRoot](https://github.com/OpenGamma/Strata/blob/987932ee95bf53e2baaff9a6b8e738a00f558b10/modules/math/src/main/java/com/opengamma/strata/math/impl/rootfinding/BracketRoot.java#L36) throws `MathException` when a bracket endpoint function value is NaN.
- Choose expiry greater than `SMALL_EXPIRY=1e-6` and cutoff greater than forward. Shorter expiry bypasses fitting; an ITM cutoff enters different fitting/sensitivity branches.

## Default Hagan/CMS exclusion

[CMS pricer lines 707-714](https://github.com/OpenGamma/Strata/blob/987932ee95bf53e2baaff9a6b8e738a00f558b10/modules/pricer/src/main/java/com/opengamma/strata/pricer/cms/SabrExtrapolationReplicationCmsPeriodPricer.java#L707) constructs the ordinary default-provider extrapolator. It is a real native caller, but this does not by itself prove the exact-zero Black case is reachable.

[Hagan second adjoint](https://github.com/OpenGamma/Strata/blob/987932ee95bf53e2baaff9a6b8e738a00f558b10/modules/pricer/src/main/java/com/opengamma/strata/pricer/impl/volatility/smile/SabrHaganVolatilityFunctionProvider.java#L381) returns `Math.max(MIN_VOL, rawSigma)`, with `MIN_VOL=1e-6`. Therefore its output is at least 1e-6, or NaN; it cannot be exact zero. Alpha zero is accepted by `SabrFormulaData`, and the scalar and first-adjoint methods have explicit alpha-zero handling. The second-adjoint method instead uses `nu/alpha` and can become NaN before Black. The narrow sigma-equals-zero patch does not fix this separate upstream failure.

The root's default-alpha-zero row correctly serves as a negative attribution control. It must not be counted as an example of this Black fix helping default CMS valuation.

## Executed downstream receipt independently read

Read locally, without re-running:
- `/Users/khamit/Documents/Codex/gero-continuous-research/research/strata-black-adjoint2-2026-09-18/downstream/DOWNSTREAM_RECEIPT.json`
- `/Users/khamit/Documents/Codex/gero-continuous-research/research/strata-black-adjoint2-2026-09-18/downstream/ExtrapolationProbe.java`

The Java fixture uses the public provider overload, returns its supplied constant volatility and fills all first and second derivatives with zero. This is consistent with its defined function. It does not claim to implement the Hagan formula.

For forwards 0.03, 0.05, 1; cutoff 2 times forward; expiry 1 or 5; mu 4:
- All six exact-zero-provider rows fail during constructor root bracketing on the released, current and restored implementations.
- All six construct successfully with the Black-only candidate correction.
- Fifteen non-target rows remain unchanged, including six tiny-positive-volatility failures and the default-alpha-zero failure.
- Positive constant volatility and default alpha 0.05/0.2 controls remain successful.
- Example at forward 0.03, cutoff 0.06, expiry 1: candidate parameters are [-100, 0, 0]. Call at strike 0.015 is 0.015; put is 0. Call at strike 0.09 is 5.6699831977150385E-40, put is 0.06; the forward derivative above cutoff is zero.

This is a measured change from inability to construct a pricing object to finite native pricing/sensitivity outputs for the synthetic constant-zero model. The native fallback is approximate: its call tail is `exp(-100) * strike^(-mu)`, so the tiny positive tail is not exactly zero. The probe does not execute `priceAdjointSabr`; do not claim that method was measured. Other zero-volatility fitting branches contain finite-difference denominators and remain outside this result.

## Bounded duplicate refresh and source history

Fresh official GitHub search found:
- Exact `priceAdjoint2` in issue/PR titles, bodies and comments: 0 indexed hits.
- `SabrExtrapolationRightFunction` in issue/PR text: 0.
- `volatilityAdjoint2` plus `alpha`: 0.
- `SABR` plus `zero`: PRs 939 and 1422.
- Additional `SABR alpha zero`: PR 939.

All saved issue-search responses report `incomplete_results=false`. Searches do not filter out closed items. This is a bounded indexed search, not proof that no private, unindexed or differently worded report exists.

[PR 939](https://github.com/OpenGamma/Strata/pull/939/files) was inspected at the relevant actual diffs. It changed the nu finite-difference step from relative to absolute because nu can be zero, and added the Hagan second-adjoint volatility floor. It does not fix the Black zero-volatility Hessian. The floor explains why the ordinary default provider does not pass an exact zero into Black. PR 1422's 24-file list and matching-diff scan had no target Black/extrapolator/Hagan-provider changes. Earlier Black origin history and contribution-channel review remain in the case's existing INDEPENDENT_REVIEW.md; do not reopen or repeat the published Normal IV issue 2796.

GitHub code search returned BlackFormulaRepository and SabrExtrapolationRightFunction for `priceAdjoint2`, and the extrapolator, CMS pricer and two tests for the extrapolator name. Code search may omit large files (including the known large Black test class); it is not exhaustive repository call-graph proof.

Raw query results, relevant PR diffs, current head receipt, seven pinned source excerpts and the read-only execution-receipt summary are preserved in SOURCES.json beside this report.

## Handoff

No external report, post, issue or publication was sent by this reviewer. No local numerical, compilation, installation or render job was run. Root can use the constructor-failure evidence in a vendor-first report with the custom-provider limitation explicit. Do not label it a default CMS, customer loss or production exposure result.
