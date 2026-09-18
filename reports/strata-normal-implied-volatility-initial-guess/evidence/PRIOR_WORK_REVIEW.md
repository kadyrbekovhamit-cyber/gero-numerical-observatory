# B01 final duplicate and vendor-route review

Checked: 2026-09-18 08:42:17 UTC. Read-only GitHub API and official documentation review. Numerical confirmation belongs to the parent's Strata2.12.74 run; this agent ran no tests and sent no messages.

**No exact zero-initial-volatility duplicate was found in this bounded review. The previous negative-forward defect and its correction must be cited.**

## Current source and release

Fresh main is `987932ee95bf53e2baaff9a6b8e738a00f558b10`. GitHub returns the same NormalFormulaRepository Git blob `9ff172af2917669068c395524a01117908648639` (16299bytes) for main and v2.12.74. This proves file identity at the API level; it is not a jar checksum.

The file-history endpoint returned10commits, with the latest modification being merge `a88ef3b68f12724b8b99b886cae9590a8702ba47` on1November2021. No later change to this file was shown. Full current NormalFormulaRepositoryImpliedVolatilityTest.java was read: successful inversion checks use nonzero starting volatilities; no positive-price zero-start test was found in that file.

## Prior work and novelty boundary

[Issue2238](https://github.com/OpenGamma/Strata/issues/2238) concerns negative forwards making initialization and step sizes negative. Its sole comment points to [PR2372](https://github.com/OpenGamma/Strata/pull/2372), which changed initialization and merged in2021.

The [review discussion](https://github.com/OpenGamma/Strata/pull/2372#discussion_r740326917) proposed Math.min while reviewing a first-draft lower-floor conditional. The final merge has the initializer now under examination. Neither the inspected report nor review describes the positive-forward, positive-price, zero-start failure. The report should therefore identify this as a specific remaining issue associated with the earlier correction, without claiming the whole initialization problem is newly discovered.

Eight fresh API searches covered all issue/PR states and titles, bodies and comments:

| Query terms within OpenGamma/Strata | Hits |
|---|---:|
| initialNormalVol | 0 |
| zero initial | 0 |
| impliedVolatility | 2:2238,2358 |
| Math.min + volatility | 1:2372 |
| NormalFormulaRepository | 2:2238,2372 |
| normal + zero + volatility | 2:1422,465 |
| initial guess | 1:710 |
| starting point + volatility | 1:2372 |

Every search returned `incomplete_results=false`; all results fit one100-item page. Related PR bodies and relevant review-comment matches for2358,1422,465,710 were examined: they concern pricer exposure, caplet stripping, swaption settlement and Black/normal conversion. None of the inspected material describes this exact zero-start inversion outcome. The complete query strings and returned metadata are in the companion JSON.

This is a bounded public search, not proof that no unindexed, private or deleted report exists. Official forum retrieval still returns502; two exact indexed forum queries produced no results. The GERO catalog exclusion remains the parent's earlier local check.

## Exact vendor route and restriction status

The current [repository issue template](https://github.com/OpenGamma/Strata/blob/main/.github/issue_template.md) routes general user questions to the forum. It says GitHub issues are primarily for the development team, **but explicitly permits a confirmed bug with details and preferably a failing test**. [Contribution instructions](https://strata.opengamma.io/contributions/) also direct bugs to GitHub Issues, requesting version, JDK/OS and reproduction.

Primary route: https://github.com/OpenGamma/Strata/issues.

If authenticated issue creation is actually rejected, the documented support route is https://forums.opengamma.com/ via [the support page](https://strata.opengamma.io/support/). A PR with a tested regression/fix is also a documented contribution route, subject to actual account permissions. Do not use the security mailbox for this numerical case without a separate demonstrated security impact.

The public issue page previously displayed a restriction banner. **Actual reporter restriction settings are not verified.** The read-only interaction-limits API returned403 because it requires repository admin privileges; that response does not show that ordinary issue creation is prohibited. Repository metadata has issues enabled and archived=false. Parent should distinguish a real authenticated submission rejection from the public-page wording.

## Recommended report framing after the parent's grid

Title: **Normal implied volatility returns zero for a positive ATM price when initialNormalVol is zero**.

Include the real2.12.74 output and pinned main/file identity; exact seven arguments; independent ATM identity `sigma = price * sqrt(2*pi) / (numeraire * sqrt(T))`; price residual; root's patched/restored controls; and related2238/2372. State that this report concerns zero initial volatility at a positive forward, whereas2238 addressed negative forwards. Do not claim newly affected customers, live products or losses.

Current status: vendor report can proceed after the parent's regression grid and final evidence review. No message, issue, PR or publication was created by this agent.
