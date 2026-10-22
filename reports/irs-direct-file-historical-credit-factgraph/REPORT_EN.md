# Historical IRS Direct File source: a joint-filer credit and a rounding witness

Xamit Kadirbekov and Daniyal Kadirbekov · GERO · 22 October 2026

**Status:** Historical public-source replay; report unanswered in reviewed channel; production applicability unestablished

## What was reproduced

The public 2024 fact graph at commit `e365f9f43446010af36cccf7edecc52953c00fe9` excludes a spouse's otherwise countable retirement distribution when that spouse is personally ineligible for Saver's Credit. For a joint return this can overstate the eligible spouse's credit. A synthetic, unstubbed fact-graph scenario returns **$750 instead of $150**. A one-flag control changes calculated total tax by **$600**.

This is a result in the released source and its checked-in Scala.js engine. It is **not evidence about a taxpayer's filed return**, the historical production deployment, IRS acceptance, or actual losses. No browser filing flow, JVM backend, PDF/MeF export or e-filing transmission was tested. The public README explains that some production functionality was removed or rewritten before release.

## Saver's Credit witness

All people and return data are synthetic. The principal witness is `repro/witness_savers_normal.json`, adapted from the earlier upstream-fixture-based witness to use taxpayers over age 59½ and fully taxable normal distributions (1099-R code 7).

| Input/result | Student-spouse scenario | One-flag control |
|---|---:|---:|
| Tax year / filing status | 2024 / married filing jointly | Same |
| Primary taxpayer, age 63 | Full-time student, ineligible | Student flag false, eligible |
| Primary taxpayer's qualifying contributions | $0 | $0 |
| Primary taxpayer's 2024 countable distribution | $1,200 | $1,200 |
| Secondary taxpayer, age 62 | Eligible | Eligible |
| Secondary taxpayer's qualifying contributions | $8,000 | $8,000 |
| Secondary taxpayer's countable distribution | $6,500 | $6,500 |
| Calculated AGI | $43,374 | $43,374 |
| Engine's distributions reducing secondary contributions | $6,500 | $7,700 |
| Engine's Saver's Credit | **$750** | **$150** |
| Engine's total tax | **$668** | **$1,268** |

Only the primary taxpayer's `isStudent` input differs between these two fixtures. These are fact-graph inputs, not a demonstration that every input combination can pass the historical live site's interview and submission gates.

[2024 Form 8880](https://www.irs.gov/pub/irs-prior/f8880--2024.pdf), line 4 and its instructions, include both spouses' countable distributions for a joint return. The exception concerning distributions in years when the couple did not file jointly does not apply to this example: both distributions are in 2024, the joint-return year. The eligible person's credit calculation is:

```python
contributions = 8000
countable_distributions = 6500 + 1200
credit = min(max(contributions - countable_distributions, 0), 2000) / 2
assert credit == 150
```

The liability limit does not bind in this witness. The $2,000 amount is a per-person **contribution base cap**, not a $2,000 per-person credit. We demonstrate a $600 difference; we do not extrapolate taxpayer frequency or total losses. The primary person's own eligibility does not remove their otherwise countable distribution from the eligible spouse's calculation.

Relevant source: [`saversCredits.xml`](https://github.com/IRS-Public/direct-file/blob/e365f9f43446010af36cccf7edecc52953c00fe9/direct-file/backend/src/main/resources/tax/saversCredits.xml), particularly `primaryFilerSaversEligibleGrossDistributions`, `secondaryFilerSaversEligibleGrossDistributions`, and their joint-filer aggregations. The distribution sums themselves are gated by the distributing person's credit eligibility.

Suggested correction: separate eligibility to receive a credit from whether a distribution must reduce the other spouse's contribution base. Preserve distribution exclusions, joint-filing history and incomplete-input handling. This is a repair proposal, **not a validated production patch**.

## Secondary EITC witness

`repro/witness_eitc_hoh2.json` is an unstubbed fact-graph scenario: head of household, two qualifying children, earned wages $38,700 and calculated AGI $40,958. `/earnedIncomeCredit` returns **$3,116**. The AGI lookup in [IRS Publication 596 (2024)](https://www.irs.gov/pub/irs-prior/p596--2024.pdf), printed page 34, $40,950–$41,000 row, HOH/two-child column, is **$3,115**. The earned-income lookup is larger, so the AGI lookup binds. This is a $1 discrepancy, not a material-loss estimate.

The earlier saved sweep reported 34 discrepant **cells**, each +$1, among 7,889 compared cells. That complete sweep was not rerun in this continuation. Its artifacts remain in `../../irs-direct-file-repro/`. The 5 October replay independently confirmed the specific $3,116/$3,115 witness. The suspected mechanism is cent-level intermediate rounding followed by whole-dollar rounding; a change to the shared monetary type must not be proposed as universally safe.

## Student loan interest: qualified question, not a third confirmed full-return defect

The isolated calculation supplies three upstream values rather than reproducing a complete return. Single filer, MAGI $80,981, qualified interest $777 gives $727. A three-decimal worksheet calculation gives $726.495, or $726 after final whole-dollar rounding; an unrounded-ratio calculation also rounds to $726.

The [2024 Form 1040 instructions](https://www.irs.gov/pub/irs-prior/i1040gi--2024.pdf), Schedule 1 line 21 worksheet, permit a ratio rounded to **at least** three decimal places. A second joint-filer example ($165,495 MAGI, $2,500 interest) gives $2,460; comparisons of $2,458 versus $2,459 depend on ratio precision and tie handling. We retain these as a documented rounding question. We have not established a $1–2 error for all IRS-permitted worksheet conventions or demonstrated its downstream tax effect.

## Version, novelty and service status

Current source pin and metadata are in `REPO.json`; the complete tree is in `TREE.json`. `LOCAL_SOURCE_VERIFICATION.json` verifies 36 XML fact modules plus the checked-in Scala.js bundle against the GitHub tree using Git blob hashes, and records SHA-256. No source change or recompilation was used for the primary replay.

All 28 public PR titles/bodies were reviewed. Six plausibly relevant or ambiguous PRs received additional file/comment review. PR 22's complete 250-file list was examined; its tax changes concern other facts. PR 27 is a broad 3,413-file initial import; its full diff was not reviewed, and the two principal formula blobs were instead checked directly. Three targeted web searches produced no applicable duplicate. This bounded review does **not** establish global priority, completeness across 1,366 forks, or the absence of private/internal corrections. Details and limitations: `NOVELTY_REVIEW.md` and `novelty/`.

The README declares the repository historical and unmaintained. The GitHub API's `archived` flag was nevertheless **false**; issues and discussions were disabled. Describe the README warning accurately instead of asserting a verified GitHub archival event. Treasury's [October 2025 report](https://home.treasury.gov/system/files/131/Report-Replacement-of-Direct-File-2025.pdf) and [TIGTA's March 2026 report](https://www.tigta.gov/sites/default/files/reports/2026-03/2026408006fr.pdf) support suspension of the Direct File service. None of this proves that no one can receive a source-code report or that the code cannot be reused elsewhere.

## Reproduction and limits

`run_replay.py` invokes XML conversion and the checked-in engine sequentially. `RESULT.json`, `RUN_0.json` and `RUN_1.json` record the actual output, commands and timing. Engine work took about one second. The shared one-worker wrapper and Node `--single-threaded` were used; no GPU, installation, database, heavy build, tax account or production transaction.

The full scenarios are not stubs. The student-loan rows explicitly are isolated calculations with stubs. Passing legacy grids for tax tables, child credits, standard deductions or Social Security taxation means only no mismatch in those grids; it is not a certification that those parts of the software are correct in every case.



## Fresh status, 22 October

The public repository head remains the same. The five full fact-graph scenarios and three explicitly isolated student-loan cases were rerun locally on 22 October and reproduce the recorded outputs. A targeted review of the original mailbox found no substantive reply to the 5 October report; the stated earliest publication date of 20 October has passed. Lack of a reply from that historical routing address is not evidence of current IRS production behaviour. No validated production patch or third confirmed student-loan defect is claimed.


## Evidence and limits

The attached archive contains the selected source, reproduction and observed output. AI-assisted investigation and writing; the stated executions were actually performed locally. Numerical jobs ran sequentially with one numerical worker and no GPU. No production account, real funds or actual customer losses were tested. Bounded public-source and issue review does not establish global priority. Private correspondence and contact details are excluded.

Report: CC BY 4.0. Original reproduction: MIT. Bundled upstream source retains its own licence; see the archived notices.
