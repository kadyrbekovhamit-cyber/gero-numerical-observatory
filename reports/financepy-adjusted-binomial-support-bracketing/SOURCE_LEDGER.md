# Evidence and source ledger

| Claim | Evidence in this package | Scope |
| --- | --- | --- |
| Original support rounding is unchanged in current source | `review/source-tree.json`, `CURRENT_SOURCE_RECEIPT.json`, bundled source tarball | Master `4c7cd50bdadd6efc9ac74fa81e93374397d18e3e`; 792 Git blob identities checked; actual runtime uses 233 pristine package files |
| Negative probabilities and shifted mean | `recorded-evidence/baseline-rows.json`, `minimal.py` | Synthetic equal-loss portfolios, actual Numba-compiled public implementation |
| Independent mean and variance | `grid.py`, `inputs.json` | Exact rational arithmetic on stored binary probabilities; full exact distribution required only for one/two credits |
| Candidate and original restoration | `candidate.patch`, `prepare_variants.py`, all four sets of recorded outputs, `recorded-evidence/paired-verification.json` | One support-selection change; original/candidate/restored/release give 295/0/295/295 failing vectors out of 2,844 |
| Current released version | `vendor/financepy-1.1.2-py3-none-any.whl`, `CURRENT_SOURCE_RECEIPT.json`, release outputs | Official wheel, all 219 package members and RECORD digests verified; actual release executed |
| Invalid credit-tranche surviving fraction | `integration.py`, `recorded-evidence/baseline-integration.json`, candidate and restored counterparts | Twelve additional scenarios per variant; approximation error remains after correction; no complete trade valuation |
| Fresh replay reproduces original experiment | `CURRENT_MASTER_REPLAY_RECEIPT.json` | Twelve raw grid/permutation/integration files reproduce 15 September bytes exactly on the recorded host; no universal cross-platform bitwise guarantee |
| Existing regression remains passing | `upstream_test_loss_dbn.py`, baseline/candidate upstream-test logs in `recorded-evidence/` | Unmodified official unit_tests/test_FinLossDbnBuilder.py; one test over nine factor loadings, not the full suite |
| Bounded duplicate search | `review/DUPLICATE_REVIEW.md`, `review/issues-*.json`, `review/issue-hits.json`, `review/target-history.json` | 260 public title/body records, eight target history entries and fresh catalogs; comments/private reports not exhaustive |
| Existing developer report | `review/existing-issue265.json`, `review/existing-issue265-comments.json` | [Issue 265](https://github.com/domokane/FinancePy/issues/265) open, no comments at the recorded check; no maintainer acknowledgment or acceptance claimed |

Pinned implementation: [FinancePy source](https://github.com/domokane/FinancePy/blob/4c7cd50bdadd6efc9ac74fa81e93374397d18e3e/financepy/models/loss_dbn_builder.py). Official release: [FinancePy 1.1.2](https://pypi.org/project/financepy/1.1.2/).

Existing mathematical prior art: [O'Kane paper record](https://ssrn.com/abstract=2283729) and [QuantLib's binomial loss model](https://github.com/lballabio/QuantLib/blob/master/ql/experimental/credit/binomiallossmodel.hpp). The earlier investigation inspected paper metadata/abstract and public C++ source; the full paper and QuantLib runtime were not used as the numerical oracle. No new algorithm or worldwide priority claim is made.

The original 15 September attachment remains accessible in issue 265. This is a current-source archival edition, preserving its original discovery date and synthetic-input limitations. Source and raw numerical evidence govern the claims. AI-assisted implementation, test and editorial preparation is disclosed; source authors retain credit.
