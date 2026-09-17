# Source and claim ledger

Prepared 17 September 2026. Original experiments: 15 September. Current-master replay and proposed upstream correction: 17 September.

| Source | Pin / date | Supported proposition |
|---|---|---|
| [FinancePy source](https://github.com/domokane/FinancePy/tree/4c7cd50bdadd6efc9ac74fa81e93374397d18e3e) | master 4c7cd50bdadd6efc9ac74fa81e93374397d18e3e | Actual executed package; full bundled archive, 792 Git blobs independently checked. |
| [Target implementation](https://github.com/domokane/FinancePy/blob/4c7cd50bdadd6efc9ac74fa81e93374397d18e3e/financepy/products/fx/fx_double_one_touch_option.py) | SHA256 73a35bac63e9bebf834a12d556cd70278f8970b0af4d04dc12b198ff56814729 | Single-small-term exit and the original damping threshold; cash-at-expiry contract and complementary price calculation. |
| [Original source](https://github.com/domokane/FinancePy/tree/2b9227fea9d832c4033421d6cd53a54316414fca) | 15 September experiment pin | Same target file bytes; original whole repository is referenced, not included a second time. |
| [Official PyPI release](https://pypi.org/project/financepy/1.1.2/) | 1.1.2 | Bundled released wheel; its SHA256 and all 219 FinancePy package files verified. |
| [Existing issue266](https://github.com/domokane/FinancePy/issues/266) | filed 15 September | Original public report; this archive does not constitute another discovery. |
| [Maintainer response](https://github.com/domokane/FinancePy/issues/266#issuecomment-5698752302) | 16 September | Maintainer independently reproduced invalid prices, calls this a bug and requests a focused PR. |
| [Proposed correction PR270](https://github.com/domokane/FinancePy/pull/270) | f97e9251de71a17d44951cd65d90bd50373e2be6 | Exact two-file change tested here. Open when this record was prepared; not a merged/released-fix claim. |
| grid.py and recorded JSON/XML | local CPU execution | 480 scenarios and 89 focused tests, original / partial / candidate / restoration / release comparisons. |

Independent oracle: a method-of-images solution of the absorbing drifted Brownian transition-density problem, integrated between the log barriers, with 60-decimal arithmetic. It does not call FinancePy's spectral series. The report describes its variables and convergence checks. Numerical agreement in the sampled parameter domain does not prove convergence over all possible inputs.

Duplicate review: current target history, existing issue266 and its comments, open PRs, latest issue batch, focused searches for double-touch/no-touch, truncation and target filename, and both current GERO/canonical GitHub catalogs were reviewed on 17 September before archival publication. The exact own issue was found; no separate matching report or fix was found in that bounded review. Unrelated historical barrier requests and scalar/array issues are not duplicates. No exhaustive worldwide-novelty claim.
