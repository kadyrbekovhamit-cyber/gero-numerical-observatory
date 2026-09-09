# Sources and provenance

Recorded 9 September 2026. All measurements use synthetic local inputs. No production model, customer data or deployed service was tested.

| Source | Version / date | Supports |
|---|---|---|
| Official MLX CPU wheel | 0.32.2; fresh publication rerun | Incorrect square gradients, rectangular shape/exception and measured loss update; reproduce.py and wheel-reproduction.json |
| Pinned MLX C++ source | ce916dbbcaa88e433b6fd1e60a17f766d49c27fe | Actual pristine/patched native execution linked to existing CPU archive |
| Inspected MLX main | 24c699ecee2f7c8b2040de8da1c8382c8bcf31c7 | Same affine orientation omission; complete method diff retained |
| Independent scalar harness | native_regression.cpp | 34 configurations, 205 execution scenarios; scalar forward and derivative references |
| Official gather_qmm documentation | inspected 9 September 2026 | transpose=False represents multiplication by the non-transposed physical weight |
| PR #4392 and PR #4051 | linked upstream changes | Related forward GPU fixes; not this VJP orientation correction |
| Original broader wheel probe | probe.py and probe-results.json | Prior transpose/flag/fallback comparisons; not rerun for publication |

Primary sources:
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html
- https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp#L3802
- https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/python/tests/test_quantized.py#L1716
- https://github.com/ml-explore/mlx/pull/4392
- https://github.com/ml-explore/mlx/pull/4051

Source hashes and the compatible archive hash are in source-metadata.json. Exact searches are in duplicate-search.json: four successful queries, one HTTP 403. Priority remains unestablished. Included upstream code retains its MIT license. Public path placeholders replace private absolute paths; they do not alter numerical results.
