# Duplicate review — nntrainer LayerNorm affine input gradient

Public GitHub issue and pull request searches were run on 2026-09-07, without
an open/closed filter. Nine queries returned 134 matches representing 94 unique
entries. Every result page reports `incomplete_results=false`, and each query
returned fewer than 100 entries. The exact queries and result metadata are in
`duplicate-index.json`. Full API bodies remain in the working audit directory;
the distributable archive includes metadata and this original review.

No exact public report of this LayerNormalizationLayer gamma/backward defect
was identified in the reviewed results. This is a scoped search conclusion,
not a guarantee of global novelty or maintainer acceptance.

Relevant neighbors:

| Entry | Relationship |
|---|---|
| [PR #1993](https://github.com/nntrainer/nntrainer/pull/1993) | Introduces LayerNorm. Its original derivative code already has the problematic placement of gamma; it does not report or fix this defect. This audit does not claim a recent version regression. |
| [PR #2053](https://github.com/nntrainer/nntrainer/pull/2053) | Changes input lifetime for backward memory use; its stated scope is distinct. |
| [Issue #3785](https://github.com/nntrainer/nntrainer/issues/3785) | NaN forward outputs in a build with ENABLE_FP16. This audit uses ordinary FP32 training inputs and an incorrect input derivative. |
| [PR #4096](https://github.com/nntrainer/nntrainer/pull/4096) | Adds an inference width-axis fast path. The inspected LayerNorm patch does not modify calcDerivative. |
| [PR #4304](https://github.com/nntrainer/nntrainer/pull/4304) | Refactors backend dispatch for inference; inspected LayerNorm changes preserve the training composite path and do not repair this derivative. |
| [PR #3859](https://github.com/nntrainer/nntrainer/pull/3859), [PR #4026](https://github.com/nntrainer/nntrainer/pull/4026) | RMSNorm forward/backward support, a different operator and implementation. |

Searches cover LayerNorm names, gamma with gradient/incorrect/ignored, and
normalization with derivative/backward. Nearest PR file diffs (#1993, #2053,
#4096, #4304) were additionally retrieved. Hidden reports, all comments across
all entries, external trackers, and arbitrary unpublished branches are not
covered by this review.

Earlier exclusions remain unchanged: QuantizeLinear is outside scope; the
previous MLX NVFP4 shape/tail behavior is already associated with
[PR #3912](https://github.com/ml-explore/mlx/pull/3912) and is not a new finding.
This report also does not repackage earlier loss, activation, cosine, or Bloom
findings as new discoveries.
