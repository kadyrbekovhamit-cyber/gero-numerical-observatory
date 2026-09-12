# Bounded public duplicate review

Reviewed 12–13 September 2026 (Asia/Tashkent). No exact matching report was found
in the material below. The search does not establish absolute novelty.

- [GitHub `layer_norm bias`](https://github.com/ml-explore/mlx/issues?q=layer_norm%20bias):
  three results, #3630, #3220, #2300. Refreshed before publication on 13 September;
  same three results, without an open-only filter.
- [LayerNorm page 1](https://github.com/ml-explore/mlx/issues?q=LayerNorm&page=1)
  and [page 2](https://github.com/ml-explore/mlx/issues?q=LayerNorm&page=2):
  35 titles inspected in the preceding audit (1 open, 34 closed).
- Focused public web searches for MLX LayerNorm / bias / dtype / precision /
  None, including a publication-time search. Search indexing can miss recent work.

Titles on page 1: 4312, 3819, 3867, 3630, 3613, 3382, 3220, 3231, 1057, 2300,
2864, 341, 2340, 2271, 2246, 2238, 2280, 2333, 1915, 1979, 1653, 1364,
1427, 1232, 1041. Page 2: 1328, 1049, 958, 883, 933, 243, 1153, 687, 267, 167.
Title review is not a full reading of every discussion.

## Closest reports

| Report | Relationship |
|---|---|
| [PR #3630](https://github.com/ml-explore/mlx/pull/3630) | Intermittent Metal bias-gradient hazard; CPU is a correct control in that report. This case is deterministic dtype narrowing before a CPU forward operation. |
| [Issue #3220](https://github.com/ml-explore/mlx/issues/3220) / [PR #3231](https://github.com/ml-explore/mlx/pull/3231) | Wrong placeholder shape when bias is absent, affecting derivatives. Here bias is present, weight is absent, and forward already differs. |
| [Issue #1041](https://github.com/ml-explore/mlx/issues/1041#issuecomment-2080795728) | The reviewed maintainer reply attributes batch dependence to a missing softmax axis in user code, not this LayerNorm branch. |
| [PR #2300](https://github.com/ml-explore/mlx/pull/2300) | Broad experimental ROCm backend. Its title was reviewed; all 105 comments were not reread. |

The author's earlier GroupNorm and BatchNorm reports use different operators
and causes. The separate float64 LayerNorm-statistics observation is not included
as another completed finding here. No upstream issue or PR was created by this
publication package; no maintainer acceptance, security impact or payment eligibility
is asserted. Failed network/API requests were not counted as empty search results.
