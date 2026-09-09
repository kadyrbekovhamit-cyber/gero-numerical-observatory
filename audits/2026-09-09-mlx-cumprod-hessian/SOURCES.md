# Sources and evidence ledger

Checked 9 September 2026. Classification: measured results, mathematical
references, public source/history and unestablished claims are separate.

- Primary implementation: https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp
- Repeated differentiation documented in quick start: https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/docs/src/usage/quick_start.rst
- Prior first-order fix: https://github.com/ml-explore/mlx/pull/1167
  Live GitHub API confirms merged 2024-05-31T19:28:42Z, title Stable cumprod
  grad at 0. Its patch adds first-gradient checks for zero positions and scan
  modes, without differentiating those gradients again.
- Related https://github.com/ml-explore/mlx/issues/4114 concerns second-order
  differentiation of sort/partition/topk/cummax/cummin through indices.
  It is distinct from the division-related cumprod NaNs tested here.
- The input audit searched repo:ml-explore/mlx cumprod (11/11), "second"
  "zero" (73/73), and "Hessian" (9/9); no incomplete_results flags. No exact
  match was identified in that selection. This does not establish novelty.
- Current main was rechecked before publication and still matched the pin.
- Actual wheel results: probe-results.json. Actual native runs: run-before.log
  and run-after.log. Independent monomial reference: native_regression.cpp.
- Patch is a division-free prototype; O(N log N) follows from array-wide work
  at doubling distances, not a measured performance benchmark. Memory costs
  and a work-efficient alternative remain open.

An unavailable web cache of PR #1167 was not used as evidence of its status;
the authenticated public GitHub API supplied the successful current response.
No delivery status, private correspondence or personal mailbox content is
included in this public evidence package.
