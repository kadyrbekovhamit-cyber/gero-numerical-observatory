# Evidence and source ledger

Review date: 9 September 2026. All material findings describe local, synthetic executions.

| Source | Supported claim |
|---|---|
| MLX 0.32.2 official CPU wheel; probe-results.json | Repeated left selections receive incorrect VJPs only with the sorted path in the small case. |
| quantized-and-loss-results.json | Exact constant affine weights; measured loss 8 to 11.28125 using original gradient, 6.125 with correct gradient. |
| broadcast-tail-results.json | Changing only initialized synthetic backing values changes the logical view's gradient. No foreign-data claim. |
| native_regression.cpp and run-before/after.log | Independent scalar references; 69 scenarios, 626 checks; 71 failures before and zero after. |
| pinned primitives.cpp, ops.cpp and CPU masked_mm.cpp | Faulty optimization guards and the required scatter/broadcast semantics. |
| source-metadata.json and GatherQMM-baseline-to-upstream.diff | Native-compatible base differs from inspected main in QMM global-scale/argument handling. |
| exploratory-dtype-probe/run-after.log | FP16 CPU fallback limitations; excluded from the primary success claim. |
| duplicate-search.json and related-public-reports.json | Bounded seven-query prior-work search, not priority proof. |

Official operation contracts:

- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html

Pinned code:

- https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp
- https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/backend/cpu/masked_mm.cpp

Prior public work:

- https://github.com/ml-explore/mlx/pull/2335 — MoE backward improvements.
- https://github.com/ml-explore/mlx/issues/4253 and https://github.com/ml-explore/mlx/pull/4261 — strided-input forward defect, distinct from this backward case.

The exact small-example derivatives are algebraic calculations; general references are independent double scalar loops. They are not independent third-party certification. Native repair is a proposed local patch with a known FP16 CPU fallback limitation. No full-model, GPU, external-memory, priority or reward claim.
