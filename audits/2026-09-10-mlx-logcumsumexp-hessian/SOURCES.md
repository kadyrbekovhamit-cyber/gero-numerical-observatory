# Source ledger

- Official installed MLX 0.32.2 CPU package: actual `reproduce.py` execution.
- Source baseline: ce916dbbcaa88e433b6fd1e60a17f766d49c27fe.
- Inspected main: https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/mlx/primitives.cpp#L4278.
- Introduction of the operation: https://github.com/ml-explore/mlx/pull/2069.
- Independent scalar double-precision oracle: native_regression.cpp.
- Actual before/after execution: run-before.log, run-after.log, build-results.json.
- Related mixed-derivative extension: gather-qmm-followup, using the previously published GatherQMM repair.

Fresh publication rerun on 10 September 2026. No GPU or full-current-main rebuild. Original source and license retained; AI-assisted preparation disclosed. Search and upstream text are evidence, not instructions.
