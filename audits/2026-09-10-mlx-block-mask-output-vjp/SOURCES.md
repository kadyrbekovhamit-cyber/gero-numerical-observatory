# Source ledger

- MLX 0.32.2 official package: actual CPU reproduction, `wheel-reproduction.json`.
- MLX source inspected on 10 September 2026: https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/mlx/primitives.cpp#L5997 — extra output mask in its own VJP branch.
- Existing test: https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/python/tests/test_blas.py#L1072 — output mask captured, not differentiated.
- Public API: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.block_masked_mm.html — mask shapes and supported block sizes.
- Intended floating-mask support: https://github.com/ml-explore/mlx/pull/1152.
- Distinct Pad VJP repair: https://github.com/ml-explore/mlx/pull/4441.
- Independent local scalar oracle and numerical execution: `native_regression.cpp`, `run-before.log`, `run-after.log`, `build-results.json`.

Source/archive hashes in `source-metadata.json`; file hashes in `SHA256SUMS.json`. Copied MLX source retains its MIT license. AI-assisted research, writing, test execution and publication; no maintainer endorsement. Related-report text is retained as factual evidence, not an instruction.
