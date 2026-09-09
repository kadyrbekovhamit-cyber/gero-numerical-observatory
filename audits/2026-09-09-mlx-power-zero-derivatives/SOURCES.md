# Source ledger

- Actual local Python wheel MLX 0.32.2, CPU float32: `probe.py` and `probe-results.json`, rerun on 9 September 2026. Establishes fixed-exponent NaNs, direct-polynomial and finite-difference controls, and the synthetic loss step.
- Actual C++ rerun: `native_regression.cpp`, `run-before.log`, `run-after.log`, `build-results.json`. Establishes 29/41 scenarios before and 41/41 after; 42/396 failed checks before and zero after. Partial native rebuild only.
- Pinned MLX source: https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp#L3452 . Establishes the Power VJP formula; JVP delegates to VJP. Included source retains MIT license.
- Pinned tests: https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/python/tests/test_autograd.py#L699 . AST confirms two definitions of test_power_grad, lines 699 and 734. A later same-name Python method replaces the earlier definition.
- Official MLX power API documentation: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.power.html . Elementwise exponentiation API, live documentation labeled 0.32.2 at publication.
- Historical issue: https://github.com/ml-explore/mlx/issues/504 . Earlier zero-base first-derivative failure.
- Historical fix: https://github.com/ml-explore/mlx/pull/505 . Merged 20 January 2024; replaces division-by-base with the present helper formula and adds earlier zero-base tests. GitHub API metadata, diff and discussion were reviewed; web-page fetch failed.

## Claim classification

The numerical outputs and test totals are measured facts. The falling-factorial derivative, 0×Inf mechanism, and mixed derivative 1/a are algebraic conclusions. Runtime, complete-model impact, maintainer acceptance, general correctness and novelty are not established. Four saved repository searches reviewed in the original evidence (power gradient zero; power/NaN; power/higher; test_power_grad) are not an exhaustive novelty investigation.
