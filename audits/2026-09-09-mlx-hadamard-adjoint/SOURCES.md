# Sources and claim ledger

9 September 2026. Primary numerical evidence is the retained local execution. Independent references are exact integer products, explicit matrix multiplication and actual-forward finite differences.

| Claim | Source | Classification and limit |
|---|---|---|
| Supported sizes and normalization | [MLX 0.32.2 operation documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.hadamard_transform.html) | Published API contract; read 9 September 2026. |
| Reverse rule calls the forward derivative | [Pinned primitives.cpp](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp#L6295) | Direct source observation. |
| H20/H28 are orthogonal but nonsymmetric | [Pinned matrix tables](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/backend/common/hadamard.h), exact-matrix-certificate.json and validate_artifacts.py | Exact integer calculation, regenerated locally. |
| Wrong first component and increased synthetic energy | probe.py / probe-results.json; e0_step.cpp / run-e0-before.log | Actual CPU measurements on explicitly given inputs. |
| Corrected update lowers energy | run-e0-after.log | Actual local patched C++ execution, supplementary to the primary suite. |
| 64 scenarios, 624 checks; 186 baseline failures, zero after repair | native_regression.cpp and run-before/after.log | Actual sequential executions; targeted selection, not a full-suite proof. |
| Initial implementation history | [MLX PR 1249](https://github.com/ml-explore/mlx/pull/1249) | Public source/history reviewed; derivative tests cover symmetric powers of two. |
| Previous non-power-of-two GPU issue is a different mechanism | [Issue 4049](https://github.com/ml-explore/mlx/issues/4049), [PR 4054](https://github.com/ml-explore/mlx/pull/4054) | GPU launch issue; does not explain this CPU adjoint counterexample. |

## Search boundary

Four recorded GitHub issue/PR queries in ml-explore/mlx: `hadamard gradient` (1 result), `hadamard vjp` (2), `hadamard transpose` (6) and `hadamard` (17). The supplied records report complete result pages. No exact adjoint duplicate was identified in those returned records. This does not cover every commit, discussion, private report or external source, and is not a priority claim.

## Limits

C++ uses a partial rebuild on a compatible base. CPU only for this publication; no full-model, GPU or complete-suite conclusion. Included MLX code remains MIT licensed. Prepared with AI assistance; no third-party validation or maintainer acceptance is implied.
