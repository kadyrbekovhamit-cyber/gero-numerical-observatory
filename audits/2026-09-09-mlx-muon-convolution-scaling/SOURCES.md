# Source ledger and evidence boundaries

| Source | Role | What it establishes |
|---|---|---|
| [MLX optimizer, pinned source](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/python/mlx/optimizers/optimizers.py#L940) | Primary code | Flatten / Newton–Schulz / restore / scale order |
| [MLX Muon documentation](https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Muon.html) | Primary API documentation, checked 2026-09-09 | Treatment of convolution filters through flattened trailing dimensions |
| [Author's Muon implementation](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py#L34) | Primary algorithm source; reviewed, not run | Scaling with matrix dimensions before caller restores shape |
| [Pinned MLX tests](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/python/tests/test_optimizers.py) | Existing regression body | Baseline test passes but does not distinguish these scale rules |
| `probe.py`, `probe-results.json` | Actual native execution | Three shape comparisons in released API and pinned Python; real convolution/linear equivalence |
| `regression.py`, logs and results | Actual native execution | The focused 31-test selection, original and patched |
| `validation.json`, `source-metadata.json`, `SHA256SUMS.json` | Provenance / local checks | Source pins, patch isolation and file integrity |

The matrix scale formula is an algebraic interpretation of the algorithm;
the half/double step examples are also measured independently in native MLX.

## Public-history review

Checked Muon-related MLX issues and pull requests, plus reshape/convolution
searches, on 9 September 2026. A publication-time `repo:ml-explore/mlx Muon`
search returned six results: #4215, #3196, #2364, #1914, #2172 and #1733.
No exact report of this convolution scale-order discrepancy was identified
in that scope. Search indexing, unpublished work and other reports may be
incomplete. No first-discovery or bounty claim is made.

- [Original Muon pull request #1914](https://github.com/ml-explore/mlx/pull/1914)
- [Another Muon proposal #2172](https://github.com/ml-explore/mlx/pull/2172)
- [Muon proposal #2364](https://github.com/ml-explore/mlx/pull/2364)

The source reference is not maintainer endorsement. This package is
published in GERO's repository, not an accepted upstream change.
