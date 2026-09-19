# Source and evidence ledger

- MLX executed source: https://github.com/ml-explore/mlx/tree/59d600b5e64c238427d0f8d897ab7c682ef4d3d2
- Erf implementation: https://github.com/ml-explore/mlx/blob/59d600b5e64c238427d0f8d897ab7c682ef4d3d2/mlx/primitives.cpp#L1969
- API: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.erf.html
- VJP: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vjp.html
- Source-equivalent release v0.32.2 (binary not executed): https://github.com/ml-explore/mlx/releases/tag/v0.32.2
- Human-authorship issue template: https://github.com/ml-explore/mlx/blob/main/.github/ISSUE_TEMPLATE/bug_report.md

The included SOURCE_LEDGER.json, SOURCE_MANIFEST.json and receipts provide exact source hashes and bounded duplicate searches. CPU float32 only; 48 raw CSV outputs byte-replayed. Independent mpmath 1.3.0 oracle at 100/150 decimal digits. Balanced candidate resolves 372 of 944 primary finite-target cases but leaves 60 of 304 tail cases and extreme-seed second derivatives failing. Synthetic guard/report only, no production/device impact measured. No tracker report submitted. AI-assisted research and publication preparation.
