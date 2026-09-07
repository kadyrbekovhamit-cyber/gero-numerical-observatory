# Source ledger and pre-render factual review

S1 — MLX Python loss implementation, pinned commit ce916dbbcaa88e433b6fd1e60a17f766d49c27fe.
https://github.com/ml-explore/mlx/blob/ce916dbbcaa88e433b6fd1e60a17f766d49c27fe/python/mlx/nn/losses.py

S2 — nntrainer LayerNormalizationLayer, pinned commit a7ea056e79ab8e14447ea305c1b634e233343258.
https://github.com/nntrainer/nntrainer/blob/a7ea056e79ab8e14447ea305c1b634e233343258/nntrainer/layers/layer_normalization_layer.cpp

S3 — Recorded cosine audit, 7 September 2026: official MLX/Metal 0.32.2, Python
source patch, mpmath 100-digit references, before/after tests and three-process
reproduction. Local immutable archive SHA-256:
09149a442b9592e0ddc3e7c2665cdc85fc3a54ecf720b6cb3b0eb288bb544c8d.

S4 — Recorded nntrainer LayerNorm audit, 7 September 2026: source-built FP32 CPU,
FP64 analytic references, finite differences of actual forward, before/after
tests and three-process reproduction. Immutable archive SHA-256:
feec74e831228c90032a8422c38c710852eed13b72df0da3a2d9522dcb3fc433.

Publication: https://www.gero.uz/research/articles/cosine-and-layernorm-contract-failures.html
Evidence: https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/audits/2026-09-07-cosine-layernorm

Claim mapping:
- Hook and FP16 cosine NaN versus 1: measured result and mathematical target, S1/S3.
- Nonzero LayerNorm input gradient for the shown x, gamma, dy: measured and
  finite-difference-confirmed result, S2/S4. epsilon=1e-5, beta=0, axis=3.
- Repair mechanisms: tested local proposals, S3/S4; no upstream acceptance claim.
- 101 cosine tests per CPU/Metal and 42 nntrainer CPU tests: exact recorded
  counts, S3/S4; not counts of independent defects.
- Three fresh processes: recorded per audit, S3/S4.
- No exact duplicate in saved searches: scoped search conclusion, S3/S4.
- Model impact, device impact, global novelty, version regression, and maintainer
  confirmation are not established. No unsupported claims are scripted.

Pre-render review: every numerical statement maps to retained native/library
executions and references. No health, legal, or financial claims. No third-party
media, synthetic human credentials, cloned voice, or undisclosed sponsorship.
