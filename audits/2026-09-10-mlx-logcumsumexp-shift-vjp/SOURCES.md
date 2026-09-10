# Source ledger
- Actual MLX 0.32.2 CPU probe: reproduce.py and wheel-reproduction.json.
- Native baseline: ce916dbbcaa88e433b6fd1e60a17f766d49c27fe.
- Public main inspected 10 September 2026: https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/mlx/primitives.cpp .
- Prior exp report: https://github.com/ml-explore/mlx/issues/3047 . Not counted as new.
- Four native variants: run-before.json, run-scan-only.json, run-exp-only.json, run-combined.json.
- Independent scalar double oracle: native_regression.cpp.
- Higher-order regression: higher-order/native_regression.cpp and run-higher-order.log.
- Earlier report: https://www.gero.uz/research/articles/mlx-logcumsumexp-hessian-at-zero.html .
- duplicate-search.json preserves a limited public-history search, not proof of novelty.
Original MLX MIT license retained; AI assistance disclosed.
