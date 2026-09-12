# MLX LayerNorm bias dtype audit

Prepared 13 September 2026 by Xamit Kadirbekov / GERO Research.

A clean native CPU build of MLX ce916dbbcaa88e433b6fd1e60a17f766d49c27fe reproduces an optional-weight dtype inconsistency: finite float32 bias is narrowed when float16 input has no weight. A one-line candidate changes 106 failed assertions out of 385 to zero; 16 controls remain unchanged.

[Full report](README.md) · [Reproduce](BUILD.md) · [Candidate patch](layernorm-bias-dtype.patch) · [Duplicate review](DUPLICATES.md)

CPU/source scope only. No GPU or installed-wheel numerical result, full newer-main build, maintainer acceptance, model-level impact or formal correctness proof is claimed.
