# Parallel vectors produce nonzero cross products in Apple MLX

Independent, AI-assisted research by Xamit Kadirbekov / GERO Research, 13 September 2026.

The native linalg.cross implementation misaligns component axes when broadcasting inputs of different ranks with a nonnegative axis. A clean CPU build at ce916dbbcaa88e433b6fd1e60a17f766d49c27fe runs 3,383 scenarios: 594 fail before the candidate correction and zero fail after. All 2,789 control records are unchanged. This is one defect; GPU, compiled graphs, large-model impact, priority and maintainer acceptance are unestablished.

[Full report](https://www.gero.uz/research/articles/mlx-cross-axis-broadcast.html). The native tests, patch and raw logs are included in this audit folder.
