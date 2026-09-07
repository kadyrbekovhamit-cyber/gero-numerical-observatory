# Cosine and LayerNorm mathematical contract failures

Two locally reproduced implementation reports, 7 September 2026. Exact inputs,
source pins, before/after logs, references, isolated patches, and duplicate-search
records are preserved in separate packages.

| Package | Reproduced behavior | Local patch validation |
|---|---|---|
| [MLX cosine](cosine/README.md) | Finite self-similarity becomes NaN; zero-vector derivative becomes NaN | 101 tests pass on CPU and Metal; CPU baseline has 62 failures |
| [nntrainer LayerNorm](layernorm/README.md) | Affine gamma is absent from the input derivative on tested shapes | 42 native FP32 CPU tests pass; 20 fail before |

The key examples were repeated in three fresh processes per audit. No exact
duplicate was identified in the saved public searches. Maintainer confirmation,
global novelty, version regression, performance and model-level impact are not
established. No claims about particular Apple or Samsung devices are made.

[Read the technical note](https://www.gero.uz/research/articles/cosine-and-layernorm-contract-failures.html)

The subdirectories and their SHA256SUMS preserve the research packages unchanged.
Their pre-publication status statements describe the completed search rounds;
live publication status and crosslinks are maintained separately in
[the publication record](../../docs/COSINE_LAYERNORM_PUBLICATION_STATUS.md).

The cosine repair changes Python source executed with official MLX/Metal 0.32.2.
The nntrainer repair was tested in a native FP32 CPU build. Neither result should
be described as a full upstream CI run or an upstream-accepted fix.
