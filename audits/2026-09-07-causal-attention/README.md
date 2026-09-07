# nntrainer causal attention: future values enter masked outputs

A source-pinned native CPU reproduction, 7 September 2026.

With `causal_mask=true`, Q=[0,0], K=[0,0,0] and V=[2,10,50], the original
`AttentionLayer` returns `[20.66666794,20.66666794]` instead of `[2,6]` under
the existing upper-triangular causal convention. The first output has a nonzero
gradient with respect to future V entries. Related cases reproduce finite-mask
penalty failure and later incremental chunks omitting causal masking.

Local patch validation: **42 passed**, compared with **21 failed / 21 passed**
before. The suite includes 28 new and 14 existing tests. Four key cases repeat
identically across three fresh processes per version. No exact duplicate was
identified in ten public searches covering 191 unique issues/PRs; this is not
proof of global novelty or upstream acceptance.

- [Reproduction and scope](evidence/README.md)
- [Native build commands](evidence/BUILD.md)
- [Implementation patch](evidence/patches/nntrainer-attention.patch)
- [Regression tests patch](evidence/patches/nntrainer-tests.patch)
- [Duplicate review](evidence/DUPLICATES.md)
- [Download the immutable evidence ZIP](attention-audit-2026-09-07.zip)
- [Live publication and notification record](../../docs/ATTENTION_PUBLICATION_STATUS.md)

ZIP SHA-256: `f100ce28cd9d86dca67eca7b29dd365f246edd40add1e7f4bfa80dffab0b7554`.

The `evidence/` directory and archive preserve the research-round state, including
the then-unsent draft. Notification/publication status is recorded separately.
Validation is native FP32 CPU NCHW, channel=1, macOS arm64; incremental cases use
batch=1. General incremental batch support, FP16/GPU, other OSes, performance,
model impact, full clean builds and upstream CI are not established. No claims
about specific Samsung devices or a version-to-version regression are made.

## Published report and discussion

- [Upstream report #4333](https://github.com/nntrainer/nntrainer/issues/4333) — awaiting maintainer review.
- [GERO technical note](https://www.gero.uz/research/articles/when-causal-attention-sees-the-future.html)
- [LinkedIn discussion](https://www.linkedin.com/feed/update/urn:li:share:7502761599108182016/)
- [48-second YouTube explanation](https://www.youtube.com/shorts/7q9ka3ePYPM)
