# Public duplicate review — 7 September 2026

Ten GitHub issue/PR searches, open and closed, returned 191 distinct records.
All pages were retained, including both pages of the 127-result `causal` query.
No API response reported incomplete results. See [duplicate-index.json](duplicate-index.json)
for exact queries, counts, URLs and titles. Full retrieved bodies are retained
locally in `duplicate-search.json`, separately from the distributable archive.

Closest relevant records:

- [#1713](https://github.com/nntrainer/nntrainer/issues/1713), supporting masking
  in attention: historical feature request to zero out selected scores. It does
  not report rectangular-mask broadcast failure or offset-chunk behavior.
- [#2407](https://github.com/nntrainer/nntrainer/issues/2407) and
  [#2409](https://github.com/nntrainer/nntrainer/pull/2409): selection of the
  `MultiHeadAttentionLayer` finite mask constant by runtime dtype rather than
  preprocessor macros. The patch changes `multi_head_attention_layer.cpp`, not
  the `AttentionLayer` implementation reproduced here. This is related mask work,
  not evidence that finite subtraction enforces a hard mask for arbitrary logits.
- [#3780](https://github.com/nntrainer/nntrainer/issues/3780): broader incremental
  interface redesign, per-batch/per-input ranges, cache position propagation.
  Its comments discuss asymmetric Q and KV range interpretation and known batch
  limitations. They do not identify the skipped causal mask for a later block
  in this single-batch reproduction. General incremental batch support is excluded.
- [#4040](https://github.com/nntrainer/nntrainer/pull/4040): multi-token prefill
  detection in skip-prefill layers. Its file list does not modify AttentionLayer.
- [#4198](https://github.com/nntrainer/nntrainer/pull/4198): CausalLM chunked
  prefill and FC/GLU layer contracts. Its file list does not modify AttentionLayer.
- [#3989](https://github.com/nntrainer/nntrainer/pull/3989): sliding-window row
  coverage and indexing in MHACore, a distinct implementation and failure mode.

The public file-history API lists 26 commits affecting `attention_layer.cpp`.
Current source is pinned to main `a7ea056e79ab8e14447ea305c1b634e233343258`.
No version bisect or claim of recent regression is made.

Conclusion: no exact match identified in this recorded public search. This is
not global novelty proof or maintainer confirmation. Private reports, unindexed
discussions, every comment and unpublished branches are not exhaustively covered.
