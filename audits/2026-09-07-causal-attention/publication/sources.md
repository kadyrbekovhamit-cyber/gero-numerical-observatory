# Source ledger and pre-render factual review

S1 — nntrainer AttentionLayer, audited commit a7ea056e79ab8e14447ea305c1b634e233343258:
https://github.com/nntrainer/nntrainer/blob/a7ea056e79ab8e14447ea305c1b634e233343258/nntrainer/layers/attention_layer.cpp

S2 — Immutable native evidence, tests and local patch:
https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/dbd6a98d7c4213d1df26252d4b5cc053c18c05d3/audits/2026-09-07-causal-attention
Archive SHA-256: f100ce28cd9d86dca67eca7b29dd365f246edd40add1e7f4bfa80dffab0b7554

S3 — Submitted upstream issue, verified open with the reviewed body:
https://github.com/nntrainer/nntrainer/issues/4333

Claim map:
- Future influence / output 20.66666794 versus 2,6: S2 native minimal case,
  Q shape [1,1,2,1], K,V [1,1,3,1], scaled_dot_product=false, causal_mask=true.
  Causal target follows existing upper-left convention; rectangular alignment is
  explicitly raised for maintainer discussion in the report.
- Final V changed from 50 to -100: retained S2 native perturbation test.
- Three mechanisms and repair: pinned S1 source and S2 patch with tests.
- Before 21 failed/21 passed, after 42 passed: S2 logs, 28 new+14 existing tests.
- Key cases repeated in three fresh processes each version: S2 reproduction logs.
- Open issue and pending review: S3; no upstream acceptance claim.

Limits: CPU FP32 NCHW channel1 macOS M4; full batch1/2, incremental batch1.
Reused build contains unrelated prior fixes but attention baseline matches pin.
No full clean repository rebuild, other OS, FP16/GPU, performance or model impact.
Duplicate review: 10 queries, 191 unique public records; no exact match in scope.
Global novelty and version regression not established. QuantizeLinear excluded.

Pre-render review: all numbers and mechanisms map to recorded executions or
pinned source. Mathematical expected results are labeled targets. Test counts
are not discovery counts. No commercial-device, health, legal or financial claims.
