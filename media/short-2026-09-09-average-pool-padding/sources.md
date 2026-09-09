# Source ledger — reviewed 9 September 2026

| ID | Source | Supported proposition | Classification |
|---|---|---|---|
| S1 | https://github.com/nntrainer/nntrainer/blob/a7ea056e79ab8e14447ea305c1b634e233343258/nntrainer/layers/pooling2d_layer.cpp | Tested implementation and original average-backward loop bounds | Primary source |
| S2 | ../../audits/2026-09-09-nntrainer-average-pool-padding/README.md | Native C++ reproduction, manual example, finite-difference oracle, 58-test before/after result and stated limits | Reproduced measurement |
| S3 | ../../audits/2026-09-09-nntrainer-average-pool-padding/results.json | Machine-readable 8/58 failures before, 0/58 after, and exact counterexample values | Reproduced measurement |
| S4 | https://github.com/nntrainer/nntrainer/pull/1360 | Historical upstream change already used bottom/right backward boundaries | Public history / novelty boundary |

All material numerical claims are direct transcriptions of S2–S3. The phrase
"transpose Jacobian" is a mathematical interpretation of the forward/backward
contract. No claim is made about a released Samsung product, security, monetary
reward, universal platform behavior or first discovery.

All graphics are generated locally from the four-value counterexample. Narration
is synthetic and disclosed. No music, third-party imagery or cloned voice is used.
