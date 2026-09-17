# Source and claim ledger

| Source | Supported proposition | Scope |
|---|---|---|
| nntrainer source `a7ea056e79ab8e14447ea305c1b634e233343258`, `nntrainer/layers/split_layer.cpp` | Accepted shape mapping; channel-zero address literals; ordinary-forward fallback | Actual executed C++ target |
| Included `probe.cpp`, original and candidate translation units | Inputs, output sentinel and actual layer calls | Synthetic FP32 NCHW full-prefill |
| `evidence/clean-paired-verification.json` and three `clean-*.tsv` files | 576 scenarios, 19,296 coordinates, 4,320→0→4,320; controls and byte identity | Original 15 September measurements |
| `reproduce.py` | Independent exact output-index oracle and standalone native replay | Validation status must come from its completed receipt |
| `review/head.json` and `review/split_layer.cpp` | Current-main target unchanged at refresh | Source comparison only |
| Official latest-release metadata in `review/release-latest.json` | v0.5.0 predates the feature | No executed release claim |
| Existing https://github.com/nntrainer/nntrainer/issues/4337 | Earlier disclosure to developers | Sent, not acknowledged or accepted |
| Issues4334/4335 and PR3998/4011/4054/4067 in `review/` | Prior related findings and introduction/refactor status | Distinguishes overlap; no global novelty proof |
| `review/canonical-catalog.json` | 102-entry canonical publication inventory at review | Catalog snapshot, not independent-defect count |

All numerical claims concern the real public library and retained outputs, not a chatbot's answer. Research and editorial preparation were AI-assisted. No production or device impact is inferred.
