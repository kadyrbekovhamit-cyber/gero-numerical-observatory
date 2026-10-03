# loan-calculator gross-up evidence, version 1

Xamit Kadirbekov / GERO, 3 October 2026. AI-assisted, not peer reviewed.

Read REPORT_EN.md. Two formula defects, not 14 distinct bugs.
Official maintainer report: https://github.com/yanomateus/loan-calculator/issues/15

## Reproduce offline

Python 3.9+ on a Unix-like system; standard library only. The original MIT-licensed
loan-calculator 1.2.2 wheel is included with its license and is never modified.
Run the following commands sequentially from this directory:

```sh
python3 verify.py baseline
python3 verify.py patched
python3 verify.py restored
```

Expected summaries: 36 cases in each run; 14, 0 and 14 failures respectively.
Outputs are replay-baseline.json, replay-patched.json and replay-restored.json;
the original receipts in evidence/ stay unchanged. The proposed correction is
applied only inside a temporary directory. Set GERO_COMPUTE_LOCK to an existing
shared lock if coordinating with other workers. Default: a lock in the OS temp
directory. No network, GPU, paid service or library installation is needed.

The harness has only path/lock/output adaptations from the original verified
script; mathematical and test logic is unchanged. Minor last-digit differences
may occur on other platforms. Tolerances and actual binary inputs are recorded.

All main cases have equal payment intervals, zero grace and equal loan/reference
dates. No current tax-law, deployed customer-loss, security, bounty, worldwide
priority or maintainer-acceptance claim is made. A proposed patch is not an
upstream release. The bounded novelty search covered current upstream issues,
PRs, comments, direct history and releases; it cannot establish global priority.

Original text: CC BY 4.0. Original verification code: MIT. The bundled third-party
wheel retains Mateus Yano's MIT license. SHA256SUMS identifies this immutable
archive. Later platform links are maintained in the live report, not by replacing
this archive under the same identity.
