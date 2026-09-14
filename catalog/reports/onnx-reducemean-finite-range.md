> Archival mirror. [Original report](https://github.com/kadyrbekovhamit-cyber/gero-onnx-reducemean-range-audit/blob/main/README.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# ONNX ReduceMean finite-range audit

`ReduceMean` is mathematically allowed to return a finite representable mean
even when summing the inputs first in the input type would overflow. Released
versions of both the ONNX Python `ReferenceEvaluator` and the ONNX Runtime CPU
provider nevertheless use that avoidable same-type intermediate.

Minimal examples:

| input | mathematical mean | both implementations |
| --- | ---: | ---: |
| float32 `[3e38, 3e38]` | `3e38` | `inf` |
| float64 `[1e308, 1e308]` | `1e308` | `inf` |

The reproducer checks both cases at opsets 17 and 18 against an independent
scale-first NumPy oracle. Both implementations mismatch in all eight
comparisons. Ordinary, all-zero, and exact-cancellation controls pass.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python reproduce.py
```

Tested on macOS ARM64 with Python 3.12, NumPy 2.5.2, ONNX 1.24.0 and ONNX
Runtime 1.29.0 CPU provider.

Upstream corrections:

- [ONNX ReferenceEvaluator PR #8456](https://github.com/onnx/onnx/pull/8456)
- [ONNX Runtime CPU PR #32576](https://github.com/microsoft/onnxruntime/pull/32576)

This is an ordinary numerical-correctness report. It does not test other
execution providers, gradients, performance or production-model impact. The
comparison count is not a count of distinct defects.
