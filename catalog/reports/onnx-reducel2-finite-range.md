> Archival mirror. [Original report](https://github.com/kadyrbekovhamit-cyber/gero-onnx-reducel2-range-audit/blob/main/README.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# ONNX ReduceL2 finite-range audit

`ReduceL2` should return the Euclidean norm of a tensor slice. The released
ONNX Python `ReferenceEvaluator` and ONNX Runtime CPU provider form the sum of
squares directly. That avoidable intermediate overflows or underflows even
when every input and the final norm are finite and representable.

Minimal float32 examples:

| input | exact L2 norm | both implementations |
| --- | ---: | ---: |
| `[1e30, 1e30]` | `1.4142135e30` | `inf` |
| `[1e-30, 1e-30]` | `1.4142136e-30` | `0` |

The same construction fails with float64 at `1e200` and `1e-200`. The
reproducer checks all four cases at opsets 17 and 18 against an independent
scale-first NumPy oracle. Both implementations mismatch in all 16 comparisons.
Ordinary `[3,4]` and all-zero controls pass.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python reproduce.py
```

Tested on macOS ARM64 with Python 3.12, NumPy 2.5.2, ONNX 1.24.0 and ONNX
Runtime 1.29.0 CPU provider.

This is an ordinary numerical-correctness report. It does not test other
execution providers, gradients, performance or production-model impact. The
comparison count is not a count of distinct defects.
