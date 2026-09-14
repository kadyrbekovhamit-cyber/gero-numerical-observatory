> Archival mirror. [Original report](https://github.com/kadyrbekovhamit-cyber/gero-onnx-lpnormalization-range-audit/blob/main/README.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# ONNX LpNormalization finite-range audit

For finite floating-point inputs, normalization is scale invariant:
`normalize(c*x) = normalize(x)` for every positive finite `c`. Released
ONNX ReferenceEvaluator and ONNX Runtime CPU violate this invariant because
they form the L1 or L2 norm directly in the input's numerical range.

Minimal float32 examples:

| p | input | exact output | ReferenceEvaluator / Runtime |
| ---: | --- | --- | --- |
| 1 | `[3e38, 3e38]` | `[0.5, 0.5]` | `[0, 0]` |
| 2 | `[1e30, 1e30]` | `[1/sqrt(2), 1/sqrt(2)]` | `[0, 0]` |
| 2 | `[1e-30, 1e-30]` | `[1/sqrt(2), 1/sqrt(2)]` | `[0, 0]` |

The large cases overflow the intermediate sum or sum of squares. The small
L2 case underflows the squares to zero. The outputs themselves are ordinary,
finite numbers, so this is avoidable intermediate-range loss rather than an
unrepresentable result.

Run:

```bash
python3.12 -m venv .venv
.venv/bin/pip install --only-binary=:all: --no-compile -r requirements.txt
.venv/bin/python reproduce.py
```

The reproducer checks float32 and float64, positive and negative axis syntax,
an analytical equal-pair oracle, an independent scale-first oracle, and
ordinary/zero controls. It records discrepancies in both the Python reference
implementation and ONNX Runtime CPU. GPU and other execution providers are
not tested.

The stable construction divides by the maximum absolute component first and
normalizes the scaled vector. It never needs to represent the norm in the
original input range.

## Upstream corrections

- [ONNX ReferenceEvaluator PR #8454](https://github.com/onnx/onnx/pull/8454)
- [ONNX Runtime CPU PR #32574](https://github.com/microsoft/onnxruntime/pull/32574)

Both corrections use the same scale-first identity but are tested and
submitted independently. The Runtime PR was compiled from source and its
`LpNormalizationTest.*` suite completed with 10 passes and 3 expected
CUDA-only skips. Replacing the corrected kernel with the old direct-norm path
makes the new finite-range test fail.

## Scope

This establishes a finite-range correctness defect in ONNX ReferenceEvaluator
and the ONNX Runtime CPU provider. It does not establish the behavior of CUDA,
WebGPU, DirectML, WebNN or other execution providers. No performance claim is
made.
