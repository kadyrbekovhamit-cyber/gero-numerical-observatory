# Apple MLX: small matrices lose finite log determinants

Independent numerical audit, 13 September 2026. Xamit Kadirbekov / GERO Research.

[Full technical report](REPORT.md) · [Reproduction archive](mlx-slogdet-range-2026-09-13.zip) · [Results](results-summary.json) · [Native tests](native_regression.cpp)

For FP32 A = 2^80 I_2, slogdet returns (1, +inf) instead of (1, 110.90354888959125). At 2^-80 I_2 it returns (0, -inf), incorrectly marking a nonsingular matrix as singular. FP64 examples also reproduce the defect. The small-matrix path forms the raw determinant before taking its logarithm. The main examples have condition number 1.

Compiled clean CPU source: ce916dbbcaa88e433b6fd1e60a17f766d49c27fe, version header 0.32.3. The whole linalg.cpp file matches main 229f5b430df7926743c5b6ac62068cae2ebc8978. No new Python-wheel or GPU result is claimed.

| Variant | Forward failures / 824 | Derivative failures / 32 |
|---|---:|---:|
| Original | 112 | 0 |
| Widen FP32 to FP64 | 56, all FP64 | 0 |
| Rejected row scaling | 4 | 0 |
| LU candidate | 0 | 32 exceptions |

No complete compatible repair is claimed. Widening passes all 428 FP32 checks but leaves FP64 unchanged. Row scaling has a retained counterexample. The LU candidate fixes tested forward values but loses existing autodiff support.

A bounded review of upstream issues, original implementation PR #3416, and our publication catalogue found no exact duplicate. Priority, maintainer acceptance, model impact and performance are unestablished. Finite tests are not a universal correctness proof.

The report is CC BY 4.0; test code and MLX-derived patches retain MIT terms. AI-assisted independent research. See LICENSE.md.
