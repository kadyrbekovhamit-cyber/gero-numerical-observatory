> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/mlx-slogdet-range.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# Apple MLX: small matrices lose finite log determinants

**Independent numerical audit · 13 September 2026 · Xamit Kadirbekov / GERO Research**

The small-matrix fast path in `mlx.core.linalg.slogdet` forms a determinant before taking its logarithm. For finite, well-conditioned 2×2 and 3×3 inputs, that intermediate value can overflow or underflow even though the requested log-determinant is finite. Underflow also returns sign zero, incorrectly classifying the tested nonsingular matrices as singular.

A clean native CPU build reproduces the defect. **A complete, compatible repair is not claimed.** A tested LU candidate fixes the forward cases but loses existing small-matrix autodiff. A row-scaling candidate was rejected after a counterexample. Widening FP32 input to FP64 is a measured CPU workaround for the FP32 cases; it does not solve the FP64 defect.

## Minimal evidence and mathematics

For `A = 2^k I_n`, every input is finite and exactly represented in the tested dtype, the 2-norm condition number is 1, the determinant sign is +1, and

```text
log(abs(det(A))) = n k log(2).
```

| Input dtype | Matrix | Observed `(sign, logabsdet)` | Expected |
|---|---|---|---|
| FP32 | `2^80 I_2` | `(1, +inf)` | `(1, 110.90354888959125)` |
| FP32 | `2^-80 I_2` | `(0, -inf)` | `(1, -110.90354888959125)` |
| FP64 | `2^600 I_2` | `(1, +inf)` | `(1, 831.7766166719343)` |
| FP64 | `2^-600 I_2` | `(0, -inf)` | `(1, -831.7766166719343)` |

For FP32, the direct intermediate determinant `2^160` exceeds its finite range and `2^-160` is below its smallest subnormal. Neither limitation applies to the modest requested logarithm. For FP64, the corresponding intermediate powers are `2^1200` and `2^-1200`. No ill-conditioned matrix or NaN input is required.

The same experiment at size 4 uses the existing LU path and returns finite results. This is an algorithm-selection discrepancy, not an assertion that ordinary `det` must represent an out-of-range determinant.

Executable C++ example, using the actual MLX runtime:

```cpp
#include <iostream>
#include "mlx/mlx.h"
int main() {
  namespace mx = mlx::core;
  mx::set_default_device(mx::Device::cpu);
  auto a = mx::multiply(mx::eye(2, mx::float32), mx::array(0x1p80f));
  auto [sign, logabs] = mx::linalg::slogdet(a);
  mx::eval(sign, logabs);
  std::cout << sign << "\n" << logabs << "\n";
}
```

## Contract and source

The [MLX API docstring](https://github.com/ml-explore/mlx/blob/229f5b430df7926743c5b6ac62068cae2ebc8978/python/src/linalg.cpp#L702) specifically describes improved numerical stability for large and small determinants. It defines `(0, -inf)` for singular matrices.

In [`mlx/linalg.cpp`, lines 807–813](https://github.com/ml-explore/mlx/blob/229f5b430df7926743c5b6ac62068cae2ebc8978/mlx/linalg.cpp#L807), the `n <= 3` branch calls `det_raw_small(input)` and then `log(abs(raw))`. The 2×2 raw formula is `a00*a11 - a01*a10`; the 3×3 formula likewise multiplies the unscaled entries. The larger-matrix branch instead sums the logarithms of LU diagonal magnitudes.

This is one underlying range defect with several inputs and dtypes, not 112 independent bugs.

## Actual validation

The [native suite](https://www.gero.uz/research/data/2026-09-13-mlx-slogdet-range/native_regression.cpp) runs 856 assertions per variant: **824 forward assertions** (436 numerical sign/log comparisons and 388 size/dtype checks), plus **32 derivative controls**. The latter check first and second derivatives of `log(abs(det(t I_n)))` at moderate, nonzero `t`.

| Variant | Forward failures / 824 | Derivative failures / 32 | Interpretation |
|---|---:|---:|---|
| Original clean source | 112 | 0 | Defect reproduced |
| Widen only FP32 small inputs to FP64 | 56, all FP64 | 0 | All 428 FP32 assertions pass; FP64 unchanged |
| Row-scaling candidate | 4 | 0 | Rejected: two counterexample matrices lose sign and log |
| LU forward candidate | 0 | 32 exceptions | Forward mitigation; incompatible autodiff |

The suite includes diagonal and triangular matrices, both determinant signs, transposed views, dense batches, mixed row scales, singular matrices, 0×0 and 1×1 controls, and sizes 2, 3 and 4. FP32 exponents range from -120 to +120, FP64 from -900 to +900. References use known integer determinants and sums of input exponents; they never form an overflowing reference determinant. Comparison tolerances are `2e-6*(1+abs(expected))` for FP32 and `2e-13*(1+abs(expected))` for FP64, multiplied by ten for derivative controls. Signs, sizes and dtypes are exact checks.

The [summary](https://www.gero.uz/research/data/2026-09-13-mlx-slogdet-range/results-summary.json) and four complete `clean-*.jsonl` logs retain passing as well as failing observations. Metadata assertions and numerical comparisons are counted separately above. These finite tests are not a proof for all inputs.

## Repair options and counterevidence

**Measured FP32 CPU workaround:** cast a small FP32 input to FP64, compute `slogdet`, then cast the two results back to FP32 if needed. The suite executes this wrapper against the original C++ library. It passes all 428 FP32 checks, including the added extreme-row counterexample and the moderate derivative controls. No analogous widening result is claimed for FP64. This does not guarantee accurate determinants for arbitrary nearly singular matrices.

**LU candidate:** [slogdet-lu-forward.patch](https://www.gero.uz/research/data/2026-09-13-mlx-slogdet-range/slogdet-lu-forward.patch) keeps only 0×0 and 1×1 in the direct branch and uses the existing LU route for larger matrices. All 824 forward checks pass. However, MLX's LU primitive has no tested VJP implementation: the 32 derivative controls throw `Not implemented for LUF` errors. This is a documented regression of this candidate, not a production-ready repair. Integration should preserve the small-matrix derivative behavior, for example with an explicitly implemented and independently tested log-determinant transform, or adopt a stable small-matrix algorithm. That further implementation is not part of this report.

**Rejected row-scaling candidate:** [slogdet-row-scaling.patch](https://www.gero.uz/research/data/2026-09-13-mlx-slogdet-range/slogdet-row-scaling.patch) passed the initial 848-check selection but failed the expanded selection. Consider

```text
A = [[H, 1/H],
     [H, 2/H]],       det(A) = 2 - 1 = 1.
```

At `H=2^120` in FP32 or `H=2^600` in FP64, all inputs and both raw products are representable. Dividing each row by `H` erases the small column; the candidate returns `(0, -inf)` instead of `(1, 0)`. The original and LU variants pass this counterexample. This matrix is ill-conditioned, so it is used to reject the proposed general repair, not as the primary evidence for the original defect. The main diagonal examples above have condition number 1.

## Versions and reproduction

- Compiled source: `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`; its version header reads **0.32.3**. This is a source-build finding, not a newly measured 0.32.2 wheel result.
- Current `main` inspected: `229f5b430df7926743c5b6ac62068cae2ebc8978` on 13 September 2026. Its entire `mlx/linalg.cpp` is byte-identical to the compiled baseline. The entire newer commit was not rebuilt.
- AppleClang 17.0.0, macOS 15.5 arm64, Accelerate, CMake/Ninja, `Release -O0`, CPU build with Metal and CUDA disabled. The final source builds used one build job and the numerical thread environment recorded in [source-metadata.json](https://www.gero.uz/research/data/2026-09-13-mlx-slogdet-range/source-metadata.json).
- Dependencies are pinned in the source metadata and the baseline MLX CMake definitions. The initial scout used an existing support library; **all reported comparison counts above were reproduced with the full clean source build**.

Install the MLX build prerequisites, CMake and Ninja, then:

```bash
git clone https://github.com/ml-explore/mlx.git mlx-source
python3 build_and_test.py --repo ./mlx-source
```

An optional `--deps /path/to/existing/_deps` reuses downloaded `fmt-src` and `json-src`. Source extraction is pinned to the recorded commit. The runner preserves expected nonzero test exits: reproducing defects and rejected candidates is the purpose of this suite. A completed runner does not mean every candidate passed. Build output is saved in `variant-build.log`, with initial clean compilation also archived in `clean-build.log`.

## Prior work, publication check and limits

The bounded GitHub API search for `repo:ml-explore/mlx slogdet` returned [the original implementation PR #3416](https://github.com/ml-explore/mlx/pull/3416), [feature request #3335](https://github.com/ml-explore/mlx/issues/3335) and [alternative PR #3415](https://github.com/ml-explore/mlx/pull/3415). Their comments/reviews and the `linalg.cpp` commit history were inspected. The separate query `repo:ml-explore/mlx determinant overflow` returned no results. No exact report of this small-matrix range defect was found in that review. This is not an exhaustive novelty guarantee or a claim of priority.

Before publication, the 68-document GERO catalogue and own GitHub report tree had no `slogdet`/determinant report; the authenticated Zenodo upload search had no `slogdet` match. Publication receipts are maintained separately from the immutable evidence package to avoid claiming future postings as completed.

No GPU, Python-wheel numerical run, compiled graph, full upstream test suite, full model, application impact or performance result is claimed. This work is independent, prepared with AI assistance, and has not been accepted by MLX maintainers. The evidence does not establish an Apple device defect, a security impact, or eligibility for compensation.

Report: CC BY 4.0. Original test/runner code: MIT. MLX source and patches retain the MLX MIT license; see [LICENSE.md](https://www.gero.uz/research/data/2026-09-13-mlx-slogdet-range/LICENSE.md) and [MLX-LICENSE](https://www.gero.uz/research/data/2026-09-13-mlx-slogdet-range/MLX-LICENSE).


## Publication and archive

[Zenodo](https://doi.org/10.5281/zenodo.22731843) · [Evidence ZIP](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/mlx-slogdet-range-2026-09-13.zip) · [GitHub](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/audits/2026-09-13-mlx-slogdet-range) · [GERO article](https://www.gero.uz/research/articles/mlx-slogdet-range.html)

#MLX #NumericalStability #LinearAlgebra #ReproducibleResearch
