# MLX median overflows while averaging finite central values

Native C++ CPU evidence, verified 15 September 2026. Independent GERO research by Xamit Kadirbekov; AI-assisted preparation. This report distinguishes the executed source pin, later source review and untested environments.

The median of two identical representable values must equal that value. A
finite real median must also lie between the minimum and maximum input. MLX's
even-length median adds the two central values in the result dtype and then
multiplies by one half. The addition can overflow even when the exact median
is representable.

Actual native C++ results on the executed MLX source pin:

| Input dtype | Repeated value | Median of 2 copies | Median of 3 copies |
|---|---:|---:|---:|
| float16 | 60000 | +Infinity | 60000 |
| float32 | approximately 2e38 | +Infinity | original stored value |
| bfloat16 | approximately 1.993842e38 | +Infinity | original stored value |

The float16 input and expected result `60000` are exactly representable. The
intermediate sum `120000` exceeds that dtype's maximum. Negative examples give
the corresponding negative infinity. No chatbot answer is used as an oracle.

## Source and execution identity

- Executed source: `d9add9d11f3154111a4c85f267ec2fd307ecd18e`; all 951 original
  source files were verified unchanged against their recorded hashes.
- Actual C++ API, CPU on macOS arm64; Metal and CUDA disabled. The initial
  selected-grid run reused an earlier library from this same pin. A subsequent
  **complete clean CPU build** finished at 04:01 UTC on 15 September 2026,
  without using that earlier library or its objects. All five archived grid
  and control CSVs were reproduced byte-for-byte.
- The candidate recompiles `mlx/ops.cpp` separately and links that object ahead
  of the unchanged baseline archive. Mutation uses the original compiled
  `ops.cpp` object with the same probe and archive.
- Latest main reviewed on 15 September 2026: `8f76a0aa2bbf9c29698337078db333c9bea1c1bf`. The entire `mlx/ops.cpp` is byte-identical to the executed pin. This later complete tree was not separately built or executed.
- Latest tagged release reviewed: `v0.32.2`, commit
  `1f8e74e3f12f31365464a6867c6579f0e9b29d85`. Its median source is byte-identical
  to the executed median implementation. A separate released runtime was **not**
  executed; this is source equivalence, not another numerical run.

The clean build uses Apple Clang 17, CMake 4.4.3, Ninja 1.13.2, one compiler
job and one thread per numerical library. All 951 original source files were
verified both before and after execution. A fresh download of the exact source
archive also matched its recorded SHA-256.

The portable package's `verified-run/` directory contains the clean-build
receipt, actual commands, build logs and raw outputs. Its `README.md` describes
the offline entry point. The earlier local provenance remains unchanged in
`evidence/provenance.json` and `evidence/paired-verification.json`; their older
"no new full build" fields describe the initial run only.

## Independent oracle and measured grid

`check_grid.py` decodes the actual stored input values exactly, sorts rational
numbers, computes the central value or exact central average, and rounds to the
target format using explicit nearest-even rounding. It does not use another
library's median as ground truth. Signed zero is not distinguished by the
mathematical oracle.

The grid has **624 base vectors** across float16, bfloat16 and float32. It uses
positive/negative range boundaries, subnormal values, ordinary numbers, zero,
and lengths 1–4. Four layouts, both keepdims settings and per-output observations
produce **7,488 rows**. They reuse base vectors and are not 7,488 independent
data sets. Duplication and row reversal preserve the expected median; a
transposed input exercises the alternate reduction axis.

- Original: **540 incorrect outputs**, affecting 60 base vectors in at least
  one layout; all 540 violate the finite-input range bound.
- Candidate: **0 incorrect outputs** under the same exact oracle.
- Original-object mutation: **540**, reproducing the complete baseline CSV
  byte-for-byte.
- The other **6,948** grid rows are unchanged.
- A separate **36-case** control run for NaN, infinities, signed zero, ordinary
  floating inputs and integer promotion is byte-identical before/after. Some
  ordinary controls overlap the main grid; do not add them as independent
  coverage.

## Bounded candidate and limits

`candidate.patch` selects `(lower * 0.5) + (upper * 0.5)` for large magnitudes
and preserves `(lower + upper) * 0.5` for small magnitudes. The latter matters
because halving each minimum subnormal value first would lose a representable
median. Output dtype, odd-length behavior and existing NaN propagation remain
unchanged on the executed checks.

This is a graph-level candidate. Both branch graphs can be evaluated, so the
unused original sum may still overflow internally; the selected output is the
tested property. Performance, compiled-graph optimization, autodiff, GPU
execution, complex medians, full applications and the complete upstream suite
were not tested. No device failure, model accuracy change or production impact
is claimed.

## Prior work and duplicate review

The general overflowing-midpoint problem is old. In particular,
[NumPy issue 22688](https://github.com/numpy/numpy/issues/22688) documents a
related `nanmedian` range failure. This report documents a separately
executed MLX implementation case, not discovery of a new mathematical failure
class.

The saved MLX searches found four median-titled records. The initial
[median addition](https://github.com/ml-explore/mlx/pull/2705) and the later
[NaN-propagation correction](https://github.com/ml-explore/mlx/pull/4146),
including their available discussions, were reviewed. The latter handles
explicit NaN input and is already present in the pin; it does not correct
finite central-value overflow. Other overflow/infinity search hits concern
matrix multiplication, FFT, categorical sampling, kernel failures or timings.
The canonical 96-publication GERO catalog checked before this publication contained no median-overflow report. A refreshed search returned the same 84 issue/PR title-body records, including the four median-specific records already reviewed.
No exact earlier MLX report was found within this bounded review; global
novelty is not guaranteed.

Primary references: [MLX median documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.median.html),
[pinned implementation](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/ops.cpp),
[tagged source](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/ops.cpp).

Research: Xamit Kadirbekov / GERO. AI-assisted preparation with native public
code execution and an independent exact oracle. Upstream Apple MIT notices are
retained in the source and copied implementation files.

## Publication links

[GERO](https://www.gero.uz/research/articles/mlx-median-finite-midpoint-overflow.html) · [GitHub](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/mlx-median-finite-midpoint-overflow.md)

[Evidence archive](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/artifacts/gero-mlx-median-finite-midpoint-evidence-2026-09-15.zip)

SHA-256: `45ea7c88dc7ddd9bc951a0f860dcda04803fb11d611d3eaf9082b636e8285630`.
