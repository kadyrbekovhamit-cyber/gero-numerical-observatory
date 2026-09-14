# MLX CPU quantized matmul: avoidable accumulation error in float16 and bfloat16

Xamit Kadirbekov · GERO Research · 14 September 2026

**An executed C++ audit of MLX's non-transposed CPU `fp_qmm` path finds large errors caused by repeatedly storing the running sum in the output dtype.** Adding 6 exactly 1,024 times should give 6,144. This path returns 2,048 in bfloat16 and 6,824 in float16. Equivalent transposed and dense controls return 6,144. A local float32 row accumulator removes all 24,192 mismatching output coordinates in the selected grid; reverting that accumulator to the output dtype reproduces the original CSV byte for byte.

This report documents one implementation issue and a separately bounded midpoint-rounding observation. It does not establish effects on GPU execution, complete models, or deployed products. The local patch has not been accepted upstream.

## Source and execution

- MLX commit: `d9add9d11f3154111a4c85f267ec2fd307ecd18e`, also the `main` revision returned by the GitHub API during the publication check on 14 September 2026.
- Native C++ Release build on macOS 15.5 arm64, Apple Clang 17. CPU enabled; Metal and CUDA disabled. No Python MLX package was involved.
- All 951 original source files were checked against the source manifest. The original source tree, static library and baseline executables were left unchanged.
- Source archive SHA-256: `f3ff5fa6dfe738d7a79663fd7a2815dd558bf521163ba0aa867b77a54b0ed8b3`.

The [pinned CPU implementation](https://github.com/ml-explore/mlx/blob/d9add9d11f3154111a4c85f267ec2fd307ecd18e/mlx/backend/cpu/quantized.cpp) and the [public operation documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html) identify the operation and its two weight orientations. The test explicitly verifies that both quantized layouts decode to the intended weights before comparing products. Orientation equivalence here is a controlled test on exactly representable values, not a demand that every floating-point reassociation be bitwise identical.

## A small counterexample

Let every input entry be 1 and every weight entry be 6, with inner dimension 1,024 and output width 32. Inputs, decoded weights, individual products and the exact answer are representable in all three tested dtypes.

| Output dtype | Exact answer | Direct CPU path | Transposed CPU control | Dense CPU control |
|---|---:|---:|---:|---:|
| float32 | 6,144 | 6,144 | 6,144 | 6,144 |
| float16 | 6,144 | 6,824 | 6,144 | 6,144 |
| bfloat16 | 6,144 | 2,048 | 6,144 | 6,144 |

These results occur for NVFP4, MXFP4 and MXFP8 in the selected tests. With 512 positive terms followed by 512 negative terms, the exact answer is zero, but the direct bfloat16 path returns −1,704. Alternating signs and a final-sign perturbation are additional controls; the latter has exact result 12.

## Mechanism and causal check

The direct `fp_qmm` implementation uses a `T*` output accumulator and narrows the running sum after each addition. Its tested transposed SIMD path accumulates in float32. An independent model uses exact Python `Fraction` arithmetic and explicit round-to-nearest, ties-to-even after each addition. It reproduces all 241,920 baseline output coordinates. For repeated additions of 6 in bfloat16, the first unchanged running sum occurs at addition 279.

The candidate replaces the direct path's accumulator with one `std::vector<float>` row buffer and converts each output to `T` once, after the inner loop. Individual product expressions remain unchanged. The additional buffer is 4N bytes for output width N; runtime and allocation cost have not been benchmarked.

To distinguish accumulator precision from the introduction of a buffer, a mutation retains the same buffer structure but changes its element type back to `T`. That mutation's CSV is byte-identical to the original. Candidate and mutation are compiled as replacement objects from a copied `quantized.cpp` and linked before the unchanged static MLX archive. Exact commands and hashes are recorded in the evidence package.

## Coverage and results

The grid covers three dtypes, three quantization modes, five inner dimensions (64, 128, 256, 512, 1,024), three exact scales (1/16, 1, 4), four sign patterns, three input shapes (`[1,K]`, `[3,K]`, `[3,1,K]`), and two repetitions. Output width is 32. Every row and column is checked.

This is 1,620 configurations without repetitions, 3,240 executions and 241,920 output-coordinate observations. Counts include the selected shapes and repetitions and are not an estimate of failure frequency in real workloads.

| Variant | Mismatching direct-path coordinates | Transposed/dense control mismatches |
|---|---:|---:|
| Original | 24,192 | 0 |
| Local float32 accumulator | 0 | 0 |
| Accumulator reverted to T | 24,192 | 0 |

The original mismatches comprise 3,024 float16 and 21,168 bfloat16 coordinates, across 216 configurations without repetitions. All 80,640 float32 coordinates remain unchanged after the correction. Repetition and layout controls pass. The machine-readable paired verification is `evidence/paired-accumulation-verification.json`.

## Additional widths and physical layouts

A subsequent native probe completed in the same project checks three dtypes, three modes, K in {64, 256, 1,024}, N in {32, 64, 96, 128, 256}, two sign patterns and six concrete layouts, with three input rows. Layouts include contiguous storage, stepped rows, stepped columns, column-major inputs, broadcast rows and stepped packed weights. The probe checks actual strides after execution, decoded weights and preservation of transformed arrays.

Across 1,620 executions and 559,872 coordinate observations per variant, the original has 116,640 direct-path mismatches (23,328 float16; 93,312 bfloat16), across 450 scenarios. The candidate has zero. The mutation restores 116,640 and again produces a byte-identical baseline CSV. Transposed, dense, float32 and shared-coordinate width/layout controls pass. These counts are separate from the primary grid; the sets can overlap and must not be added as distinct test cases. Evidence: `evidence/stride-verification.json`, `qmm_stride_probe.cpp`, `run_stride_probe.py`, `check_strides.py`.

For these widths the row buffer holds 128–1,024 bytes, excluding allocator overhead. No performance measurement was made. Six concrete layouts do not establish coverage of every possible stride pattern.

## Related work and novelty boundary

Low-precision accumulation is an established numerical issue. MLX already has related work on [CPU reductions (#3909)](https://github.com/ml-explore/mlx/pull/3909), [CPU scans (#3907)](https://github.com/ml-explore/mlx/pull/3907) and [GEMV/GEMM accumulation precision (#1962)](https://github.com/ml-explore/mlx/pull/1962), as well as a [Metal QMM precision discussion (#963)](https://github.com/ml-explore/mlx/issues/963). This report does not claim discovery of the general mechanism.

A bounded search of issues and pull requests did not identify an exact duplicate for the current non-transposed CPU `fp_qmm` path and these controls. Known global-scale TODOs, quantized tails not divisible by 32, previous gather-gradient reports and GPU-specific failures were excluded. Search coverage cannot prove exhaustive novelty.

## Separate observation: FP4 midpoint rounding

Explicit NVFP4 quantization and fused `qqmm` quantization choose differently at certain midpoints. One output is 3 versus 6 across float32, float16 and bfloat16. The separate matrix contains 432 scenarios and 27,648 coordinate comparisons, with 2,592 differences (162 distinct dtype/scale/position combinations out of 1,728). Neighboring values, repetitions and layout controls were checked.

No explicit public midpoint tie-breaking contract was established. This is reported as path inconsistency, not an additional unqualified defect. The accumulator candidate leaves this entire observation byte-identical to the baseline.

## Reproduce and interpret

The frozen evidence archive supplies the pinned source snapshot, native probes, raw CSVs, exact arithmetic model, candidate patch, mutation builder, machine-readable results and an English reproduction guide. No compiled executable is required from the author. The source files retain their original license; see `LICENSES.md`.

The focused native grid and its paired checks were executed. The complete MLX test suite, Python bindings, Metal/CUDA kernels, all possible stride patterns, other SIMD platforms, output widths beyond the selected set and complete models were not tested in this audit. The scalar fallback `fp_qmm_t` and individual product expressions were not corrected. The candidate is a tested local proposal within this scope, not a production-ready or maintainer-approved release.

Research and editorial review: Xamit Kadirbekov / GERO Research. AI-assisted test development, analysis and preparation. Independent work; no affiliation with Apple or the MLX maintainers is implied.

## Publication links

[GERO article](https://www.gero.uz/research/articles/mlx-cpu-fp-qmm-accumulation.html) · [Zenodo evidence](https://zenodo.org/records/22756546) · [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/mlx-cpu-fp-qmm-accumulation.md) · [Hugging Face](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/mlx-cpu-fp-qmm-accumulation.md)

Frozen evidence SHA-256: `6b169485ff661e0bd1a35de3722e81bfcb4d8dbcbe1f6574b9f9281b10d0a142`.
