# MLX arctan: finite weighted derivatives collapse to zero after intermediate overflow

Xamit Kadirbekov · Independent GERO research · 24 September 2026

**A finite `float32` directional derivative of unary `arctan` becomes signed zero in the tested Apple MLX CPU implementation.** At `x = -2^64` and finite directional seed `v = -2^120`, the mathematical action is exactly `-2^-8 = -0.00390625`; MLX returns `-0.0` for both JVP and VJP. A scale-aware research candidate reduces **352 → 0 → 352** discrepancies per layout in the original/candidate/restored comparison.

This is a source-pinned numerical result on synthetic inputs. It is not evidence of an incident in an Apple product, loss to users, or how often a real model reaches this range.

## Formula → changed decision → measured consequence

For real `x`, the directional derivative is

```text
D atan(x)[v] = v / (1 + x²).
```

The tested implementation first forms `1 + x*x`. At `|x| = 2^64`, the square is about `2^128`, outside finite binary32, so the denominator becomes infinity before division. The final mathematical value can nevertheless remain finite because `v` is also large.

A deliberately simple GERO adapter turns the returned derivative into an update and a threshold decision. Starting from `1.0` with learning rate `128`, the independent oracle and candidate produce `1.5`; the original result leaves the value at `1.0`. With a `0.001` magnitude threshold, the oracle and candidate select review while the original skips it. This measured consequence is synthetic demonstration code, not an MLX optimizer, training recipe, customer workflow, or production-impact claim.

| Quantity | Independent oracle | MLX original | Research candidate |
|---|---:|---:|---:|
| JVP/VJP | -0.00390625 | -0.0 | -0.00390625 |
| Synthetic update | 1.5 | 1.0 | 1.5 |
| `abs(g) >= 0.001` | true | false | true |

## Executed implementation and oracle

- Source: [`ml-explore/mlx` at `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`](https://github.com/ml-explore/mlx/tree/59d600b5e64c238427d0f8d897ab7c682ef4d3d2).
- Device and type: complete C++ MLX library, CPU, `float32`, one configured numerical worker.
- Affected method: `ArcTan::jvp` in [`mlx/primitives.cpp`](https://github.com/ml-explore/mlx/blob/59d600b5e64c238427d0f8d897ab7c682ef4d3d2/mlx/primitives.cpp).
- Oracle: `v/(1+x²)` evaluated at 160 decimal digits and rounded to binary32.
- Inputs: 3,918 input/seed rows in each of flat, row and column layouts.
- Layout result: 352 failed rows in each original layout, zero with the candidate, and the same 352 after restoration; no layout disagreements.
- All 352 expected results are finite and nonzero. In 256 rows, input, seed and expected output are normal finite values.
- Forward `arctan` output is bit-identical before and after the candidate on the tested grid. Fifteen NaN/infinity compatibility rows are also bit-identical.

The original and restored CSVs are byte-identical. Restored source SHA-256 is `6a045372433d91ec0ec00caf134ad5010fe34d8c0ae7a114496f632b24f74f09`; candidate source SHA-256 is `201f758a2638ee958617b5cb9620b422d844df775e0c27f906189cd2710fc2e5`; patch SHA-256 is `96320a1240368c70cee43c8c6f1caa3255a9607b9954304fffc352201609cbff`.

## Research candidate and its boundary

The candidate rescales the real finite expression before forming a square and retains the original complex and nonfinite paths. It is a **first-order research candidate**, not a complete universal correction.

For `atan(2^k t)` at `t=1`, reverse-mode controls show:

- at `k=32`, the candidate recovers the first three tested derivatives;
- at `k=64`, it recovers the first and second derivatives, while the third remains `NaN`;
- at `k=80`, first and second derivatives still collapse to zero and the third is `NaN`.

At these composition extremes, an unweighted local derivative can underflow before an outer chain multiplier recovers it. This residual is separate from the verified first-order weighted overflow case and is included so the patch is not presented as complete.

## Release and duplicate review

MLX release `v0.32.2` contains a byte-identical `ArcTan::jvp` method; its packaged runtime was not executed. A fresh 24 September check found no exact unary real-`float32` scale-overflow report in the official MLX issue/PR search or GERO register.

Related prior work is distinct:

- [MLX issue 3778](https://github.com/ml-explore/mlx/issues/3778) and [PR 3779](https://github.com/ml-explore/mlx/pull/3779) concern missing conjugation in complex VJPs.
- [PR 4227](https://github.com/ml-explore/mlx/pull/4227) proposed gradient reference tests on `[-1.5, 1.5]`; it did not cover the binary32 square-overflow range or large finite seeds.
- The prior [GERO arctan2 report](https://www.gero.uz/research/articles/mlx-arctan2-scale-autodiff.html) covers a binary primitive and different code path.
- The prior [GERO rsqrt report](https://www.gero.uz/research/articles/mlx-rsqrt-directional-derivative-scale.html) covers another unary formula.

No absolute priority claim is made.

## Reproduction

Build MLX at the pinned commit with the CPU backend, compile `arctan_probe.cpp` against that build, then run:

```bash
python3 check_arctan.py /path/to/arctan_probe original
```

Apply `candidate.patch`, rebuild, rerun with label `candidate`, restore the source and rerun with label `restored`. `arctan_orders.cpp` records the higher-order composition controls. The archive contains code, exact input bits, raw CSVs, summaries, the patch, duplicate review and checksums.

## Limits

- CPU `float32` only; no GPU run.
- The current-source implementation was executed; `v0.32.2` was compared by source only.
- The configured build did not include the full MLX test suite, so focused primitive/grid controls were used.
- No real model, training job, Apple device behavior, customer loss, or production frequency was measured.
- The synthetic threshold and update illustrate a changed downstream decision; they do not represent a built-in MLX safeguard or optimizer.
- The candidate is incomplete for all higher-order/composition extremes and is not claimed as an accepted upstream fix.

## Disclosure

Independent GERO research by Xamit Kadirbekov. AI-assisted experiment and archival preparation. The measured claims come from the pinned executable implementation, independent high-precision oracle and retained raw results.

<!-- GERO_PUBLICATION_LINKS_BEGIN -->
## Verified publication and reproduction links

- [GERO article](https://www.gero.uz/research/articles/mlx-arctan-directional-derivative-scale.html)
- [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/mlx-arctan-directional-derivative-scale.md)
- [Hugging Face mirror](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/mlx-arctan-directional-derivative-scale.md)
- [Zenodo record and evidence archive](https://zenodo.org/records/22938506) — DOI [`10.5281/zenodo.22938506`](https://doi.org/10.5281/zenodo.22938506)
- [LinkedIn native video post](https://www.linkedin.com/feed/update/urn:li:ugcPost:7508854938483970049/)
- [YouTube Short](https://www.youtube.com/shorts/Hxz1xk6k6P8)
- [Instagram Reel](https://www.instagram.com/gero.math.tech/reel/DdqzgN2h8jK/)
- [Complete reproducibility ZIP](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/raw/refs/heads/main/reports/mlx-arctan-directional-derivative-scale/gero-mlx-arctan-directional-derivative-scale-2026-09-24.zip)

Archive SHA-256: `345f934d734dc65c2d88f17e096164edf136e9a8b94aa2702168bcddc1316ea0`.

Apple Open Source was notified before publication; the final plain-text message was verified in Sent. No acknowledgment or accepted fix is claimed.
<!-- GERO_PUBLICATION_LINKS_END -->
