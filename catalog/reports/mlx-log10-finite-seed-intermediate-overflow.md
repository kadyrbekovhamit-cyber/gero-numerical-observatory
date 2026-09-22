# MLX log10: finite gradients, nonfinite updates under extreme loss scaling

Xamit Kadirbekov · Independent GERO research · 21 September 2026

**A deliberately extreme, but finite, loss scale turned all 65 parameters of a small trained digit classifier nonfinite after one update in the tested MLX CPU implementation.** The resulting logits and probabilities were nonfinite for all 108 held-out images. An experimental change to `log10` differentiation preserved the update: weights and predictions matched the ordinary unscaled control exactly in each of three trials. Restoring the original implementation restored the failure.

This is a measured consequence in a GERO-authored experimental classifier using real MLX C++ operations and public handwritten-digit data. It is **not** evidence of an incident in a deployed Apple product, customer losses, or how often training reaches this condition. A finite-gradient guard prevented corruption by skipping the update. The candidate correction remains incomplete on other boundary inputs.

## Implementation and mathematical discrepancy

Executed source: [`ml-explore/mlx` at `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`](https://github.com/ml-explore/mlx/tree/59d600b5e64c238427d0f8d897ab7c682ef4d3d2), complete C++ CPU library, float32. The affected methods are `Log::jvp` and `Log::vjp` in [`mlx/primitives.cpp`](https://github.com/ml-explore/mlx/blob/59d600b5e64c238427d0f8d897ab7c682ef4d3d2/mlx/primitives.cpp).

For a positive finite input `x` and finite tangent/cotangent `v`, the base-ten logarithm derivative action is `v / (x ln(10))`. The implementation first forms `v/x`, then multiplies by approximately `0.43429448`. The first operation can overflow even when the final mathematical answer fits float32.

At `x = float32(0.5)`, `v = float32(3.2e38)`, both native JVP and VJP produce infinity. Independent 80- and 120-decimal-digit evaluation, rounded directly to IEEE binary32 with integer ties-to-even arithmetic, gives **`2.779484577684454e38`**, a finite result. The input decimal denotes its actual float32 representation throughout, not exact decimal `3.2e38`.

The current main pin was checked on 21 September. Release `v0.32.2` has matching affected method bodies, verified by source comparison. **The installed release binary was not executed:** its import failed before evaluation with a Metal-device initialization error. No GPU or system-setting workaround was attempted.

## Measured chain: derivative → parameter update → predictions

We trained a 65-parameter binary logistic classifier to distinguish digits 3 and 8. Its operations—matrix multiplication, sigmoid, `log10`, autodifferentiation, gradient unscaling, and parameter subtraction—execute in MLX C++ on CPU. The training driver, plain SGD update, nonfinite guard and invalid-prediction policy are GERO demonstration code, not claims about MLX's built-in optimizer or safeguards.

The data comprise 357 images from scikit-learn's `load_digits`: 249 training and 108 held-out examples. Pixel values are divided by 16 and a constant bias feature is added. The split, seed `20260921`, 60 full-batch pretraining steps, learning rates, selection rule and metrics were recorded before training in `EXPERIMENT_PLAN.json`. After pretraining from zero weights, the frozen checkpoint correctly classified **103/108 images (95.37%)**, with all parameters and outputs finite.

We selected one training example in each predeclared true-label-probability interval: `[0.5,0.65)`, `[0.65,0.8)`, `[0.8,0.9]`. The tie-break was the smallest original dataset row ID. The selected IDs were 378, 439 and 18, with probabilities approximately 0.62547, 0.72568 and 0.82256. All three happen to have label 8; the model intervention does not test both label signs. Selection used no held-out outcome.

Each trial begins independently from the same frozen checkpoint and performs **one** update. The loss is `L = -lambda * log10(sigmoid(y * (w·x)))`, where `y` is +1 for digit 8 and −1 for digit 3. We divide the computed gradients by `lambda` **before** applying SGD at ordinary float32 learning rate `0.05`. Tested scales are float32 `1`, `1e20` and `3.2e38`; there is no momentum, weight decay or clipping. The scale `3.2e38` is an injected extreme stress condition, not a normal recommended setting.

| One-step branch | Finite parameters | Valid predictions | Correct held-out classifications |
| --- | ---: | ---: | --- |
| Frozen checkpoint, before intervention | 65/65 | 108/108 | 103/108 |
| Original implementation, scale 1 or 1e20 | 65/65 | 108/108 | 104, 104, 103 out of 108 in the three trials |
| Original implementation, extreme scale, unguarded | 0/65 | 0/108 | Accuracy undefined; all logits and probabilities nonfinite |
| Experimental candidate, extreme scale, unguarded | 65/65 | 108/108 | 104, 104, 103 out of 108; identical weights and predictions to scale-1 controls |
| Original implementation, extreme scale, guarded | 65/65 | 108/108 | 103/108; update skipped, checkpoint byte-identical |
| Restored original implementation | Same as original | Same as original | Original results reproduced exactly |

The three trials reuse the same held-out set; they are **not** 324 independent test images or three complete retrainings. Candidate updates changed 37, 39 and 31 parameters, respectively, so preserved predictions are not explained by a zero update. At scales 1 and 1e20 the candidate leaves saved gradients, weights and predictions byte-identical to the original. At the extreme scale the candidate weights and predictions are byte-identical to the corresponding ordinary scale-1 step. The largest absolute discrepancy from the independent binary64 analytic gradient/update references is below `1e-7`.

The scaled forward losses remain finite. For these inputs, `|dL/dq|` is below `2.78e38` and every true scaled parameter-gradient component is below `6.95e37`, both within float32 range. Therefore, these failures are not an unavoidable overflow of the final derivative. A separately executed native `log10` VJP with the singleton backward seed `-lambda` reproduces the intermediate-overflow diagnosis; it is a diagnostic evaluation, not instrumentation of an internal backward tensor.

This establishes two bounded consequences: an unguarded update can destroy a previously usable checkpoint, whereas a finite-gradient guard can preserve that checkpoint while foregoing an otherwise mathematically finite step. We did not measure long-run training cost, training frequency, model quality in normal workloads, or production harm.

## Primitive grid, correction scope and restoration

The packet retains 294 distinct input/seed pairs, including both signs and final-overflow boundaries, exercised as a vector, row and column through both JVP and VJP. The same 66 inputs have finite references but nonfinite original outputs in each direction and layout. They are one implementation case, not 396 distinct defects.

The experimental float32 base-ten fallback scales the seed first when the initial quotient is infinite and `x` is nonzero. On this grid, finite-to-nonfinite failures change **66 → 0 → 66** for original → candidate → restoration. However, the candidate introduces **four different final-overflow-boundary mismatches**: finite values where correct final rounding is infinite. Of the other 228 controls, 224 retain identical derivative bits. Forward bits are unchanged. The patch is a causal experiment and a candidate starting point, **not a complete or production-ready fix**. Performance and a full upstream regression suite were not assessed.

The complete native `primitives.cpp` translation unit was compiled for the candidate and for restoration, then linked ahead of the unchanged complete MLX static CPU library. The original executable uses the unmodified library. Build/source/object hashes and commands are retained. No replacement mathematical mock was substituted for MLX.

## Prior-art review and disclosure status

A bounded review of current main, releases, GitHub issue/PR searches for `log10`, `Log::jvp` and logarithm overflow, plausible bodies and diffs (including [PR 3605](https://github.com/ml-explore/mlx/pull/3605) and [PR 4266](https://github.com/ml-explore/mlx/pull/4266)), and the GERO catalog did not identify an exact earlier report or proposed fix for this finite-seed base-ten overflow. Complex-gradient conjugation and compilation equivalence are distinct. This is not an assertion of exhaustive search or absolute priority. Search evidence is archived; recheck before first external distribution.

This case has **not** been sent to MLX maintainers. Their current issue template prohibits AI-written issues. The owner authorized independent publication on 21 September 2026 before a personally authored upstream notice. This AI-assisted research packet is not a maintainer submission or an accepted correction.

Publication note, 21 September 2026: the immutable evidence ZIP was frozen before that publication-order decision and retains its preparation-time status. The current loose report updates disclosure status only; numerical evidence and the ZIP hash are unchanged.

## Reproduction and data rights

The packet includes exact split inputs, float32 bit patterns, the frozen checkpoint, all trial outputs, independent references, the experimental diff, native drivers, a portable replay runner, source archive and checksums. `README.md` explains a clean CPU build and replay. Historical acquisition scripts retain original paths for provenance; `code/replay.py` accepts local paths and does not require scikit-learn. Separate replay receipts identify exactly what was freshly compiled versus reused.

Dataset: Alpaydin, E. & Kaynak, C. (1998), *Optical Recognition of Handwritten Digits*, UCI Machine Learning Repository, [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The scikit-learn copy contains 1,797 images from the original UCI test subset. Our digit selection, normalization and new experimental split are transformations; this is **not the original writer-independent UCI benchmark**. Original licenses and attribution remain with MLX and included dependency headers. No endorsement by Apple, MLX, UCI or scikit-learn is implied.

Research and editorial responsibility: Xamit Kadirbekov / GERO. AI assistance was used for research orchestration, code, checking and writing. The evidence supports the stated experiments and their limits; it does not establish that every possible failure has been found.

## Public records and delivery status

- [github](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/mlx-log10-finite-seed-intermediate-overflow.md)
- [huggingface](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/mlx-log10-finite-seed-intermediate-overflow.md)
- [gero](https://www.gero.uz/research/articles/mlx-log10-finite-seed-intermediate-overflow.html)

[Complete evidence ZIP](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/raw/refs/heads/main/reports/mlx-log10-finite-seed-intermediate-overflow/gero-mlx-log10-impact-evidence-2026-09-21.zip). SHA-256: `991a532b44d8806f237691089738daca1b51f52cc9257bc3a384cb90df38ae01`.

Pending distribution: zenodo, linkedin, youtube. Upstream notice has not been sent; independent publication was authorized before a personally authored notice. The experimental candidate is incomplete; no upstream acceptance is claimed.
