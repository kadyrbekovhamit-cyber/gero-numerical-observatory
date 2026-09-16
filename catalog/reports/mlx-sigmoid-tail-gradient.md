# Apple MLX: sigmoid derivatives lose a representable positive tail

Independent numerical audit by Xamit Kadirbekov / GERO Research. Evidence dated 16 September 2026. Version 1.0.0.

For `f(x) = 100000000 * sigmoid(x)` at `x = 20`, the analytic derivative is approximately **0.206115361394185**. The tested native MLX CPU path returns **0** in both JVP and VJP. The candidate returns **0.20611535012722015** in float32.

The relevant `Sigmoid::vjp` and `Sigmoid::jvp` source sections are identical in release v0.32.2 and current main commit `7241f12e631440387a2156ec1018d1c9fe8e56b9`. The native experiment links release-matched `primitives.cpp` to an existing CPU-only MLX archive. It is not a new full build of current main, a Metal result, or an installed-wheel Python result.

## Cause and correction

The old derivative forms `s * (1 - s)` from the rounded forward output `s`. For sufficiently positive finite inputs, `s` rounds to one even though the derivative remains representable. The subtraction destroys the small complementary factor.

The candidate forms `sigmoid(x) * sigmoid(-x)` instead. The forward output is unchanged. This preserves the small tail without constructing it by subtraction from one. The smooth expression also avoids an artificial absolute-value branch at zero in higher derivatives.

The reference computes `e = exp(-abs(x))` and `sigmoid'(x) = e / (1 + e)^2` independently in host floating-point arithmetic. Second and third derivatives use their analytic formulas. Inputs are finite and quantized to the tested dtype before evaluating the reference.

## Verification and a separate float64 dependency

There are **284 native checks** across float16, bfloat16, float32 and float64, including JVP/VJP, zero and scaled incoming gradients, both signs of the input, and second/third derivatives for float32/float64.

| Native configuration | Failing checks |
|---|---:|
| Released derivative; existing CPU exp approximation | 106 / 284 |
| Sigmoid derivative correction alone | 56 / 284 |
| Known float64-exp correction only | 66 / 284 |
| Both corrections | **0 / 284** |

The 56 remaining failures with the derivative-only patch are float64 accuracy cases. The CPU's existing SIMD `exp` implementation narrows double inputs to float32. This is already described in [MLX issue #3047](https://github.com/ml-explore/mlx/issues/3047) and in our earlier local float64-exp investigation; it is **not a new finding**. The prior simple double-precision exp correction is applied symmetrically to the before/after control to isolate the sigmoid defect. Its effect is limited to the rebuilt unary translation unit in this experiment.

For float16, bfloat16 and float32 alone, the new derivative patch changes **48 failures out of 196 checks to zero**, without the exp control. These findings are not dependent on float64 support.

All intermediate outcomes are retained, including the incomplete derivative-only result. This avoids presenting a combined patch as if it were a self-sufficient float64 correction.

Files:

- [Derivative correction](mlx-sigmoid-tail-gradient.patch).
- [Known exp control](known-float64-exp-control.patch), not claimed as new.
- [Native regression](mlx_sigmoid.cpp) and [sequential build runner](build_mlx_sigmoid.py).
- `evidence/mlx-run-before.json`, `mlx-run-after.json`, `mlx-run-exp-control-before.json`, `mlx-run-exp-control-after.json`.
- `evidence/mlx-build-results.json`: all compile, link and run commands, return codes and CPU times. The complete recorded C++ work took about 6.14 CPU seconds, sequentially, with each subprocess capped at 30 CPU seconds.

Reproduction from the project root:

```sh
MLX_AUDIT_BUILD=/path/to/build-native MLX_AUDIT_REPO=/path/to/mlx python3 -B build_mlx_sigmoid.py
```

It requires the existing CPU archive and original build configuration specified by MLX_AUDIT_BUILD and MLX_AUDIT_REPO. The build directory must contain compile_commands.json and mlx-build/libmlx.a; its compile commands must refer to the matching checkout. A full clean build is outside the recorded experiment. The shared MLX checkout and its pre-existing uncommitted changes are not modified.

## Duplicate screen and limits

Two GitHub searches each returned 34 records. The nearest reports #2659, #2666, #4227 and #4461 were examined with available comments. They concern forward accuracy, proposed forward literal changes, ordinary-range reference tests, or eager/compiled Metal disagreement. #2252 concerns float64 activation compilation errors. None establishes the demonstrated positive-tail sigmoid derivative failure in the scope inspected. #3047 is the known, separate float64 exp issue.

An earlier local `logaddexp` report concerns a related loss of a complementary sigmoid factor. This new packet audits the sigmoid primitive itself and its higher derivatives; it is not a claim to a new mathematical identity or a wholly new class of rounding error. Exact queries and public-source receipts are preserved in `evidence/duplicate-screen.json`.

Sources: [release](https://github.com/ml-explore/mlx/releases/tag/v0.32.2), [pinned current derivative source](https://github.com/ml-explore/mlx/blob/7241f12e631440387a2156ec1018d1c9fe8e56b9/mlx/primitives.cpp).

GPU, compiled/fused paths, throughput, large-model training and general tensor-shape coverage were not tested. No production impact or speed claim is made. An upstream submission has not been created.


## Reuse and attribution

Report and original audit code: Xamit Kadirbekov / GERO Research, 2026. Original audit contributions are licensed under MIT; upstream MLX notices are retained in LICENSE. Public issue-search material is included for attribution and duplicate screening, not as a claim of authorship. No company affiliation or maintainer acceptance is implied.
