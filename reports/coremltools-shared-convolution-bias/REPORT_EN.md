## One shared array, two wrong convolution biases

An optimization should preserve the function computed by a model. We reproduced a case where coremltools' convolution-bias fusion silently changes subtraction into addition on two branches.

The result is an implementation defect in the converter's graph pass, not a demonstrated security vulnerability. We submitted a focused fix and regression tests in [Apple coremltools PR #2901](https://github.com/apple/coremltools/pull/2901). The PR is open; acceptance and an official corrected release are not claimed.

## The smallest useful experiment

Build two 1×1 convolutions with 128 output channels. Their weights are respectively 1 and 2. Each branch subtracts the offset `[1, 2, ..., 128]`. The input contains the exact small integers 1, 2, 3 and 4.

For channel 128 at input value 1, the independent calculation is:

- First branch: `1 × 1 − 128 = −127`.
- Second branch: `2 × 1 − 128 = −126`.

With the original default pipeline, those values become 129 and 130. The absolute error is 256 on each branch. This example uses exactly representable values; the discrepancy is not a floating-point rounding tolerance.

| Configuration | Largest absolute error, first output | Second output |
|---|---|---|
| coremltools 9.0, empty pass pipeline | 0 | 0 |
| coremltools 9.0, default pass pipeline | 256 | 256 |
| Current main plus proposed fix, default pipeline | 0 | 0 |

The native execution was on macOS 15.5, using `CPU_ONLY`, `FLOAT32`, and an iOS 17 model deployment target. That target is a model-format setting: it is **not an iPhone test**. The patched main used native runtime libraries from the 9.0 wheel. GPU, ANE, physical iPhone behavior and production models were not tested.

## Why the sign changes

In [the pinned source](https://github.com/apple/coremltools/blob/a207b1a3452eafe02f2d870f486287834af2c82d/coremltools/converters/mil/mil/passes/defs/optimize_conv.py#L748), the pass obtains the constant's NumPy array and squeezes its dimensions. Squeezing can return a view into the original storage. The subsequent in-place negation changes that storage:

```python
bias_value *= -1
```

Another branch reading the same constant now sees the altered value. When the second convolution/subtraction pair is rewritten, it can flip the buffer already used as the first convolution's new bias. Both branches end up adding the original offset.

This sharing can arise naturally: the default pipeline deduplicates equal constants of at least 100 elements. The 128-channel witness triggers that path even when the model was built with two separate equal arrays.

The proposed fix creates a new negated array:

```python
bias_value = -bias_value
```

It preserves the original constant for other consumers. The same root cause also affects the shared transformation path for transposed convolutions.

## What was checked

- Six new regression cases fail against the original main and pass with this isolated fix. They cover ordinary and transposed convolution, existing bias, a second consumer, two subtractions and deduplicated constants.
- All 323 existing tests in `TestConvBiasFusion` pass with the fix.
- A separate native CPU execution agrees with an independent float64 arithmetic reference after the fix, and with an empty pass pipeline before it.
- The checked main commit is `a207b1a3452eafe02f2d870f486287834af2c82d`. The fix commit is linked in PR #2901.

These are bounded regression checks, not proof that all converted models are correct. The full coremltools test suite was not run. All numerical jobs used the existing shared compute lock and ran sequentially without GPU execution.

## Related work and novelty boundary

Public GitHub issues, PR bodies/comments, relevant diffs and this source file's history were checked on 17 October 2026. No exact public duplicate was found within that scope. This is not a claim of worldwide priority or knowledge of Apple's private tracker.

Related contributions remain credited to their authors: [#2776](https://github.com/apple/coremltools/pull/2776) concerns operand order in convolution scale fusion, [#2795](https://github.com/apple/coremltools/pull/2795) addresses mutation in `thresholded_relu`, and [#2805](https://github.com/apple/coremltools/pull/2805) addresses constexpr deduplication equality. Those patches do not fix this subtraction path.

## Evidence and status

The upstream PR contains the patch, six graph tests and an executable runtime example. The accompanying evidence archive contains the small runtime script, observed outputs, regression patch, test summaries, dependency versions and a checksum manifest. It contains no model weights from a customer and no private correspondence.

Investigation and report: **Xamit Kadirbekov and Daniyal Kadirbekov / GERO**. AI assistance was used in investigation, coding and writing. The tests described above were executed locally. The report does not claim independent human review by Apple, measured customer losses, or an accepted fix.
