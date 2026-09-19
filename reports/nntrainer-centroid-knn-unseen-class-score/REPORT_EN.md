# nntrainer CentroidKNN chooses an unobserved class over an exact centroid match

Independent GERO numerical audit by Xamit Kadirbekov, 20 September 2026. Native implementation discrepancy confirmed; one-line local candidate tested. AI-assisted research and preparation. [Report submitted to nntrainer](https://github.com/nntrainer/nntrainer/issues/4341); no maintainer acceptance or upstream fix is claimed.

## Concrete result

Configure three possible classes, but supply training examples for only two. In the tested public nntrainer C++ model, **the class with no training examples wins**, even when the query exactly matches a trained class.

The synthetic training data are class 0 at `[0,0]` and class 1 at `[3,4]`; class 2 has no samples. The model successfully compiles, initializes, trains and runs inference. Its score vector for the query `[0,0]` is:

```text
Class 0, exact match:       0
Class 1, distance five:    -5
Class 2, no samples:      +1.17549435e-38
```

Selecting the largest score chooses class 2. The correct nearest observed class is 0. Querying `[3,4]` similarly returns `[-5,0,+1.17549435e-38]`, again choosing the unobserved class instead of class 1.

These are ordinary finite feature values. No extreme input or rounding-boundary construction is needed.

## Why the score ordering breaks

For an observed class, the implementation computes the negative Euclidean distance to its arithmetic centroid:

```text
score(class) = -sqrt(sum((query - centroid(class))**2))
```

Every finite score is nonpositive. The explicit branch for a class with zero samples instead uses:

```cpp
std::numeric_limits<float>::min()
```

For a floating-point type, `min()` is the smallest positive normal value. It is greater than every nonpositive distance score, including an exact match at zero. As long as at least one class is unobserved and the compared distances are finite, this score ordering promotes an unobserved class.

[The pinned implementation and zero-sample branch](https://github.com/nntrainer/nntrainer/blob/2d1e4974ae84e1e6d37784416bdfb52b647919a6/nntrainer/layers/centroid_knn.cpp#L118) are the source of the discrepancy. This report counts one implementation defect, not one defect for each failing test scenario.

## Measured decision and consequence

The experiment uses actual public APIs: `createModel`, `layer::CentroidKNN`, a generator dataset, `compile`, `initialize`, `train`, and `inference`. The native model produces the scores. A small GERO adapter applies `std::max_element` to select the class. A separate native component test uses nntrainer's own `Tensor::argmax` and reaches the same result.

The executed chain is:

**positive zero-sample score → unobserved class becomes the largest score → both exact-match queries receive the wrong class label.**

| Model training set | Queries | Original wrong labels | Local candidate | Restored original |
|---|---:|---:|---:|---:|
| Two observed classes out of three | 2 | 2 | 0 | 2 |
| All three classes observed | 3 | 0 | 0 | 0 |

The fully trained control adds class 2 at `[6,8]`. Its three score vectors and selected labels are unchanged by the candidate.

This establishes a changed classification decision in a small executable model built with the real library. It does not establish how often deployed applications leave a class unobserved. An application that always has samples for every configured class does not enter this zero-sample branch. The audit did not execute an image feature extractor, a real-image benchmark, a Samsung device or a customer workflow. No product-wide accuracy loss or monetary consequence is measured.

## Candidate and restoration controls

The local candidate replaces only the zero-sample sentinel:

```diff
- hidden_.setValue(b, 0, 0, i, std::numeric_limits<float>::min());
+ hidden_.setValue(b, 0, 0, i, std::numeric_limits<float>::lowest());
```

For the tested finite distance range, the most negative finite float is below all eligible observed-class scores. The public-model outputs become `[0,-5,-3.40282347e38]` and `[-5,0,-3.40282347e38]`, selecting classes 0 and 1. Restoring the original source reproduces both errors.

A separate component grid covers 1,728 inference scenarios: 2, 3 or 5 configured classes; feature widths 1, 2 or 4; every nonempty subset of observed classes; one or two samples per observed class; and modest finite query offsets. The two-sample construction tests centroid averaging with symmetric dyadic inputs. An independent Python Decimal oracle at 80 decimal digits calculates the expected arithmetic centroids and Euclidean distances.

| Scenario group | Count | Original mismatches | Candidate | Restored |
|---|---:|---:|---:|---:|
| At least one unobserved class | 1,548 | 1,548 | 0 | 1,548 |
| Every class observed | 180 | 0 | 0 | 0 |
| Total | 1,728 | 1,548 | 0 | 1,548 |

Observed-class scores, centroids and sample counts are unchanged. Inference leaves the tested weights untouched. Original and restored component JSONL files are byte-identical. The grid is synthetic coverage of the stated mechanism, not a prevalence or real-model accuracy estimate.

For the public-model A/B/A check, copies of the complete native core were relinked with exactly one newly compiled centroid object replaced. The other 194 core objects remained unchanged. Runtime loader output verifies the intended library in every run. The five official C++ API source files are unchanged. This avoids inferring a model-level correction merely from a separately compiled layer function.

## Versions, execution and reproduction

- Executed source: nntrainer main `2d1e4974ae84e1e6d37784416bdfb52b647919a6`, refreshed before disclosure.
- Latest reviewed release source: [v0.5.0](https://github.com/nntrainer/nntrainer/blob/v0.5.0/nntrainer/layers/centroid_knn.cpp). It contains the same sentinel. Its released binary was not executed.
- Host: macOS 15.5 (24F74), ARM64, Apple clang 17.0.0. CPU FP32 NCHW, batch size 1, BLAS/thread backend disabled, one configured worker. No GPU, paid compute or audio/video playback.
- The archive contains a compact source snapshot with 2,005 files checked against the pinned Git blobs, original dependency source archives and licenses, a build-only Darwin compatibility header, C++ probes, the independent oracle, raw outputs, candidate patch and verification receipts.

<!-- CLEAN_REPLAY_STATUS -->
A separate clean full-source CPU replay completed at `2026-09-19T20:19:25.310269+00:00`. It reproduced all three 1,728-row component outputs byte for byte and all five public-model score/decision rows for each original/candidate/restored variant. The independent Decimal oracle was recomputed against these new outputs. All archived source files remained unchanged. See `REPLAY_RECEIPT.json`.
<!-- CLEAN_REPLAY_STATUS_END -->

The portable wrapper targets the recorded macOS ARM64 environment. With Python 3.9+, a C++17 compiler, Meson and Ninja available, run `python3 /path/to/package/replay.py` from an empty working directory. It uses only bundled source, builds with `ninja -j1`, reproduces all three native variants, compares the component outputs byte for byte, compares public-model score/decision rows, and recomputes the independent oracle. Other operating systems require the project's normal build configuration; their execution is not claimed here.

## Duplicate review and disclosure

The bounded review covered current source, the current-path history, the canonical GERO catalog and relevant open/closed issue and PR bodies, comments and actual diffs. Relevant PRs included #859, #1394, #1499, #1580, #1585 and #1702. Eleven source-history commits and the initial pre-move implementation were examined. No exact earlier report or proposed correction for this positive zero-sample score was found in the reviewed material.

[Historical issue #1521](https://github.com/nntrainer/nntrainer/issues/1521) mentions an accuracy-drop cause without describing it or linking a fix. It is not possible to establish from that public record whether it concerned the same cause. This ambiguity is retained in the new report; no exhaustive novelty guarantee is made.

[Vendor issue #4341](https://github.com/nntrainer/nntrainer/issues/4341) was submitted and its body verified before independent distribution. It includes the complete public-model reproducer and discloses AI assistance. Submission does not imply acknowledgment, acceptance or an upstream correction.

## Limits

The tested policy is nearest-centroid selection over at least one observed class. An entirely untrained model requires an explicit policy; this audit does not resolve it. Rejection of partially trained models would be an alternative API policy to a sentinel-based mask.

The one-line candidate is validated for the stated finite input/distance range. Extreme or nonfinite distances, ties at the most negative float, other dtypes, larger batches and full application integrations are outside scope. No complete upstream suite or cross-platform correctness claim is made. The primary evidence is a score-ordering violation and its measured synthetic classification consequence.
