# SplitLayer leaves later channels unwritten during full-prefill incremental execution

Xamit Kadirbekov · GERO Research · 17 September 2026

This is an archival report of an nntrainer defect measured and reported to the maintainer on 15 September. It is not a claim of a new discovery on the archive date.

For a valid FP32 NCHW tensor with shape `[1,2,1,2]` and values `[1,2,3,4]`, a width split into two outputs should produce `[1,3]` and `[2,4]`. The ordinary C++ `SplitLayer::forwarding` method does exactly that. Calling `incremental_forwarding(context,0,1,false)` over the complete height writes only the first channel of each output.

The experiment initializes every output coordinate to `-99999` before the incremental call. The results are `[1,-99999]` and `[2,-99999]`. That deliberately chosen sentinel demonstrates an unwritten coordinate. It is not a claim that an application ordinarily returns this particular number.

## Implementation and contract

The executed source is nntrainer commit [`a7ea056e79ab8e14447ea305c1b634e233343258`](https://github.com/nntrainer/nntrainer/blob/a7ea056e79ab8e14447ea305c1b634e233343258/nntrainer/layers/split_layer.cpp). The layer's finalization accepts these dimensions and the requested split. Its documented dimension mapping for `axis=3` is `[B,C,H,W]` to outputs `[B,C,H,W/parts]`; it does not restrict `C` to one.

The invariant is exact: a complete incremental prefill and ordinary forwarding must place the same input coordinate at each output coordinate. Splitting is a data-routing operation. With the exact representable test inputs used here, no floating-point tolerance is needed.

For width splitting, the incremental implementation loops over batches, steps and split parts, then copies between addresses containing a literal channel index of zero:

```cpp
const float *src = input_.getAddress(b, 0, s, idx * split_w);
float *dst = output_.getAddress(b, 0, s, 0);
std::memcpy(dst, src, split_w * sizeof(float));
```

It never visits subsequent channels. Other split axes use the ordinary-forward fallback. The candidate in `candidate.patch` adds a channel loop and uses that channel index in both addresses. This proposal addresses the measured FP32 missing-channel defect only; it is not a general repair of every incremental or dtype behavior in this layer.

## Independent check and results

The driver `probe.cpp` constructs the actual C++ layer and tensor context. It exercises ordinary and full-prefill incremental forwarding over 576 distinct combinations of batches, channels, heights, widths, valid split axes, split counts and two input patterns. The patterns are consecutive integers and alternating signed quarter-integers. Every expected output is independently derived by decoding a flat output index into coordinates, mapping the split coordinate back to the original input, and evaluating the input pattern with Python integers and `Fraction`.

The oracle does not call nntrainer. The ordinary-forward result is an additional comparison, not the definition of the expected answer.

| Variant | Incremental mismatches / 19,296 coordinates | Affected scenarios / 576 | Ordinary-forward mismatches |
|---|---:|---:|---:|
| Original C++ source | 4,320 | 168 | 0 |
| Local channel-loop candidate | 0 | 0 | 0 |
| Restored original translation unit | 4,320 | 168 | 0 |

These counts describe overlapping checks of **one implementation defect**, not thousands of separate bugs. All 12,096 predeclared control coordinates—one-channel inputs or non-width split axes—remain unchanged. All 14,976 previously correct incremental coordinates remain unchanged. All ordinary-forward outputs, input values and recorded shapes remain unchanged. Original and restored raw TSV files are byte-identical.

The authoritative original experiment built a fresh original native core with one worker and verified 603 required original source/header files. An initial probe had linked a previous audit's core containing an unrelated DivideLayer correction. That limitation was resolved before the maintainer report: the clean-core original, candidate and restored results match the initial raw results byte for byte. The `clean-*` evidence is the primary historical record.

## Standalone reproduction

The archive includes the compact pinned source tree, its source manifest, the original googletest and iniparser vendor archives, the C++ driver, candidate patch, independent oracle, and frozen raw outputs. Large application assets and application-only links omitted from the native build are listed in the provenance. This is not described as a complete repository backup.

`reproduce.py` creates a new working directory, verifies source/archive hashes, applies and reverses the candidate patch to check its exact bytes, builds the original core, compiles and executes three real C++ variants, independently checks each output coordinate, and compares the new TSV hashes with the frozen experiment. The source is checked again after execution. It does not overwrite the package or download dependencies.

The recorded build adapter is for **macOS arm64**, with Python 3.9 or newer, a C++17 compiler, Meson and Ninja already installed. It uses an external Darwin compatibility header and a generated build-directory `malloc.h`, retaining upstream source bytes. Use:

```sh
python3 -B reproduce.py --output /absolute/path/to/new-replay
```

If existing build tools are outside `PATH`, add `--tools-bin /absolute/path/to/tool-bin`. The output directory must not already exist. The runner uses `ninja -j1`, disables dependency downloads, and configures numerical thread counts to one. See `REPRODUCE.md` and the final replay receipt for exact tool versions and commands.

## Fresh standalone replay on 17 September

The standalone runner completed a new original-core build and all three C++ variants. All three raw TSVs reproduce the frozen experiment byte for byte: **4,320 → 0 → 4,320** incremental coordinate mismatches, with all controls preserved. A total of 2302 included manifest-matching source files, including all 603 required native source/header files, were checked before and after execution. The patch application/reversal produced exact expected source bytes. See `evidence/PORTABLE_REPLAY_RECEIPT.json` for tool versions, commands and verification. This confirms portability of the packaged workflow within the stated Mac environment; it is a replay of the same case.

## Source refresh and prior reports

At the 17 September source refresh, current main was `2d1e4974ae84e1e6d37784416bdfb52b647919a6`; the target file remained byte-identical to the executed source. This is a target-source comparison, not execution of the entire newer repository. The latest recorded release, `v0.5.0`, predates the incremental feature. **No released-version defect is claimed.**

The exact existing report is our own [issue #4337](https://github.com/nntrainer/nntrainer/issues/4337). It was open with no maintainer replies at refresh. No duplicate issue has been created and no upstream acceptance is claimed.

The review covered 161 distinct issue/PR title-and-body records from three focused searches, the latest 100 updated records, target-file history, and the 102-entry canonical GERO catalog. No additional exact missing-channel report was found in that bounded review. Search absence is not a universal novelty guarantee.

[Issue #4334](https://github.com/nntrainer/nntrainer/issues/4334), item M2, concerns invalid incremental ranges exceeding tensor extents. [Issue #4335](https://github.com/nntrainer/nntrainer/issues/4335), item F7, concerns FP16 byte-size misuse. Their cited source overlaps this report, but their triggers and defects differ; neither is presented here as a new finding. The introduction is in merged [PR #3998](https://github.com/nntrainer/nntrainer/pull/3998). PRs #4054 and #4067 proposed broader incremental API consolidation but were closed without merging at review. Their titles do not establish an accepted fix. Details and retained responses are in `review/DUPLICATE_REVIEW.md`.

## Limits and disclosure

The measured scope is FP32, NCHW, contiguous synthetic tensors and a full prefill `[0,height)`. No claim is made about arbitrary decode intervals, FP16, GPU execution, gradients, model predictions, performance, the full upstream test suite, deployed Samsung devices or user losses. The test directly establishes incorrect tensor routing for the stated inputs.

Independent GERO research by Xamit Kadirbekov. Investigation and preparation were AI-assisted. No private model or customer data is used. Upstream code, licenses and attribution are retained. The correction remains a local candidate.

## Evidence archive

[Download the frozen standalone reproduction ZIP](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/raw/refs/heads/main/catalog/evidence/nntrainer-split-incremental-channels/gero-nntrainer-split-incremental-channels-research-2026-09-17.zip). SHA256: `072dbfd6967d765fc7091a30ab1de80c9d9bb9609e155cdbf4750e71557685be`. Unzip and run the included reproduction guide. The ZIP remains immutable.
