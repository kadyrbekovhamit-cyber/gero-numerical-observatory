# Small losses and gradients: MLX and nntrainer audit

7 September 2026. Three reproduced implementation cases in two failure families.
These are source-pinned observations and local patch proposals, not maintainer
confirmation, proof of priority, model-training impact or a version regression.

[Technical note](https://www.gero.uz/research/articles/when-small-losses-and-gradients-disappear.html)

| FP32 case | Original | Mathematical target |
|---|---|---|
| MLX BCE, logit 20, label 1 | loss 0 | 2.0611536203e-9 |
| MLX logaddexp(20,b), derivative in b at b=0 | VJP/JVP 0 | 2.0611536182e-9 |
| nntrainer sigmoid CE, logit 20, label 1 | loss 0; gradient 0 | loss 2.0611536203e-9; gradient -2.0611536182e-9 |

The expected values are normal FP32 numbers. Swapping logaddexp arguments
restores the small gradient in the original code, violating derivative exchange
symmetry. Reflecting the MLX BCE logit and binary label restores its small loss.
The two BCE cases share a cancellation family; they are not counted as separate
mathematical discoveries for each input, dtype, shape, or backend.

## Recorded validation

| Suite | Before | After |
|---|---|---|
| MLX BCE, CPU | 23 failed / 8 passed | 31 passed |
| MLX BCE, Metal | Minimal defects separately reproduced | 31 passed |
| MLX C++ logaddexp, CPU | 72 failures / 438 checks | 0 failures / 438 checks |
| nntrainer CPU losses | 4 failed / 73 passed | 77 passed |

BCE references use mpmath at 100 decimal digits on the already rounded input.
Small-tail tests have zero absolute tolerance and per-dtype relative tolerances.
FP16 tail tests stop at logit 8 to keep the expected output normal. Controls cover
weights, soft labels, reductions, label symmetry, batch shapes, the derivative at
zero and the existing upstream BCE test. The C++ checks include VJP/JVP,
argument exchange, cotangent scaling, common-shift direction and infinity
semantics. The nntrainer tests execute the actual layer and compare analytic
derivatives and finite differences; labels remain unchanged.

The 77-test nntrainer run includes five new sigmoid tests, six previously added
softmax tests and 66 upstream loss semantics tests. A fresh checkout with only
this release's test patch has five new tests: four fail before the source patch,
and all five pass after. Previous independent Softplus and softmax patches
remain in the recorded working build, but are excluded from the exported patches.
All new tests use loss_scale=1, so the known scaling issue #4329 is separate.

## Exact scope

- MLX main: `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`.
- nntrainer main: `a7ea056e79ab8e14447ea305c1b634e233343258`.
- Host: Apple M4, macOS 15.5 arm64.
- Python BCE source from main, executed using official `mlx==0.32.2` and
  `mlx-metal==0.32.2` on CPU and Metal.
- Native C++ logaddexp core built from the pinned main source, CPU only.
  The installed Command Line Tools lack `xcrun metal`; its native Metal patch
  was not built or tested. The original defect was reproduced on official Metal.
- nntrainer: real C++ FP32 CPU. FP16, CUDA, Android, NPU, other platforms, full
  repository CI, performance and model-training outcomes were not assessed.

## Minimal installed-runtime reproduction

On an Apple silicon Mac:

```sh
python3 -m venv .venv
.venv/bin/pip install mlx==0.32.2 mlx-metal==0.32.2 numpy==2.2.6 mpmath==1.3.0 pytest==8.4.1
.venv/bin/python reproduce_mlx.py
```

This calls the official installed package, with no language-model API. The
script prints BCE values and logaddexp VJP/JVP in both argument positions.
Some sandboxes prevent Metal initialization even when selecting CPU.

## Run BCE tests before and after

```sh
git clone https://github.com/ml-explore/mlx.git /absolute/path/to/mlx
export AUDIT_MLX_SOURCE=/absolute/path/to/mlx
git -C "$AUDIT_MLX_SOURCE" checkout ce916dbbcaa88e433b6fd1e60a17f766d49c27fe
AUDIT_ORIGINAL=1 AUDIT_DEVICE=cpu .venv/bin/python -m pytest -q test_bce.py
# Expected: 23 failed / 8 passed.
git -C "$AUDIT_MLX_SOURCE" apply /absolute/path/to/this/package/patches/mlx-bce.patch
AUDIT_DEVICE=cpu .venv/bin/python -m pytest -q test_bce.py
AUDIT_DEVICE=gpu .venv/bin/python -m pytest -q test_bce.py
# Expected: 31 passed on each device.
```

The proposed formula separates the linear term by sign and uses log1p for the
small exponential. A sign branch also gives the correct derivative at zero.
The probability-input branch is preserved.

## Run native logaddexp checks

Use the same pinned MLX checkout, before applying its logaddexp patch. CMake
3.25+, Ninja, C++20 and Apple Accelerate are needed. Python bindings are not built.
Configure from this package directory:

```sh
cmake -S native -B build-native -G Ninja \
  -DAUDIT_MLX_SOURCE="$AUDIT_MLX_SOURCE" \
  -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_METAL=OFF -DMLX_BUILD_GGUF=OFF \
  -DMLX_BUILD_SAFETENSORS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --target logaddexp_regression -j 4
build-native/logaddexp_regression cpu
# Expected: 72 failures / 438 checks.
git -C "$AUDIT_MLX_SOURCE" apply /absolute/path/to/this/package/patches/mlx-logaddexp.patch
cmake --build build-native --target logaddexp_regression -j 4
build-native/logaddexp_regression cpu
# Expected: 0 failures / 438 checks.
```

The patch calculates sigmoid(a-b) and sigmoid(b-a) directly. It avoids the
destructive subtraction 1-sigmoid(a-b), at an unmeasured execution cost.

## Reproduce nntrainer

See [NNTRAINER_BUILD.md](NNTRAINER_BUILD.md) for the exact pinned checkout,
Darwin accommodations and build commands. Apply `nntrainer-tests.patch`, build
and run `--gtest_filter='SigmoidLossStability.*'`. Four of five tests fail.
Then apply `nntrainer-bce.patch`, rebuild, and rerun: all five pass.

The patch uses log1p, a sign-dependent linear term and double intermediates,
while retaining FP32 results. Backward calculates the small sigmoid tail
directly. `nntrainer_regression.cpp` also contains the standalone test source.

## Duplicate search and exclusions

`duplicate-search.json` records the primary query date, current main SHAs,
queries and results. Supplemental responses: `logaddexp-search.json`,
`cancellation-search.json`, `nntrainer-bce-search.json`,
`nntrainer-precision-search.json`. All recorded searches include open/closed
issues and PRs; `incomplete_results=false`, fewer than 100 results per query.
[DUPLICATES.md](DUPLICATES.md) classifies the nearest matches.

No exact public match was found within this search. Unindexed, private or
differently named reports may exist. No upstream acceptance is asserted.
QuantizeLinear, previous activation cases and the known loss_scale issue are
excluded. No new Anthropic/Bloom finding is claimed. Preliminary normalization
and cosine observations are omitted from this confirmed-case package.

Local machine path prefixes in exported logs were replaced by `<WORKSPACE>`;
the numerical results and diagnostics were preserved. Exported patches apply to
the unmodified pinned HEAD files (`validation.json`). Verify `SHA256SUMS` for
this public package; its hashes differ from the private working bundle.

Research, tests, local patches and publication preparation used AI assistance.
Patch code is subject to the respective upstream project licenses. Numerical
claims are based on actual recorded execution and independent mathematical checks.
