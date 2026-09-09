# Reproduce

The measured environment reused the existing nntrainer `build-audit`
directory with CPU FP32, `nntr-num-threads=1`, `thread-backend=none`,
`enable-blas=false`, macOS 15.5 arm64 and Apple clang 17.
No new full checkout build or cross-platform CI run was performed.

In an isolated checkout of revision
`a7ea056e79ab8e14447ea305c1b634e233343258`, configure the project's layer
unit tests and apply only `patches/tests.patch` first. The complete
Darwin setup recipe and compatibility header are also recorded in the
sibling [pooling audit's BUILD.md](../nntrainer-average-pool-padding-2026-09-09/BUILD.md).
Use the Dropout patches and driver here in place of that audit's patches
and driver. The setup instructions are a reproduction recipe, not a claim
that a fresh build was performed.

Set these variables to the isolated checkout and this evidence directory:

```sh
export DROPOUT_REPO=/absolute/path/to/nntrainer
export DROPOUT_AUDIT=/absolute/path/to/nntrainer-dropout-routing-2026-09-09
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
git -C "$DROPOUT_REPO" apply "$DROPOUT_AUDIT/patches/tests.patch"
ninja -C "$DROPOUT_REPO/build-audit" -j1 test/unittest/layers/unittest_layers
python3 "$DROPOUT_AUDIT/reproduce.py" \
  --binary "$DROPOUT_REPO/build-audit/test/unittest/layers/unittest_layers" \
  --output "$DROPOUT_AUDIT/reproduced-original" --expect original
git -C "$DROPOUT_REPO" apply "$DROPOUT_AUDIT/patches/source.patch"
ninja -C "$DROPOUT_REPO/build-audit" -j1 test/unittest/layers/unittest_layers
python3 "$DROPOUT_AUDIT/reproduce.py" \
  --binary "$DROPOUT_REPO/build-audit/test/unittest/layers/unittest_layers" \
  --output "$DROPOUT_AUDIT/reproduced-patched" --expect patched
```

The driver succeeds for the original only if exactly the three expected
new tests fail and the single-input control passes. For the patched
version it requires all four tests to pass. A crash, missing tests,
unexpected failure or timeout is not classified as a reproduction.

To include the existing 15 semantics/golden tests, run the same binary
from `build-audit` with `--gtest_filter='DropoutRoutingAudit.*:Dropout/*'`.
That measured suite gave 19 tests / 3 failures before and 19 / 0 after.
The existing 20%-dropout training golden test is stochastic; the four
new tests selected by the driver are deterministic.

