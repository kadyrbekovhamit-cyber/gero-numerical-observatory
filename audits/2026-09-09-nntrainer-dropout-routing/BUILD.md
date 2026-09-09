# Reproduce the Dropout routing regression

Source pin: `a7ea056e79ab8e14447ea305c1b634e233343258`.
Measured environment: macOS 15.5 arm64, Apple clang 17, Meson 1.12.0,
Ninja 1.13.2; CPU FP32, one worker, no BLAS/thread backend.

These commands describe a fresh reproduction recipe. The measured audit
reused an existing configured build with unrelated earlier repairs; a new
full checkout build and platform matrix were not run. The retained Dropout
baseline exactly matches the pin, and only Dropout patches are supplied.

Set `DROPOUT_AUDIT` to this directory, containing this BUILD.md:

```sh
export DROPOUT_AUDIT=/absolute/path/to/2026-09-09-nntrainer-dropout-routing
export DROPOUT_REPO=/absolute/path/to/isolated/nntrainer
git clone https://github.com/nntrainer/nntrainer.git "$DROPOUT_REPO"
git -C "$DROPOUT_REPO" checkout a7ea056e79ab8e14447ea305c1b634e233343258
git -C "$DROPOUT_REPO" submodule update --init --depth 1 \
  subprojects/googletest subprojects/iniparser
git -C "$DROPOUT_REPO" apply "$DROPOUT_AUDIT/evidence/patches/tests.patch"
cd "$DROPOUT_REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
meson setup build-audit \
  -Denable-blas=false -Denable-app=false -Denable-ccapi=true \
  -Denable-capi=disabled -Dml-api-support=disabled \
  -Denable-tflite-backbone=false -Denable-tflite-interpreter=false \
  -Denable-nnstreamer-tensor-filter=disabled \
  -Denable-nnstreamer-tensor-trainer=disabled \
  -Denable-test=true -Dnntr-num-threads=1 -Dwerror=false -Dthread-backend=none \
  "-Dcpp_args=['-include','$DROPOUT_AUDIT/nntrainer-darwin-build.h']"
```

On macOS, create `build-audit/nntrainer/malloc.h` containing
`#include <malloc/malloc.h>`. The forced include and shim are Darwin build
compatibility only. Omit those compatibility steps on non-Darwin systems;
those platforms were not tested here.

```sh
ninja -C build-audit -j1 test/unittest/layers/unittest_layers
python3 "$DROPOUT_AUDIT/evidence/reproduce.py" \
  --binary "$DROPOUT_REPO/build-audit/test/unittest/layers/unittest_layers" \
  --output "$DROPOUT_AUDIT/reproduced-original" --expect original
git -C "$DROPOUT_REPO" apply "$DROPOUT_AUDIT/evidence/patches/source.patch"
ninja -C build-audit -j1 test/unittest/layers/unittest_layers
python3 "$DROPOUT_AUDIT/evidence/reproduce.py" \
  --binary "$DROPOUT_REPO/build-audit/test/unittest/layers/unittest_layers" \
  --output "$DROPOUT_AUDIT/reproduced-patched" --expect patched
cd build-audit
test/unittest/layers/unittest_layers --gtest_filter='DropoutRoutingAudit.*:Dropout/*'
```

The four-test driver requires exactly three failures plus the passing
single-input control on the original, and four passes on the patch.
Crashes, missing XML, skipped tests and unexpected failures are not accepted
as a reproduction. The 19-test selection adds 15 existing tests. Its logger
writes to `./logs/` under the build directory, which must be writable.
The final full-suite command exits nonzero when run against the original.
