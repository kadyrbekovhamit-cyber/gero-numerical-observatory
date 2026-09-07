# Native FP32 CPU reproduction

Recorded host: macOS 15.5 arm64, Apple M4, Apple clang 17.0.0,
Meson 1.12.0, Ninja 1.13.2, CMake 4.4.3, clang-format 14.0.6.
Use an isolated checkout; these commands do not push to upstream.

```sh
export AUDIT_PACKAGE=/absolute/path/to/extracted/package
export AUDIT_NNTRAINER=/absolute/path/to/fresh/nntrainer
git clone https://github.com/nntrainer/nntrainer.git "$AUDIT_NNTRAINER"
git -C "$AUDIT_NNTRAINER" checkout a7ea056e79ab8e14447ea305c1b634e233343258
git -C "$AUDIT_NNTRAINER" submodule update --init --depth 1 subprojects/googletest subprojects/iniparser
git -C "$AUDIT_NNTRAINER" apply "$AUDIT_PACKAGE/patches/nntrainer-tests.patch"
cd "$AUDIT_NNTRAINER"
meson setup build-audit \
  -Denable-blas=false -Denable-app=false -Denable-ccapi=true \
  -Denable-capi=disabled -Dml-api-support=disabled \
  -Denable-tflite-backbone=false -Denable-tflite-interpreter=false \
  -Denable-nnstreamer-tensor-filter=disabled \
  -Denable-nnstreamer-tensor-trainer=disabled \
  -Denable-test=true -Dnntr-num-threads=1 -Dwerror=false -Dthread-backend=none \
  "-Dcpp_args=['-include','$AUDIT_PACKAGE/nntrainer-darwin-build.h']"
```

On macOS create `build-audit/nntrainer/malloc.h` containing
`#include <malloc/malloc.h>`. This and the supplied forced-include header address
Darwin build includes without changing numerical kernels or vendored subprojects.
Meson emits a thread-backend fallback warning. Linux was not tested in this audit.

```sh
ninja -C build-audit -j 4 test/unittest/layers/unittest_layers
cd build-audit
test/unittest/layers/unittest_layers \
  --gtest_filter='LayerNormNumericalAudit.*:EveryNonBatchAxis/*:LayerNormalization/*' \
  --gtest_color=no --gtest_output=xml:before.xml
# Expected: 20 failed, 22 passed (exit status 1).
python3 "$AUDIT_PACKAGE/reproduce.py" \
  --binary "$AUDIT_NNTRAINER/build-audit/test/unittest/layers/unittest_layers" \
  --output "$AUDIT_NNTRAINER/build-audit/reproduction" --expect original
cd "$AUDIT_NNTRAINER"
git apply "$AUDIT_PACKAGE/patches/nntrainer-layernorm.patch"
ninja -C build-audit -j 4 test/unittest/layers/unittest_layers
cd build-audit
test/unittest/layers/unittest_layers \
  --gtest_filter='LayerNormNumericalAudit.*:EveryNonBatchAxis/*:LayerNormalization/*' \
  --gtest_color=no --gtest_output=xml:after.xml
# Expected: 42 passed (exit status 0).
python3 "$AUDIT_PACKAGE/reproduce.py" \
  --binary "$AUDIT_NNTRAINER/build-audit/test/unittest/layers/unittest_layers" \
  --output "$AUDIT_NNTRAINER/build-audit/reproduction" --expect patched
```

`reproduce.py` runs four focused tests three times in fresh processes and checks
the expected outcome. It only runs the provided binary; it does not alter sources.
No Python ML implementation substitutes for the native layer under test.

The 18 upstream LayerNorm tests use the repository's golden fixtures prepared by
Meson. Run the complete filter from `build-audit` as shown. For reconfiguration,
upstream `cp -l` may fail on an existing generated `build-audit/res/test/label.dat`
hardlink; remove only that generated hardlink if necessary, preserving the
original `packaging/label.dat`.

The recorded build reused existing objects, including unrelated earlier loss
and activation repairs. The two files in this report's patches were separately
checked against the exact pinned HEAD. A complete fresh build and the full
upstream CI matrix are outside the recorded validation.
