# nntrainer FP32 CPU reproduction

The recorded host was macOS 15.5 arm64. Build tools: Meson 1.12.0, Ninja 1.13.2,
CMake 4.4.3 and clang-format 14.0.6. Set `AUDIT_PACKAGE` to this extracted
package and `AUDIT_NNTRAINER` to a fresh checkout. Use absolute paths.

```sh
export AUDIT_PACKAGE=/absolute/path/to/this/package
export AUDIT_NNTRAINER=/absolute/path/to/nntrainer
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

On macOS create `build-audit/nntrainer/malloc.h` containing one line:
`#include <malloc/malloc.h>`. The supplied forced-include header and this shim
address missing Darwin system includes in the pinned source. They do not change
the mathematical implementation or vendored subprojects. Meson emits an upstream
thread-backend fallback warning. The corresponding Linux build was not tested.

```sh
ninja -C build-audit -j 4 test/unittest/layers/unittest_layers
build-audit/test/unittest/layers/unittest_layers --gtest_filter='SigmoidLossStability.*'
# Expected: 4 failed / 1 passed.
git apply "$AUDIT_PACKAGE/patches/nntrainer-bce.patch"
ninja -C build-audit -j 4 test/unittest/layers/unittest_layers
build-audit/test/unittest/layers/unittest_layers --gtest_filter='SigmoidLossStability.*'
# Expected: 5 passed.
build-audit/test/unittest/layers/unittest_layers --gtest_filter='SigmoidLossStability.*:LossCross/*'
# The 66 upstream loss semantics tests are also included.
```

The archived 77-test run additionally retains six independent softmax tests
from the previous audit. Those six are not introduced by this package's patches.
For reconfiguration, upstream's `cp -l` may fail on an existing generated
`build-audit/res/test/label.dat` hardlink. Remove only that generated hardlink
if needed; preserve the original `packaging/label.dat`.
