# Native CPU reproduction

The recorded run used the main source commit below, macOS 15.5 arm64,
Apple clang 17, Meson 1.12.0, Ninja 1.13.2 and clang-format 14.0.6.
Use an isolated checkout. No upstream push is required to reproduce this case.

```sh
export AUDIT_PACKAGE=/absolute/path/to/extracted/attention-audit
export AUDIT_NNTRAINER=/absolute/path/to/nntrainer
git clone https://github.com/nntrainer/nntrainer.git "$AUDIT_NNTRAINER"
git -C "$AUDIT_NNTRAINER" checkout a7ea056e79ab8e14447ea305c1b634e233343258
git -C "$AUDIT_NNTRAINER" submodule update --init --depth 1 \
  subprojects/googletest subprojects/iniparser
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
`#include <malloc/malloc.h>`. The forced-include header and this shim address
Darwin includes without editing vendored subprojects or mathematical kernels.
Meson reports a thread-backend fallback warning. Other platforms were not tested.

```sh
ninja -C build-audit -j 4 test/unittest/layers/unittest_layers
cd build-audit
test/unittest/layers/unittest_layers \
  --gtest_filter='AttentionNumericalAudit.*:CausalShapes/*:Attention/*' \
  --gtest_color=no --gtest_output=xml:before.xml
# Original: 21 failed / 21 passed, exit status 1.
python3 "$AUDIT_PACKAGE/reproduce.py" \
  --binary "$AUDIT_NNTRAINER/build-audit/test/unittest/layers/unittest_layers" \
  --output "$AUDIT_NNTRAINER/build-audit/reproduction" --expect original
cd "$AUDIT_NNTRAINER"
git apply "$AUDIT_PACKAGE/patches/nntrainer-attention.patch"
ninja -C build-audit -j 4 test/unittest/layers/unittest_layers
cd build-audit
test/unittest/layers/unittest_layers \
  --gtest_filter='AttentionNumericalAudit.*:CausalShapes/*:Attention/*' \
  --gtest_color=no --gtest_output=xml:after.xml
# Patched: 42 passed, exit status 0.
python3 "$AUDIT_PACKAGE/reproduce.py" \
  --binary "$AUDIT_NNTRAINER/build-audit/test/unittest/layers/unittest_layers" \
  --output "$AUDIT_NNTRAINER/build-audit/reproduction" --expect patched
```

Run from the build directory so the 14 existing attention tests find their
golden fixtures. `reproduce.py` runs four focused cases in three fresh processes
and verifies their expected outcomes without modifying any source.

The recorded build reused existing objects and earlier unrelated loss,
activation and LayerNorm repairs. Both files in this report's patches were
separately applied to clean pinned copies and compared with the tested files.
The fresh-checkout instructions describe the intended independent reproduction;
a complete fresh repository build and upstream CI matrix have not been executed
in this round. The original attention source exactly matched the pinned commit.
