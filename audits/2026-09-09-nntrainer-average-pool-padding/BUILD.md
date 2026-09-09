# Reproduce the average-pooling padding regression

The measured checkout was nntrainer main commit
`a7ea056e79ab8e14447ea305c1b634e233343258` (Meson reports development
version 0.6.0), macOS 15.5 arm64, Apple clang 17, Meson 1.12.0,
Ninja 1.13.2 and clang-format 14.0.6. CPU FP32 only was tested.

Use a separate checkout. Commands below describe a fresh reproduction;
this audit reused the existing configured build and did not run a new
complete checkout/platform build. The baseline pooling source exactly
matched the pinned commit. Earlier unrelated changes to activations,
losses, attention and LayerNorm are present in that build and are not
part of either supplied patch. No vendored source was edited.

```sh
export POOL_AUDIT=/absolute/path/to/this/evidence-directory
export POOL_REPO=/absolute/path/to/isolated/nntrainer
git clone https://github.com/nntrainer/nntrainer.git "$POOL_REPO"
git -C "$POOL_REPO" checkout a7ea056e79ab8e14447ea305c1b634e233343258
git -C "$POOL_REPO" submodule update --init --depth 1 \
  subprojects/googletest subprojects/iniparser
git -C "$POOL_REPO" apply "$POOL_AUDIT/patches/tests.patch"
cd "$POOL_REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
meson setup build-audit \
  -Denable-blas=false -Denable-app=false -Denable-ccapi=true \
  -Denable-capi=disabled -Dml-api-support=disabled \
  -Denable-tflite-backbone=false -Denable-tflite-interpreter=false \
  -Denable-nnstreamer-tensor-filter=disabled \
  -Denable-nnstreamer-tensor-trainer=disabled \
  -Denable-test=true -Dnntr-num-threads=1 -Dwerror=false -Dthread-backend=none \
  "-Dcpp_args=['-include','$POOL_AUDIT/nntrainer-darwin-build.h']"
```

On macOS, create `build-audit/nntrainer/malloc.h` containing
`#include <malloc/malloc.h>`. The forced-include header and shim address
Darwin headers only. For non-Darwin systems omit these compatibility
steps; those platforms were not run in this audit.

```sh
ninja -C build-audit -j 1 test/unittest/layers/unittest_layers
python3 "$POOL_AUDIT/reproduce.py" \
  --binary "$POOL_REPO/build-audit/test/unittest/layers/unittest_layers" \
  --output "$POOL_AUDIT/reproduced-original" --expect original
# The driver succeeds only when the C++ suite has 58 tests / 8 failures.
git -C "$POOL_REPO" apply "$POOL_AUDIT/patches/source.patch"
ninja -C build-audit -j 1 test/unittest/layers/unittest_layers
python3 "$POOL_AUDIT/reproduce.py" \
  --binary "$POOL_REPO/build-audit/test/unittest/layers/unittest_layers" \
  --output "$POOL_AUDIT/reproduced-patched" --expect patched
# The driver now requires 58 tests / 0 failures.
```

Tests use only synthetic tensors. The two focused tests check a hand
calculation and central differences of the real C++ forward. Ten matrix
cases compare a separate window-incidence Jacobian and check incoming
gradient scaling, zero gradient, immutability and gradient-sum conservation.
The other 46 tests are existing pooling semantics/property tests, not
new numerical golden coverage. Build commands run one Ninja job;
NNTrainer is configured for one thread with BLAS and thread backend off.
