# Activation stability audit — 7 September 2026

Three families of locally reproduced numerical failures in pinned public implementations: MLX ELU/SELU, exact/approximate GELU, and nntrainer Softplus. No matching public report was found in the documented issue/PR searches. That is not a proof of never-reported novelty or upstream acceptance.

| Implementation | Input / dtype | Original result | Mathematical target / patched result |
|---|---|---|---|
| MLX ELU | `100`, float32 | value `100`, gradient `NaN` | value `100`, gradient `1` |
| MLX exact GELU | `40000`, float16 | `+inf` | `40000` at float16 precision |
| MLX approximate GELU | `40000`, float16 | finite value, `NaN` gradient | finite value, gradient `1` at tested precision |
| nntrainer Softplus | `1000`, float32 | `+inf` | `1000` at float32 precision |
| nntrainer Softplus | `-40`, float32 | `0` | about `4.248354e-18`, representable in float32 |

SELU inherits the ELU gradient issue and is grouped with it. Two GELU implementations fail at different intermediate expressions and are presented within one activation family, not counted as repeated discoveries per input or device.

## Environment and source

- Apple M4, macOS 15.5 arm64, Python 3.12.14.
- MLX Python source: `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`, official [ml-explore/mlx](https://github.com/ml-explore/mlx). Native runtime: official `mlx==0.32.2`, `mlx-metal==0.32.2` wheels. Main's entire native core was not built. CPU and Metal tested, float16 and float32.
- nntrainer: `a7ea056e79ab8e14447ea305c1b634e233343258`, official [nntrainer/nntrainer](https://github.com/nntrainer/nntrainer). Real C++ library and tensor activation dispatch, CPU FP32, one thread, BLAS disabled. FP16 nntrainer build, Android, CUDA and throughput were not tested.
- Both main commit IDs were rechecked on 7 September and matched these checkouts.
- Independent ordinary-value and derivative checks use mpmath 1.3.0 at 80 decimal digits. Binary inputs are converted exactly to Python float before evaluation. Large-tail tests use the correctly rounded limiting value at the tested precision, not an equality claim about the real-valued transcendental function.

## What changes

**ELU/SELU:** mask the argument of the inactive exponential branch before evaluating it. Use `expm1` for the negative branch. This avoids `0 * infinity` contamination during autodiff while preserving the original branch choice and its derivative at zero. SELU uses ELU internally.

**Exact GELU:** compute the CDF factor `0.5 * (1 + erf(...))` before multiplying by `x`. The prior order forms `2*x` in the positive tail and overflows even when the final value is representable.

**Approximate GELU:** bound the argument used by the cubic polynomial to `[-10,10]`; the exterior factor keeps the original input. In this tail, tanh has already rounded to +/-1. This avoids overflow in the polynomial's derivative and the resulting NaN. The documented approximation interval `[-6,6]` is unchanged. This is a locally tested patch proposal, not a claim about every compiler/backend.

**Softplus:** replace `log(1 + exp(x))` with `max(x,0) + log1p(exp(-abs(x)))`, respecting the existing beta constant. Intermediate computation uses double and the return type is unchanged. Both positive overflow and cancellation of the small positive negative-tail result are covered.

## Validation

| Suite | Original | Patched |
|---|---|---|
| MLX focused suite, CPU | 10 failed, 16 passed | 26 passed |
| Same MLX focused suite, Metal | Original values independently captured in `mlx-probe.json` | 26 passed |
| nntrainer activation suite, CPU | 4 failed, 20 passed | 24 passed; 2 pre-existing disabled tests remain disabled |

The MLX suite consists of 24 new parameterized checks and 2 existing upstream ELU/GELU tests. It covers positive-branch derivatives, ordinary positive/negative inputs, independent high-precision values and analytic derivatives, representable large tails and elementwise batch-layout invariance. It is not the full MLX suite. The nntrainer suite exercises both scalar helpers and real `ActiFunc::run_fn` tensor dispatch. Existing activation tests remain present.

Earlier loss-function patches in the local checkouts touch different files and are not included in this release's patches. The original activation tests use the pinned HEAD activation source. No model training accuracy, hardware security issue, Claude-model behavior or version-to-version regression is established.

## Reproduce MLX

Use Python 3.12 on macOS Apple Silicon. From this evidence folder, create a virtual environment and a separate source directory. The commands below assume that `AUDIT_SOURCE_ROOT` contains the two checkouts.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export AUDIT_SOURCE_ROOT="$(pwd)/sources"
mkdir -p "$AUDIT_SOURCE_ROOT"
```

```bash
git clone https://github.com/ml-explore/mlx.git "$AUDIT_SOURCE_ROOT/mlx"
git -C "$AUDIT_SOURCE_ROOT/mlx" checkout ce916dbbcaa88e433b6fd1e60a17f766d49c27fe

python run_mlx_tests.py --original --device cpu
# Expected: 10 failed / 16 passed, demonstrating the original failures.

git -C "$AUDIT_SOURCE_ROOT/mlx" apply /absolute/path/to/patches/mlx-activations.patch
python run_mlx_tests.py --device cpu
python run_mlx_tests.py --device gpu
```

The harness explicitly loads the checkout's Python activation module. Installing the wheel and calling `mlx.nn` alone would test the wheel's Python source instead. A sandbox without Metal-device initialization can prevent even CPU MLX imports.

## Reproduce nntrainer

```bash
git clone https://github.com/nntrainer/nntrainer.git "$AUDIT_SOURCE_ROOT/nntrainer"
git -C "$AUDIT_SOURCE_ROOT/nntrainer" checkout a7ea056e79ab8e14447ea305c1b634e233343258
git -C "$AUDIT_SOURCE_ROOT/nntrainer" submodule update --init --depth 1 subprojects/googletest subprojects/iniparser
git -C "$AUDIT_SOURCE_ROOT/nntrainer" apply /absolute/path/to/patches/nntrainer-activation-tests.patch
```

Configure/build in the nntrainer checkout with the documented dependencies and these flags:

```bash
meson setup build-audit -Denable-blas=false -Denable-app=false -Denable-ccapi=true \
  -Denable-capi=disabled -Dml-api-support=disabled \
  -Denable-tflite-backbone=false -Denable-tflite-interpreter=false \
  -Denable-nnstreamer-tensor-filter=disabled -Denable-nnstreamer-tensor-trainer=disabled \
  -Denable-test=true -Dnntr-num-threads=1 -Dwerror=false -Dthread-backend=none \
  "-Dcpp_args=['-include','/absolute/path/to/nntrainer-darwin-build.h']"
```

On macOS, also create `build-audit/nntrainer/malloc.h` containing `#include <malloc/malloc.h>`. These build-only include accommodations are necessary for the pinned Darwin source; no vendored dependency is modified. Meson emits a thread-backend fallback warning in this configuration. For reconfiguration, upstream's `cp -l` may require removing only the generated `build-audit/res/test/label.dat` hardlink; preserve `packaging/label.dat`.

```bash
ninja -C build-audit -j 4 test/unittest/unittest_nntrainer_activations
build-audit/test/unittest/unittest_nntrainer_activations
# Expected: four new tests fail before the source patch.
git apply /absolute/path/to/patches/nntrainer-softplus.patch
ninja -C build-audit -j 4 test/unittest/unittest_nntrainer_activations
build-audit/test/unittest/unittest_nntrainer_activations
```

## Duplicate review

Saved results: `duplicate-search.json`, `duplicate-search-extra.json`, including timestamps, queries, full issue/PR bodies and search completeness flags. Searches included ELU + NaN/gradient, SELU, GELU + overflow/float16/NaN, inactive branch, Softplus and Mish + overflow. Closest matches were MLX #117 (activation introduction), #691/#2151 (compilation/performance), #1499 (memory-growth report), and nntrainer #2545 (Softplus introduction). Comments on #117/#2545 were reviewed. The latter references #2557 for a half-precision build conversion fix, not this numerical failure.

No exact match was identified within that search scope. This release reports reproducible observations and local repairs, not maintainer confirmation or absolute priority. The previously reproduced Bloom #56 duplicate is excluded. QuantizeLinear remains excluded.

Experiments, tests and draft patches were prepared with AI coding assistance. Conclusions rely on recorded execution and mathematical checks. The archive contains source pins, test logs, proposed patches and SHA-256 checksums; it contains no credentials or model API responses.
