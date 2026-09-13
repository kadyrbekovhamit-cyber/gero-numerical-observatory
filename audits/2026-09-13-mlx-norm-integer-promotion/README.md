# Apple MLX: integer norms can be zero, negative, or NaN before float promotion

Independent numerical audit · 13 September 2026 · Xamit Kadirbekov / GERO Research

`mlx.core.linalg.norm` accepts integer arrays and documents integer examples with a floating-point result. Its native implementation nevertheless performs some integer arithmetic before converting the result. That order can destroy the value before the floating-point result is formed.

The CPU reproduction below confirms one input-promotion defect with several manifestations. This is not a claim of a defect in an Apple device, a security issue, or measured impact on a complete model.

## Minimal evidence

| Input | Requested norm | Original CPU result | Mathematical result | Patched CPU result |
|---|---|---:|---:|---:|
| `[100]`, int8 | default / L2 | 4 | 100 | 100 |
| `[65536]`, int32 | default / L2 | 0 | 65536 | 65536 |
| `[50000]`, int32 | default / L2 | NaN | 50000 | 50000 |
| `[-128]`, int8 | L1 | -128 | 128 | 128 |
| `[-128]`, int8 | infinity | -128 | 128 | 128 |

For a one-element vector, the norm at positive orders is its absolute value. All five expected answers above are exactly representable in float32. No NaN input or ill-conditioned matrix is involved.

Native C++ reproduction:

```cpp
#include <iostream>
#include "mlx/mlx.h"
int main() {
  namespace mx = mlx::core;
  mx::set_default_device(mx::Device::cpu);
  auto a = mx::array({100}, mx::int8);
  auto b = mx::array({65536}, mx::int32);
  auto c = mx::array({-128}, mx::int8);
  auto x = mx::linalg::norm(a);
  auto y = mx::linalg::norm(b);
  auto z = mx::linalg::norm(c, 1.0);
  mx::eval(x, y, z);
  std::cout << x << "\n" << y << "\n" << z << "\n";
}
```

The Python binding forwards to these same native overloads. A Python wheel and GPU were not executed in this audit.

## Why it happens

In [the inspected native source](https://github.com/ml-explore/mlx/blob/229f5b430df7926743c5b6ac62068cae2ebc8978/mlx/linalg.cpp#L59), `l2_norm` computes `sqrt(sum(square(a)))` for real inputs. `square(a)` retains the integer input type. Promotion by a later operation cannot recover information already lost.

On the tested CPU path:

- For int8, `100 × 100 = 10000`, but the retained 8-bit product is `10000 mod 256 = 16`. The final square root is 4.
- For int32, `65536 × 65536 = 2^32`, and the retained 32-bit product is zero. A nonzero vector acquires zero norm.
- For int32, `50000 × 50000 = 2500000000`. The observed signed 32-bit result is -1794967296; taking its square root produces NaN.
- The magnitude of the minimum signed integer is not representable in the same signed type. The tested int8 `abs(-128)` remains -128. L1 and infinity branches compute this magnitude before their final cast, so a quantity defined as nonnegative becomes negative.

These observations describe the measured library behavior; they do not prescribe portable signed-overflow behavior for arbitrary C++ programs.

The helper `at_least_float` already selects float32 for integer and boolean inputs, but the selected dtype is applied too late in several branches. The [API examples](https://github.com/ml-explore/mlx/blob/229f5b430df7926743c5b6ac62068cae2ebc8978/python/src/linalg.cpp#L118) and existing integer tests establish that integer input is part of the supported norm interface. Rejecting every integer input would contradict that interface.

## Proposed repair and its boundary

[norm-integer-promotion.patch](norm-integer-promotion.patch) converts the input to `at_least_float(input.dtype())` at the entry to each of the three public native norm overloads, before square, absolute value, power, or reduction. Inexact inputs keep their existing dtype. The default, numerical-order and string-order entry points therefore share the same input-promotion policy.

For the examples, the intermediate square is evaluated in float32: `10000` and `2^32` are representable, and `abs(-128.0f)` is 128. A user-side workaround for the same bounded inputs is an explicit conversion to float32 before calling norm.

**This patch is not a general overflow-safe norm algorithm.** The deliberately separate [limitation probe](limitation_probe.cpp) retains three cases outside its range guarantee:

- A uint64 maximum promoted to float32 rounds to `2^64`; squaring it overflows float32 even though its L2 norm is representable.
- A very large int64 value with order 3 can overflow an intermediate floating-point cube.
- An already-float32 input near `1e20` has the pre-existing sum-of-squares range limitation.

Those examples require a scale-aware floating-point norm algorithm as an additional change. The candidate does not solve them. It can change the wrong answer in an out-of-scope case, so the bounded passing count must not be presented as correctness over the entire integer domain.

## Validation and assumptions

The regression suite has **1,356 scenarios**, containing **1,428 numerical value comparisons**, plus an output shape and dtype check for each evaluated scenario. These are test counts, not distinct defects.

| Native variant | Failing scenarios / 1,356 |
|---|---:|
| Original pinned source | 244 |
| Input-promotion candidate | 0 |

Coverage includes all eight signed/unsigned integer dtypes, boolean inputs, signed minima, multiplication-overflow boundaries, selected int64/uint64 values, zero vectors, empty default reductions, numerical norm orders, Frobenius norms, matrix row/column reductions, positive/negative axes, transposed views and both keepdims settings. Higher-power tests on very large integers are intentionally excluded from this passing domain and retained in the limitation probe.

The references use host absolute values, long-double `hypot`, sums and powers, without performing the reference square in the input integer dtype. Float32 comparisons use a documented relative tolerance of `3e-6` with unit absolute floor; the selected fractional and negative powers use `1e-5`. Shapes and dtypes are exact checks. Passing tests do not establish correctness for every possible input.

There are 33 inexact control scenarios: float16, bfloat16, float32, float64, a complex64 value, and first/second derivative controls on the real function `norm([3t,4t]) = 5|t|` for nonzero t. The original and candidate records match exactly for these controls. Integer-input gradients are not claimed or needed.

[The complete native suite](native_regression.cpp), [before log](clean-before.jsonl), [after log](clean-after.jsonl), [summary](clean-results-summary.json), and both `clean-*-limits.jsonl` files preserve the observations, including the unresolved range cases. An initial isolated relink and the clean-source build are recorded separately.

## Versions and reproduction

- Compiled source: `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`; version header 0.32.3.
- Current main inspected: `229f5b430df7926743c5b6ac62068cae2ebc8978`, 13 September 2026. Its complete `mlx/linalg.cpp` is byte-identical to the compiled baseline. The whole newer main was not rebuilt.
- Native CPU, AppleClang 17.0.0, macOS 15.5 arm64, Accelerate, CMake/Ninja, Release with C++ `-O0`. Metal and CUDA disabled; one build job and one numerical thread in the recorded environment.
- Source and dependency versions: [source-metadata.json](source-metadata.json).

Install CMake, Ninja and the MLX CPU build prerequisites, then:

```bash
git clone https://github.com/ml-explore/mlx.git mlx-source
python3 build_and_test.py --repo ./mlx-source
```

The runner extracts the pinned source into its own `clean-build` directory. `--deps /path/to/cmake/_deps` may reuse existing `fmt-src` and `json-src`; `--reuse` only reuses this runner's extracted build. The expected original test exit is 1; the candidate exit is 0. The limitation probe records observations and is not a passing-domain test. Build output is in `clean-build.log`.

## Prior work and publication boundary

A bounded GitHub search covered `linalg.norm`, norm/integer, norm/int32, norm/overflow and norm/negative. The original implementation [PR #187](https://github.com/ml-explore/mlx/pull/187), draft #184, matrix-norm alias #3749, shape correction #4166, axis corrections #3827/#3756 and spectral/nuclear addition #1894 were inspected. The search and reviewed messages are recorded in [prior-work.json](prior-work.json). No exact report of this integer-promotion defect was identified in that reviewed material; priority and exhaustive novelty are not established.

This differs from our earlier gradient-clipping range report: that report concerns floating-point sum-of-squares range in clipping. This report concerns supported integer inputs losing information before float promotion inside native `linalg.norm`. The already-published slogdet report concerns a different function and determinant-before-log computation.

GPU, Python-wheel execution, compiled graphs, SVD-backed integer norm behavior, full upstream tests, full models, throughput and maintainer acceptance are not established. The complete signed/unsigned integer range is not repaired. This work was prepared with AI assistance and is intended as a reproducible technical report for review.

Report: CC BY 4.0. Original tests and runner: MIT. MLX source and the candidate patch retain the MLX MIT license; see [LICENSE.md](LICENSE.md) and [MLX-LICENSE](MLX-LICENSE).
