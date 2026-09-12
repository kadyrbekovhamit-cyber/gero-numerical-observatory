# MLX LayerNorm: an optional unit weight changes bias precision

**Native CPU case study · 13 September 2026 · Xamit Kadirbekov / GERO Research**

A finite float32 bias becomes infinity, loses low bits and underflows when
`fast.layer_norm` receives a float16 input and no weight. Supplying a float16
unit weight preserves the same bias exactly. The two calls represent the same
affine normalization formula, but select different output dtypes.

A clean native CPU build reproduces this inconsistency. A one-line candidate
patch changes **106 failed assertions out of 385 to zero**. These assertions
describe one cause, not 106 independent bugs. Maintainer acceptance and
model-level impact are not established.

## Minimal observation

Let `x = [0, 0, 0, 0]` in float16, positive epsilon, and
`b = [70000, 1.000244140625, 2**-30, -70000]` in float32.
All bias values are finite and exactly representable in float32.

| Native CPU call | Output dtype | Observed output |
|---|---|---|
| Original, `weight=None` | float16 | `[inf, 1, 0, -inf]` |
| Original, float16 unit weight | float32 | exactly `b` |
| Candidate, `weight=None` | float32 | exactly `b` |

```cpp
#include <iostream>
#include <optional>
#include "mlx/mlx.h"
int main() {
  namespace mx = mlx::core;
  mx::set_default_device(mx::Device::cpu);
  auto x = mx::zeros({4}, mx::float16);
  auto b = mx::array({70000.f, 1.000244140625f, 0x1p-30f, -70000.f});
  auto a = mx::fast::layer_norm(x, std::nullopt, b, 1e-5f);
  auto c = mx::fast::layer_norm(x, mx::ones({4}, mx::float16), b, 1e-5f);
  mx::eval(a, c);
  std::cout << a << "\n" << c << "\n";
}
```

The backward pass also loses information: with zero input and zero float32
bias, a finite incoming cotangent equal to `b` should pass unchanged through
the additive bias. The original bias VJP is `[inf, 1, 0, -inf]`, already labeled
float32; casting back cannot recover the lost values. The candidate returns
the original cotangent exactly in this test.

## Formula, documented behavior and proposed policy

For each last-axis row,

`mu = mean(x)`, `v = mean((x-mu)^2)`,
`y = ((x-mu) / sqrt(v+eps)) * w + b`.

The [official API documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.layer_norm.html)
describes optional multiplicative weight and additive bias. With zero `x` and
positive epsilon the normalized component is zero; replacing an omitted
weight by one does not change this formula.

That documentation does **not** specify a complete mixed-dtype promotion
table. We therefore distinguish the observed inconsistency from the proposed
repair policy: include every present affine parameter in normal MLX dtype
promotion. This is not a claim that every finite mixed-precision operation
must have finite output or preserve all real-number identities.

In [the implementation's output-type branch](https://github.com/ml-explore/mlx/blob/dfe17bafb23e66fe56596df532a497ab3611d0e5/mlx/fast.cpp#L337),
bias participates in `result_type` only when weight exists. Otherwise the
output type is the input type, and bias is narrowed before addition.
The [candidate patch](layernorm-bias-dtype.patch) changes only that last branch:

```cpp
// no weight
has_bias ? result_type(x, *bias) : x.dtype()
```

An input-dtype unit weight is a measured workaround for the covered CPU cases.
GPU behavior and performance of that workaround remain untested.

## Actual validation

The [native suite](native_regression.cpp) uses the real C++ MLX runtime.
It covers:

- 48 constant-input scenarios: 16 input/bias dtype pairs across float16,
  bfloat16, float32 and float64, with shapes `(4,)`, `(2,4)` and `(2,3,4)`.
- 32 nonconstant scenarios with contiguous and strided `(2,4)` inputs.
- Values, output dtype, absent/unit-weight equivalence, input/bias JVP and VJP;
  one additional bias-VJP precision case.
- 16 same-dtype controls spanning the four weight/bias-presence combinations.

The original and candidate each execute 385 assertions. The original fails
106; the candidate fails zero. All 16 control records are unchanged.
See [original results](run-before.json), [candidate results](run-after.json),
[original native log](run-before.log), and [candidate native log](run-after.log).

For a nonconstant row `[-1,1,-1,1]` and epsilon 3, the normalized row is
`[-0.5,0.5,-0.5,0.5]`. The analytical JVP for `[1,0,0,0]` is
`[.34375,-.09375,-.15625,-.09375]`; the analytical VJP for `[1,-2,3,-4]` is
`[.4375,-.4375,1.4375,-1.4375]`. These binary-exact fixtures support exact
comparisons after representation in the expected dtype. The dtype reference
uses MLX's general `promote_types`, independently of LayerNorm's branch.

Publication verification rebuilt the complete CPU library from a clean Git
archive of **`ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`** with AppleClang 17,
Accelerate, `-O0`, Metal/CUDA disabled, and one build/numerical thread. It then
applied the candidate, rebuilt the changed source and relinked. Both result
sets match the earlier isolated-object experiment byte for byte as JSON data.
This removes dependence on the earlier reused CPU archive. See
[build provenance](verification.json) and [portable instructions](BUILD.md).

The same LayerNorm function text was checked against saved source at
`81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`. The relevant branch was also
read at public commit `dfe17bafb23e66fe56596df532a497ab3611d0e5`.
**The complete newer commit was not built.** Installed Python wheel 0.32.2
could not import because Metal was unavailable in this session; no numerical
wheel or GPU result is claimed.

## Existing tests and duplicate search

The [upstream LayerNorm test](https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/python/tests/test_fast.py#L600)
checks parameter-presence combinations using the same dtype for input and
parameters. This misses the mixed-dtype condition here. Its ordinary
[reference helper](https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/python/tests/test_fast.py#L58)
adds bias after normalization. Those Python tests were not run in this session.

No exact duplicate was found in the bounded public review: the titles of
35 LayerNorm issues/PRs were inspected and the closest reports read. A fresh
`layer_norm bias` search before publication still returned #3630, #3220 and
#2300. [Search details and distinctions](DUPLICATES.md) are retained.
This is not an exhaustive novelty guarantee or a version-to-version regression claim.

## Limits and reuse

The candidate is a research patch. GPU, compiled execution, higher derivatives,
integer/complex parameters, empty arrays, full training and performance are
outside this validation. Changing promotion may change acceptance of unsupported
type combinations; maintainers should decide that policy before integration.
The patch does not repair separate float64-statistics narrowing in LayerNorm.

This report is independent and prepared with AI assistance. Source pins,
measured failures and limitations are part of the result. Finite tests are
not a formal correctness proof. See [licenses](LICENSE.md).
