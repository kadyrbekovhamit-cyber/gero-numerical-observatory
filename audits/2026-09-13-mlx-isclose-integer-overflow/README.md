# Integer overflow reverses MLX isclose and allclose results

Independent numerical audit by **Xamit Kadirbekov / GERO Research**, 13 September 2026. Prepared with AI assistance.

## Confirmed result

MLX evaluates subtraction and absolute value in the original integer dtype before comparing with a floating tolerance. Unsigned subtraction wraps; signed subtraction and absolute value can overflow. This produces false acceptance and false rejection in `isclose`, and in `allclose`, which reduces its result.

This is **one integer-arithmetic defect family in one helper**, not 233 separate bugs. A bounded CPU prototype corrects the tested inputs where both arrays are integral/bool and no wider than 32 bits. **It is not a complete GPU or 64-bit-integer repair.**

## Minimal reproduction

Equivalent native C++ calls were executed. These Python snippets express the public API; a released Python wheel was not tested:

```python
import mlx.core as mx

a = mx.array([0], dtype=mx.uint8)
b = mx.array([255], dtype=mx.uint8)
mx.isclose(a, b, rtol=0, atol=1, stream=mx.cpu)  # [True]; expected [False]
mx.allclose(a, b, rtol=0, atol=1, stream=mx.cpu) # True; expected False

c = mx.array([-2147483648], dtype=mx.int32)
d = mx.array([0], dtype=mx.int32)
mx.allclose(c, d, rtol=0, atol=0, stream=mx.cpu) # True; expected False
```

| dtype | a | b | rtol | atol | Expected | Original |
|---|---:|---:|---:|---:|---|---|
| uint8 | 0 | 1 | 0 | 1 | True | False |
| uint8 | 0 | 255 | 0 | 1 | False | True |
| int8 | -128 | 0 | 0 | 0 | False | True |
| int32 | -2147483648 | 0 | 0 | 0 | False | True |
| int8 | -127 | -128 | 0.01 | 0 | True | False |

All five also produce the corresponding wrong `allclose` result. See `probe.cpp`, `probe.log` and the individual regression rows.

## Mathematical cause

The [documented comparison](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.isclose.html) is `|a-b| <= atol + rtol*|b|`. The version-pinned Python binding is included as evidence.

The central expressions at `mlx/ops.cpp:2147–2149` are:

```cpp
auto rhs = add(array(atol), multiply(array(rtol), abs(b, s), s), s);
auto lhs = abs(subtract(a, b, s), s);
auto out = less_equal(lhs, rhs, s);
```

For uint8, `0-255` becomes 1 modulo 256, so the code compares `1 <= 1`, although the intended distance is 255. Conversely, `0-1` becomes 255 and rejects a true distance of 1.

For int8, the absolute value of -128 remains -128 in the observed native evaluation because positive 128 does not fit. A negative distance satisfies `-128 <= 0`. Int32 at `-2^31` exhibits the analogous false acceptance.

The relative-tolerance side is also affected: for -127 versus -128, the true distance is 1 and the allowance is 1.28. Integer `abs(-128)` instead yields a negative allowance. A cast after subtraction or after absolute value cannot recover lost information.

## Bounded CPU prototype

The candidate converts both inputs to float64 **before** subtraction and absolute value, and constructs both tolerance scalars as float64. This branch applies only on CPU when both inputs are integral/bool and at most four bytes wide. Other inputs and devices retain their original path.

Every signed/unsigned 32-bit value is exactly representable in binary64. In the joint interval `[-2^31, 2^32-1]`, differences have magnitude below `2^33` and are exact too. The integer oracle uses Python's unbounded integers and exact rational representations of the supplied tolerances.

A blanket float32 cast is unsafe: 16777217 and 16777216 become equal. The candidate passes a regression guard for that pair. A blanket float64 cast is unsafe for int64: 9007199254740993 and 9007199254740992 become equal. `limitations.cpp` executes this example and an unresolved uint64 wrap case. Wider inputs deliberately remain unchanged.

A production repair needs a full integer-distance and tolerance-precision design, device coverage and benchmarks. This branch demonstrates repairability in its stated domain. It is not a proposal to blindly cast arbitrary integer comparisons to floating point.

For exact integer equality, consider `array_equal`, accounting for its different shape/broadcast contract. For the tested CPU domain, explicit conversion of both arrays to float64 before `isclose` is a workaround; it is unsafe for arbitrary int64/uint64 values.

## Executed evidence

Pinned source: [`ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`](https://github.com/ml-explore/mlx/tree/ce916dbbcaa88e433b6fd1e60a17f766d49c27fe), version header **0.32.3**. The entire `isclose` function is identical in inspected main [`229f5b430df7926743c5b6ac62068cae2ebc8978`](https://github.com/ml-explore/mlx/blob/229f5b430df7926743c5b6ac62068cae2ebc8978/mlx/ops.cpp); other parts of that file differ. This is a native source-build result.

| Measurement | Original | CPU prototype |
|---|---:|---:|
| Scenarios | 351 | 351 |
| Failed scenarios | 233 | 0 |
| Elementwise comparisons | 826,670 | 826,670 |
| Previously passing records preserved | — | 118 of 118 |

The suite exhausts all 65,536 ordered pairs separately for int8 and uint8 at six tolerance pairs. It also checks signed/unsigned 16- and 32-bit boundaries, mixed integer dtypes, bool, broadcasting, noncontiguous inputs, empty outputs and unchanged float16/bfloat16/float32/float64 controls with infinities and NaNs. Allclose and output shape/dtype are checked separately from the element count.

The reference calculation does not call MLX and does not use MLX `allclose` to grade MLX. Floating control values are simple exactly representable numbers, with documented special-value handling.

Clean pinned archive, macOS 15.5 arm64, Apple Clang 17, Accelerate, C++20, `-O0 -DNDEBUG`; Metal/CUDA disabled. Build jobs and numerical threads are limited to one. A packaged-runner check repeats the results. Source hashes, exact input corpus, raw JSONL, compiler logs and patch are included. These runs are not performance benchmarks.

## Reproduction

Prerequisites: Python 3, Git, CMake 3.25+, Ninja and a compatible CPU C++20 environment. From the evidence package:

```sh
python3 build_and_test.py
# Optional existing fmt/json dependencies:
python3 build_and_test.py --deps /absolute/path/to/_deps
```

The runner downloads the pinned official MLX archive, builds sequentially, expects the original regression to fail and the candidate to pass, saves both logs and runs the limitation examples. `python3 generate_cases.py` regenerates the oracle corpus without NumPy.

## Prior work and limitations

[Issue #522](https://github.com/ml-explore/mlx/issues/522) and [PR #539](https://github.com/ml-explore/mlx/pull/539) address infinities and NaNs. PR #539 introduced `isclose`; its public source already contains this integer arithmetic. This report does not claim a newly introduced regression. Integer-overflow and large-array reports were also checked. No exact duplicate was identified in the bounded searches saved in `prior-work.json`; priority and maintainer acceptance remain unestablished. The checked public GERO catalog contained 75 documents and no matching comparison audit.

Not established: GPU behavior or repair, full-domain int64/uint64 handling, mixed integer/floating precision beyond the controls, unusual tolerance boundaries, compiled/shapeless graphs, `vmap`, performance, the full upstream suite or downstream-model impact. Passing a finite suite is not proof for every input. No reward eligibility is claimed.

A separate exploratory batched `solve` call raised a shape exception. Its intended broadcasting contract and prior art have not been fully assessed, so it is not presented as another confirmed defect here.

Report: CC BY 4.0. Original test/runner code: MIT. MLX source and patch retain MLX's MIT license. See `LICENSE.md` and `MLX-LICENSE`.
