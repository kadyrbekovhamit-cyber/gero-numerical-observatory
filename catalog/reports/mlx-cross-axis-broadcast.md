> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/mlx-cross-axis-broadcast.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# Parallel vectors produce nonzero cross products in Apple MLX

Independent numerical audit by **Xamit Kadirbekov / GERO Research**, 13 September 2026. Prepared with AI assistance.

## Result and scope

The native `mlx::core::linalg::cross` implementation can silently compute the wrong vector products when its inputs have different ranks and a nonnegative component axis. Other valid shape combinations raise a broadcasting exception. The implementation broadcasts the original shapes before moving the component axes into alignment. Right-aligned broadcasting can therefore change which output axis contains the components of the lower-rank input.

This is **one axis-alignment defect**, with multiple input and dtype manifestations. It is separate from the earlier integer-norm report and from the already-fixed negative out-of-bounds validation in PR #4118. Priority and maintainer acceptance are not established.

Tested source: commit [`ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`](https://github.com/ml-explore/mlx/tree/ce916dbbcaa88e433b6fd1e60a17f766d49c27fe), whose version header is **0.32.3**. The full `mlx/linalg.cpp` file is byte-identical at inspected main [`229f5b430df7926743c5b6ac62068cae2ebc8978`](https://github.com/ml-explore/mlx/blob/229f5b430df7926743c5b6ac62068cae2ebc8978/mlx/linalg.cpp). This is a source-build result, not a claim about a released Python wheel or every Apple device.

## A zero answer becomes a nonzero matrix

Let `a = [1, 2, 3]` and let the columns of `B` all equal `a`:

```python
import mlx.core as mx

a = mx.array([1., 2., 3.])
B = mx.array([[1., 1., 1.],
              [2., 2., 2.],
              [3., 3., 3.]])
y = mx.linalg.cross(a, B, axis=0, stream=mx.cpu)
mx.eval(y)
print(y)
```

Each column should be `a × a = [0, 0, 0]`. The original native code returns:

```text
[[ 1,  2,  3],
 [-2, -4, -6],
 [ 1,  2,  3]]
```

No extreme values, rounding thresholds or ill-conditioned matrices are involved. All values are small exactly representable integers stored as float32. The Python snippet expresses the public call; the executed reproducer is the equivalent native C++ call in `probe.cpp` and `native_regression.cpp`.

An immediate workaround for this particular vector-versus-column-batch case is to explicitly insert the missing batch dimension:

```python
mx.linalg.cross(a[:, None], B, axis=0, stream=mx.cpu)
```

The native counterpart was executed and returns zero. For general different-rank inputs, align the component axis of each input before broadcasting batch dimensions; blindly adding one dimension is not a universal workaround.

## Why the formula changes

For the intended interpretation, the `j`th output is

`C[:, j] = a × B[:, j] = a × a = 0`.

The original implementation instead broadcasts shape `(3,)` to `(3,3)` by copying the row `[1,2,3]`. The `j`th column of the first argument becomes `a[j] * [1,1,1]`, while the code still treats output axis zero as its component axis. It consequently computes

`C[:, j] = a[j] * ([1,1,1] × [1,2,3]) = a[j] * [1,-2,1]`.

That expression gives the observed matrix exactly. Explicit batch expansion restores the intended vector identity.

The same issue appears against the three coordinate basis vectors. For `B = eye(3)` and `axis=0`:

| Row | Expected | Original MLX |
|---|---|---|
| 0 | `[0, -3, 2]` | `[0, -2, 3]` |
| 1 | `[3, 0, -1]` | `[1, 0, -3]` |
| 2 | `[-2, 1, 0]` | `[-1, 2, 0]` |

The independent NumPy calculation agrees with the explicit vector formula. NumPy defines `axis` as the component axis of each input and the result and supports broadcasting; MLX's existing cross-product tests also compare with NumPy. MLX supports two-component inputs as zero-padded three-component vectors. This audit preserves MLX's three-component output for that case rather than importing NumPy's historical scalar-output convention. [NumPy documentation](https://numpy.org/doc/stable/reference/generated/numpy.cross.html)

## Candidate correction

The patch is limited to `linalg::cross` in `mlx/linalg.cpp`:

1. Move each input's specified component axis to its last axis.
2. Broadcast the remaining batch dimensions while retaining each input's component count of two or three.
3. Apply the existing cross-product arithmetic on the common last axis.
4. Move the three-component result back to the requested output axis.

See [`cross-axis-broadcast.patch`](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/audits/2026-09-13-mlx-cross-axis-broadcast/cross-axis-broadcast.patch). The existing dtype promotion, two-component zero padding and invalid-axis validation remain in place. No changes to autodiff primitives are proposed.

## Executed validation

A full clean CPU build from the pinned source archive repeats the isolated native-object result:

| Measurement | Original | Candidate |
|---|---:|---:|
| Scenarios | 3,383 | 3,383 |
| Failed scenarios | 594 | 0 |
| Exceptions on intended valid inputs | 432 | 0 |
| Wrong-result scenarios | 162 | 0 |
| Numerical values actually compared | 69,918 | 74,838 |

The original run compares fewer values because some calls throw before returning an array. Shapes and dtypes are checked separately. The **2,789 control records are identical** before and after; those include the generated same-rank and negative-axis cases plus five expected invalid-input rejections. Test counts are not counts of independent defects.

The suite contains 320 deterministic shape/vector cases, evaluated across float16, bfloat16, float32, float64, int8 and int32, mixed inputs, complex64 arithmetic and noncontiguous views. It includes two- and three-component vectors, ranks one through four, positive and negative axes, broadcasting, singleton and empty batch dimensions. Values are deliberately small to isolate axis semantics from overflow and rounding.

The independent host oracle expands the Levi-Civita formula using explicit coordinate indexing. All 320 outputs agree with NumPy 1.23.5 after explicit zero padding of two-component inputs. A further 162 host finite-difference comparisons check derivatives of the explicit formula. Float64 native VJP/JVP checks compare with the analytic oracle, and one composed scalar case checks second derivatives.

For that scalar case, `a(t)=[1,2,t]`, `B=eye(3)`, and `L(t)=cross(a(t),B,axis=0)[0,1]²/2`. The intended function is `t²/2`. At `t=3`, original MLX returns loss `2`, gradient `0`, Hessian `0`; the candidate returns `4.5`, `3`, and `1`. These derivative errors follow from computing the wrong forward operation; they are not claimed as separate autodiff defects.

Build and tests ran sequentially on macOS 15.5 arm64 with Apple Clang 17, Accelerate, one build job and numerical thread limits of one. Metal and CUDA were disabled. Source/compiler/dependency details are in `source-metadata.json`; raw results are in `clean-before.jsonl` and `clean-after.jsonl`. An initial harness compile error in dtype log formatting was corrected before executing these tests, and its log is retained.

## Reproduce

Prerequisites: Python 3, Git, CMake 3.25 or newer, Ninja and a compatible C++20 CPU build environment. The build downloads the pinned MLX source archive and its declared dependencies. An optional existing CMake dependency directory can avoid downloading fmt and nlohmann/json again.

```sh
python3 build_and_test.py
# Or reuse dependencies:
python3 build_and_test.py --deps /absolute/path/to/_deps
```

The runner expects a failing original regression process and a passing candidate process, writes both logs, and stops on any other result. To independently regenerate and verify the input corpus, install NumPy and run `python3 generate_cases.py`. The distributed `cases.json` makes NumPy unnecessary for the native regression itself.

## Prior work and limits

Public searches inspected the cross-product feature request [#1244](https://github.com/ml-explore/mlx/issues/1244), its implementation [PR #1252](https://github.com/ml-explore/mlx/pull/1252), review comments and the later bounds-validation correction [PR #4118](https://github.com/ml-explore/mlx/pull/4118). The latter concerns invalid negative axes and does not align component axes for valid different-rank inputs. No exact duplicate was identified in the bounded searches; this is not proof of novelty. Search details are preserved in `prior-work.json`. The existing 72-document GERO catalog contained no publication of this finding at the check time.

Not tested: GPU backends, released Python-wheel execution, compiled graphs, `vmap`, the complete upstream test suite, large-tensor performance or downstream models. No production-device impact, exploit, reward eligibility or maintainer acceptance is claimed. Passing the finite suite is not a proof for every shape or dtype. Integer overflow at large magnitudes is outside this report.

Report: CC BY 4.0. Original test and runner code: MIT. MLX source and the candidate patch retain MLX's MIT license. See `LICENSE.md` and `MLX-LICENSE`.
