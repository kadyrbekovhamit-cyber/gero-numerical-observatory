# A real loss, a reversed gradient: complex autodiff in MLX

**Native CPU reproduction and a tested C++ patch, 9 September 2026.**
Two mechanisms are reproduced: missing conjugation in seven complex VJPs,
and an additional branch/sign error in complex `arccosh` JVP.

The conjugation error class is already known in MLX. This report covers
additional functions and a separately motivated `arccosh` repair. It does
not claim first discovery, an accepted upstream patch or measured harm to
an application. [Prior work #3766](https://github.com/ml-explore/mlx/pull/3766)
is credited explicitly.

[GERO article and short video](https://www.gero.uz/research/articles/mlx-complex-autodiff-reversed-gradient.html)
· [C++ tests](native_regression.cpp) · [Patch](complex-derivatives.patch)
· [Source ledger and prior art](SOURCES.md)

## Minimal example: a real parameter and a real loss

```text
L(x) = Re(cos(i*x)) = cosh(x)
L'(1) = sinh(1) = +1.1752011936438014
```

The official MLX 0.32.2 wheel, explicitly on CPU:

```python
import mlx.core as mx
mx.set_default_device(mx.cpu)
loss = lambda x: mx.real(mx.cos(1j*x))
x = mx.array(1.0)
g = mx.grad(loss)(x)
print(g.item())                 # -1.175201177597046
print(loss(x - 0.01*g).item())   # 1.5569984912872314
```

| Quantity | Original native C++ | Patched native C++ |
|---|---:|---:|
| Derivative at x=1 | −1.17520117759705 | +1.17520117759705 |
| Loss before update | 1.54308068752289 | 1.54308068752289 |
| Loss after update, learning rate 0.01 | 1.55699849128723 | 1.52937591075897 |

The original update increases the loss. Because the parameter and
objective are real, this sign reversal cannot be resolved by renaming a
complex-gradient convention. This is a synthetic scalar example, not a
measurement of a full training run. The wheel reproduces the original
values; the after-patch measurements are from the C++ build described below.

## Mechanism 1: the missing complex conjugate

The affected implementations are `Cos`, `ArcSin`, `ArcCos`, `ArcTan`,
`ArcSinh`, `ArcCosh` and `ArcTanh` in
[pinned primitives.cpp](https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp).
Their backward derivative delegates to the forward derivative:

```cpp
return jvp(primals, cotangents, argnums);
```

For a holomorphic scalar function away from singularities and branch cuts,
using the real inner product `<a,b> = Re(conj(a)*b)`:

```text
JVP(z,t) = f'(z)*t
VJP(z,c) = conj(f'(z))*c
<c, JVP(z,t)> = <VJP(z,c), t>
```

These equations specify the reference independently of the implementation.
MLX's existing `exp` and `log` rules use conjugation; they and `sin` are
controls in the experiment. Returning the unmodified JVP drops that
conjugate. The patch follows the existing conjugate-input / conjugate-output
pattern, which preserves real-input behavior in the tested controls:

```cpp
return {conjugate(
    jvp(primals, {conjugate(cotangents[0], stream())}, argnums)[0],
    stream())};
```

At `z=0.25+0.5i`, cotangent `c=0.75−0.375i`, tangent `t=0.5+0.25i`,
the released wheel gives the following absolute errors:

| Function | VJP error | JVP error | Finite-difference reference error |
|---|---:|---:|---:|
| cos | 0.846736 | 3.03e−8 | 1.80e−5 |
| arcsin | 0.157660 | 6.75e−9 | 3.94e−6 |
| arccos | 0.157660 | 6.75e−9 | 8.97e−5 |
| arctan | 0.580169 | 2.92e−8 | 1.36e−5 |
| arcsinh | 0.270466 | 1.41e−8 | 4.33e−5 |
| arccosh | 1.514187 | 2.32e−8 | 4.71e−6 |
| arctanh | 0.284698 | 1.42e−8 | 1.01e−5 |

Forward results agree with Python `cmath` in this selection. Finite
differences call actual MLX forward on `Re(conj(c)*f(z))`, perturbing real
and imaginary coordinates separately with `h=2^-10`. Scalar products use
double precision. References start from exactly representable inputs.
[Raw vectors and results](probe-results.json) are retained.

Seven affected operations are seven instances of a known class, not seven
claims of independent discoveries.

## Mechanism 2: arccosh's principal-branch derivative

The implementation uses `1/sqrt(z*z−1)`. For the principal branch, away
from the cut and singular points, the relevant derivative is:

```text
1 / (sqrt(z−1) * sqrt(z+1))
```

Principal square roots do not generally distribute over a product. At
`z=−0.25+0.5i` and tangent `t=0.5+0.25i`, these formulas differ in sign:

| arccosh JVP | Complex value |
|---|---|
| Original | −0.178716450929642 + 0.474945813417435i |
| Analytic reference | +0.178716464067613 − 0.474945814576223i |
| Patched | +0.178716480731964 − 0.474945813417435i |

JVP does not require conjugation, so repairing the VJP alone cannot repair
this case. The patch computes the two square roots separately only for
`complex64`; it retains the prior real-input path. Finite differences of
the actual forward support the reference at the tested points. Behavior
on the cut, near ±1 and at extreme magnitudes is outside this audit.

## Actual before/after execution

| Variant | Scenarios | Passing | Failing | Failed numerical assertions |
|---|---:|---:|---:|---:|
| Original | 131 | 46 | 85 | 176 / 622 |
| Patched | 131 | 131 | 0 | 0 / 622 |

Coverage consists of:

- 120 complex scenarios: ten functions × four points `±0.25±0.5i` × three
  cotangents `1`, `i`, `0.75−0.375i`. Each checks forward, JVP, VJP,
  the adjoint relation and finite differences of the real forward loss.
- Ten real-input controls: one per function, including dtype checks for
  float32 JVP and VJP.
- One real-loss chain checking the sign and actual loss decrease above.

The original fails 84 VJP assertions, 84 adjoint assertions, six arccosh
JVP assertions and two assertions in the real-loss example. Forward and
finite-difference assertions pass in both versions. Assertions, scenarios
and root causes are different counts.

Tolerance: `3e-6 + 3e-5*abs(expected)`; finite differences use absolute
tolerance `3e-4`. A passing finite selection is not a universal proof.

[Original log](run-before.log) · [Patched log](run-after.log)
· [Summary](validation-results.json) · [Recorded commands](build-results.json)

## Versions and build provenance

| Component | Recorded value |
|---|---|
| Python probe | Official `mlx==0.32.2` wheel; CPU complex64 / float32 |
| Source inspected on main | `24c699ecee2f7c8b2040de8da1c8382c8bcf31c7` |
| Source used for native compilation | `ce916dbbcaa88e433b6fd1e60a17f766d49c27fe` |
| Compiler | Apple clang 17.0.0 (`clang-1700.0.13.5`), C++20 |
| Environment | macOS 15.5 arm64; Python 3.12.14 |
| Native method | Pristine or patched `primitives.cpp` compiled separately and linked before an existing CPU-only static archive |

**This is a partial native rebuild on an older compatible base, not a clean
build of current main.** The script asserts byte equality of all 20 tested
JVP/VJP method bodies between the base and the inspected main. The patch
also applies cleanly to main and yields the saved candidate byte-for-byte.
The archive hash is in [native-provenance.json](native-provenance.json).

Before publication, all 35 input-manifest entries matched, the wheel probe
was rerun, and the exact native tests and patch were recompiled and rerun
sequentially. Only the build script's two directory assignments were made
configurable through environment variables. The source checkout, original
archive and installed wheel were not edited.

## Reproduce

For the wheel probe on a compatible Mac:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python probe.py
```

For the native comparison, use a compatible CPU-only static build of the
recorded base, with `compile_commands.json` and `mlx-build/libmlx.a`:

```sh
export MLX_SOURCE_ROOT=/absolute/path/to/mlx-at-ce916db
export MLX_CPU_BUILD=/absolute/path/to/cpu-build
.venv/bin/python build_and_test.py
```

The runner reads the build's compiler flags, rebuilds the harness and two
translation-unit variants, links each against the recorded archive and
stores its outputs in this evidence directory. It checks for baseline
exit code 1 and patched exit code 0. The baseline's failed numerical
assertions are intentional, not compiler or execution failures.

[BUILD.md](BUILD.md) includes the CPU-build setup recipe and its validation
boundary. Thread-related environment variables are set to 1 and compilation
and execution are sequential. GPU is disabled in the reused native build.
Physical-core affinity is not claimed. macOS wheel import can initialize
Metal even when all test calculations explicitly use CPU.

## Prior art, limits and attribution

The missing-conjugate class was previously reported for other operations;
see [#3765](https://github.com/ml-explore/mlx/issues/3765),
[#3766](https://github.com/ml-explore/mlx/pull/3766) and
[#3605](https://github.com/ml-explore/mlx/pull/3605).
The inspected main includes commit `af554062e789ccf8d4913a0cb91384515d2a9f2f`
for #3766. Cached web pages and the live PR API disagree in availability;
the API returned 404 at publication, so no current PR status is inferred
from a cached Open badge. Details and search limits are in [SOURCES.md](SOURCES.md).

No full-model training, complete upstream suite, GPU/CUDA, FP16/BF16,
large arrays, branch-cut boundaries or extreme magnitudes were tested.
The original harness initially read an unevaluated const array; that test
error was corrected before the reported runs and is not an MLX finding.

Prepared with AI assistance. Evidence is actual local execution, supported
by separately specified analytic and finite-difference references; it is
not a third-party audit. Included MLX source retains its MIT license.
Independent reproduction and scoped numerical-correctness reviews are welcome.

#MachineLearning #MLX #Autodiff #ComplexNumbers #SoftwareTesting
