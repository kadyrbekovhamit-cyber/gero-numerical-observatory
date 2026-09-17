# MLX rsqrt: finite directional derivatives lost to an intermediate ratio

Independent GERO research by Xamit Kadirbekov, 17 September 2026.

The current CPU C++ implementation of reciprocal square root can return infinity
or zero for a finite, representable Jacobian-vector or vector-Jacobian product.
The forward result remains accurate on this test set. A local correction removes
the observed failures; restoring the original code restores them exactly.

## Contract and concrete examples

For a positive real primal `x` and incoming tangent/cotangent `v`,

`f(x) = 1 / sqrt(x)` and `Jv = vJ = -v / (2 * x^(3/2))`.

The incoming weight matters: these examples do **not** claim that the unweighted
derivative at `1e-30` fits in float32. They test the result of the documented JVP
and VJP operation with a nonzero finite weight.

| Float32 inputs (decimal shorthand) | Expected weighted derivative | Current JVP and VJP |
| --- | ---: | ---: |
| `x = v = 1e-30` | approximately `-5e14` | `-inf` |
| `x = v = 1e32` | approximately `-5e-17` | negative zero |

All references use the **actual binary32 input values**, not the decimal
shorthands. For the first row, `x=v=1.0000000031710769e-30` and the rounded reference
is `-499999993495552`. For the second, `x=v=1.0000000331813535e32` and the rounded
reference is `-4.9999997534396726e-17`.

## Source and cause

Executed source: [`59d600b5e64c238427d0f8d897ab7c682ef4d3d2`](https://github.com/ml-explore/mlx/commit/59d600b5e64c238427d0f8d897ab7c682ef4d3d2).
The current head was rechecked after execution and was still this commit.
All 958 Git blobs were verified before and after the experiments, including the
symlink using its link-target bytes. The original tree and build are restored.

[`Sqrt::vjp`](https://github.com/ml-explore/mlx/blob/59d600b5e64c238427d0f8d897ab7c682ef4d3d2/mlx/primitives.cpp#L5596)
forms `rsqrt(x) / x` before multiplying by `-0.5 * v`.
The intermediate can overflow or underflow even when the complete weighted
derivative is a normal finite float32 number. `Sqrt::jvp` delegates to that path.
The [transform API contract](https://github.com/ml-explore/mlx/blob/59d600b5e64c238427d0f8d897ab7c682ef4d3d2/mlx/transforms.h)
defines these operations as Jacobian products.

The two targeted methods are byte-identical in release `v0.32.2`. This is a source
comparison; the release runtime was not executed in this experiment.

## Measured results

There are **2,136 distinct input/seed pairs**, each run as `[2136]`, `[1,2136]`, and
`[2136,1]`. Shapes are additional observations, not additional distinct cases.

| Implementation | Failed pairs per layout | Forward failures |
| --- | ---: | ---: |
| Fresh current CPU build | 443 | 0 |
| Local correction | 0 | 0 |
| Exact original restored and rebuilt | 443 | 0 |

JVP and VJP both fail on the same 443 pairs. Of these, 425 have nonzero seeds;
421 have normal finite primals, seeds **and** expected results. The remaining
18 failures are zero-seed controls. The nonzero-seed cases are the central finding.

- All 50 ordinary control pairs pass in every variant.
- All 39 ordinary first-, second-, and third-derivative checks pass before and
  after the correction, including the scaling split at `x=1`.
- A separate compatibility check covers 64 sqrt/rsqrt cases using negative,
  signed-zero, ordinary positive, infinite and NaN inputs and four seeds.
  All 192 recorded scalar outputs are bit-identical before and after the patch.
  This is compatibility evidence, not a mathematical derivative claim at a pole
  or outside the real function's domain.
- Corrected forward outputs are bit-identical to the original. Restored CSVs
  and the restored probe binary are byte-identical to the original.
- Each variant returns identical bits in all three tested layouts. No batch-shape
  defect is claimed by this report.

An independent mpmath oracle uses exact binary inputs at 100 and 150 decimal
digits and directly rounds to binary32, ties to even. Both precisions give the
same references. The acceptance tolerance is `2e-6 * abs(reference) + 2 * 2^-149`.

## Local correction

For positive finite real `x`, select a detached scale `s=max(x,1)` and evaluate

`((v/s) * (-0.5 * rsqrt(x))) / (x/s)`.

This avoids forming the large or tiny unweighted derivative first. Detaching the
scale leaves a locally constant algebraic factor that cancels, including for
higher derivatives at the split point. For nonfinite primals the scale is one.
The existing complex branch is retained. See `candidate.patch`.

The baseline is a fresh complete CPU library build. The complete portable package was then independently rebuilt from its archived dependencies: all 2,176 source/dependency Git blobs verified, the oracle regenerated exactly, and all nine observation CSVs reproduced byte for byte. The candidate rebuild changes
only `mlx/primitives.cpp` in that build; the restoration repeats the build with
the exact original file. The package includes official pinned fmt 12.1.0 and nlohmann/json 3.11.3 source archives. All 128 fmt and 1,090 json Git blobs were verified against their official repositories. The portable replay uses those archived dependencies. Metal and CUDA are disabled, compilation uses one job, and
the configured numerical worker count is one.

## Prior work and bounded novelty review

The searches and full issue metadata are retained in `source-review/`.

- [PR 3733](https://github.com/ml-explore/mlx/pull/3733) and its related reports
  concern zero incoming cotangents at singular sqrt/rsqrt inputs. This report's
  central examples have positive finite primals and **nonzero** seeds. Zero-seed
  controls are not presented as a separate new defect.
- [Issue 3780](https://github.com/ml-explore/mlx/issues/3780) and
  [PR 3781](https://github.com/ml-explore/mlx/pull/3781) concern complex conjugation.
  Complex correctness is outside this result's scope.
- [PR 4227](https://github.com/ml-explore/mlx/pull/4227) adds derivative reference
  tests over ordinary interior points. It does not report this scale failure.
- The GERO divide-scale and inverse-hyperbolic reports are related numerical
  work, involving different operations. No rsqrt directional-scale report was
  located in the reviewed canonical catalog or bounded local searches.
- Integer mean, offset variance and the earlier local logaddexp complementary
  gradient case were screened out as prior work rather than added as discoveries.

No exact duplicate of the positive-finite, nonzero-seed rsqrt result was identified
in the reviewed material. This is not a claim of exhaustive search or absolute
priority. Repeat the source, issue/PR and live platform checks before publication.

## Reproduction and retained files

Run `reproduce.py` as described in README.md. It unpacks the three source archives, verifies their Git blobs, regenerates the oracle, builds on CPU, applies the patch and restores the original. The resulting replay receipt records exact commands and observations. The following commands describe the individual checks used by that runner:

```text
cmake --build build --target rsqrt_probe rsqrt_controls rsqrt_compat --parallel 1
python check_rsqrt.py prepare
python check_rsqrt.py run build/rsqrt_probe original
build/rsqrt_controls
build/rsqrt_compat
```

Apply `candidate.patch` within the pinned MLX source directory, rebuild the same
targets and run the checks with a new label. Reverse the patch, rebuild and replay
the original. The `observed/` directory retains the executed outputs;
`oracle.json` and `inputs.txt` retain the complete grid. `evidence/research-VERIFICATION_RECEIPT.json` records the research scope. `evidence/portable-replay-receipt.json` records the clean package replay. `SHA256SUMS` covers the frozen files. Compiled binaries and videos are not included.

## Limits and publication status

No GPU, compiler/vmap transformation, performance, full upstream test suite,
extreme-domain Hessian, release-runtime or end-to-end model testing is claimed.
No customer losses or real-device impact was measured. The correction is a
research patch, not an accepted upstream change.

The frozen archive retains source, oracle, native probes, a research patch, raw observations and limitations. Later publication links and maintainer status are tracked separately in the mutable GERO catalog.

Independent GERO research by Xamit Kadirbekov; AI-assisted research and preparation.
