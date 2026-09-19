# MLX erf: a finite gradient becomes infinity before damping is applied

Status: confirmed native CPU discrepancy; local correction candidates with explicit remaining limitations. Independent publication edition 1.0.0; not submitted to the MLX tracker. Prepared 19 September 2026 by GERO, Xamit Kadirbekov, with AI assistance. No upstream acceptance is claimed.

## Concrete result

On Apple MLX current source `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`, evaluated float32 JVP and VJP of `erf` at `x=1` with incoming seed `float32(3.2e38)` return positive infinity. The correct rounded derivative action is approximately **1.32834399e38**, well within float32 range. The input, stored seed, forward erf value and weighted scalar loss are finite.

The full derivative is

`g * (2 / sqrt(pi)) * exp(-x*x)`.

The implementation first forms `(2 / sqrt(pi)) * g`. With stored seed `3.19999997882106e38`, that intermediate overflows, although multiplication by `exp(-1)` makes the complete mathematical result finite. This is an extreme finite-seed stress case, not a claim about typical training workloads.

A native scalar composition `L(theta)=s*erf(theta)` shows the consequence. At theta=1, s=float32(3.2e38), its evaluated forward is `2.696642089778819e38`. An unscaled update `theta - float32(0.1) * (grad(L)/s)` produces `-Infinity` instead of `0.9584892392158508`. A local GERO finite-value guard consequently skips the update and retains theta=1. The balanced candidate permits the update and saves theta=0.9584892392158508, matching the ordinary unscaled-erf step.

The actual MLX core operations produce the loss, gradient and proposed step. The guard and saved JSON/HTML decision documents are GERO demonstration adapters. Eight separately executed scalar cases give four changed skip/apply decisions and four unchanged controls. These are not a real model, a built-in optimizer product integration, a device failure or measured customer loss.

## Source and API contract

- [Pinned Erf VJP/JVP implementation](https://github.com/ml-explore/mlx/blob/59d600b5e64c238427d0f8d897ab7c682ef4d3d2/mlx/primitives.cpp#L1969).
- [Official erf definition](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.erf.html).
- [Official VJP contract](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vjp.html).
- Current main was refreshed on 19 September and still has the same pin.
- The latest release checked is [v0.32.2](https://github.com/ml-explore/mlx/releases/tag/v0.32.2). Its Erf VJP/JVP source is identical. Its released binary was **not** executed in this audit.

The first probes compiled fresh source objects against an existing complete matching CPU library. A separate clean full-source replay completed successfully at 2026-09-19T12:33:23Z: 958 MLX, 128 fmt and 1090 json Git blobs were verified before building; all 48 grid/scalar/boundary CSV files reproduced byte-for-byte. See `REPLAY_RECEIPT.json`. The source was restored exactly afterward. Metal/CUDA were disabled. No GPU, paid compute or audio playback was used.

## Independent oracle and measured scope

All inputs are preserved as actual binary32 bit patterns. The independent oracle uses mpmath 1.3.0 at 100 and 150 decimal digits, reconstructs exact stored floats, and rounds the analytical derivative to binary32 with ties to even. Both precisions produce identical reference files. No MLX derivative is used as the oracle.

The grid contains 1,292 unique bit-pattern input/seed pairs. Each is executed as a flat vector, row and column. Shape variants are repeated layouts of the same cases, not extra independent cases. Pass tolerance is `3e-6*abs(reference) + 4*2^-149`; finiteness and sign of genuine overflows are checked separately.

| Subset | Cases | Original first-derivative failures | Coefficient-first candidate | Balanced candidate | Restored original |
|---|---:|---:|---:|---:|---:|
| Finite targets, abs(x) <= 8 | 944 | 372 | 4 | 0 | 372 |
| Additional tails, abs(x)=9,10,11,12 | 304 | 216 | 184 | 60 | 216 |
| Correct derivative genuinely overflows | 44 | 0 | 0 | 0 | 0 |

A case fails if any of direct JVP, direct VJP or the composed scalar-loss gradient fails. All evaluated erf forward values pass the same tolerance and their bits are unchanged between variants. All three layouts have identical output bits for each variant. Restoring the original implementation reproduces all original CSV output bytes.

Of the 372 primary failures, 368 occur away from x=0 and show unnecessary intermediate overflow. Four x=+0/-0 rows sit at the float32 upper boundary and depend on rounding of the stored constant; these are retained and explicitly separated from the main x=1 example. The count is a grid result, not 372 independently discovered defects.

## Candidate corrections and limitations

The simple coefficient-first candidate forms `c*exp(-x*x)` before multiplying by the seed. It repairs the headline case but leaves 188 first-action mismatches across the full 1,292-row grid. It does not fix the second derivative in the headline example.

The balanced candidate sets

`h = sqrt(2/sqrt(pi)) * exp(-x*x/2)`

and returns `(g*h)*h`. In exact arithmetic, if h<=1 the intermediate magnitude is at most abs(g); if h>1 it is at most abs(g*h*h). This avoids the original growing prefix when the complete result is representable. Floating-point rounding and backend exponential range still matter. The corresponding native patch is `erf-balanced-candidate.patch`.

This candidate is **not a universal fix**:

- Sixty tail rows at x=+/-12 miss the fixed tolerance (roughly 5.7e-6 relative error in representative cases). They are retained as failures; the tolerance was not relaxed to obtain a zero count.
- An extra control at stored x=13.267000198364258 and seed=float32(3.2e38) still returns zero despite a nonzero finite analytical action. The CPU exponential's early-zero behavior remains a range limitation.
- Reverse-over-reverse second derivatives remain faulty for extreme seeds. In the primary subset, failures fall from 488 to 292; at x=1 and the headline seed, the result remains -Infinity although the complete analytical second derivative is finite. No general Hessian repair is claimed.
- Ordinary small-seed second-derivative controls in the primary subset pass. Tail controls, nonfinite primals and signed-zero behavior are recorded separately.
- No float16/bfloat16/float64, compiled/fused transform, GPU, full upstream suite, real model, optimizer integration or device impact is established.

The balanced patch is a candidate for further engineering review, not an upstream fix or a claim that all numerical limitations have been resolved.

## Downstream documents

`documents/original.json` and `.html` record original proposed steps and the GERO finite-value guard decisions. `documents/balanced.json` and `.html` record corrected first-gradient proposals. `documents/restored.*` reproduce the original behavior. The JSON uses strings for nonfinite values, preserving valid JSON. Raw IEEE bit patterns remain in CSV evidence.

The measured chain is: prescaled derivative overflows -> native unscaled parameter proposal becomes nonfinite -> synthetic guard skips four updates -> the proposed first-gradient correction permits those four updates. The four control decisions remain unchanged. The tiny-step alternative in the raw probe is auxiliary; the reported chain uses ordinary learning rate float32(0.1) after loss-scale division.

## Prior-art review and reporting status

The review read the 31 earlier MLX report IDs and the existing GERO activation report. Six GitHub searches covered erf, erfinv, sqrt gradients, log10, erf overflow and erf cotangents, including open and closed issues/PRs. Relevant bodies and actual patches included #4227, #3927, #3025, #4182, #3692 and #3733; nearby issue #3047 was also reviewed. The recorded search found no exact earlier report of this finite-seed Erf prefix overflow. This is bounded duplicate review, not an exhaustive priority guarantee.

PR #4227 adds unit-cotangent gradient tests; #3025 concerns near-zero forward erf; #3047/#4182 address CPU float64 narrowing; #3692/#3733 address zero-cotangent singularities. The earlier GERO activation report's GELU/Softplus causes are different. DivMod zero-gradient behavior investigated in the same session was excluded from a new-defect claim because its prior issue/PR explicitly preserve that convention.

MLX's issue template prohibits AI-written issues and requests a personal authorship declaration. No issue, PR or maintainer email has been submitted for this case. This is an independent GERO publication with disclosed AI assistance, not an upstream contribution or an accepted fix. On 19 September 2026 the author explicitly authorized this public edition before any eligible upstream report. Historical research receipts describe the earlier local state; the numerical evidence is unchanged.

<!-- GERO_PUBLICATION_LINKS_BEGIN -->
## Verified publication and reproduction links

- [github](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/mlx-erf-finite-seed-intermediate-overflow.md)
- [zenodo](https://zenodo.org/records/22845404)
- [gero](https://www.gero.uz/research/articles/mlx-erf-finite-seed-intermediate-overflow.html)
- [huggingface](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/mlx-erf-finite-seed-intermediate-overflow.md)
- [linkedin](https://www.linkedin.com/feed/update/urn:li:share:7507063710302244865/)
- [youtube](https://www.youtube.com/shorts/DQ1XiJh29H0)

[Complete reproducibility ZIP](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/raw/refs/heads/main/reports/mlx-erf-finite-seed-intermediate-overflow/gero-mlx-erf-finite-seed-publication-2026-09-19.zip).
Archive SHA-256: 9b75d95fd7b45217d54977d46e06e43db4f482b834f05dbc6fcfe746241dfbb6.
Zenodo DOI: 10.5281/zenodo.22845404.
Verified YouTube master duration: 31.521333 seconds.
Hugging Face hosts the report and patch; its ZIP mirror is pending a network upload error. The complete verified ZIP is available on GitHub and Zenodo.

Upstream report has not been sent. Candidate corrections are local; no upstream acceptance or real-model impact is claimed.
<!-- GERO_PUBLICATION_LINKS_END -->
