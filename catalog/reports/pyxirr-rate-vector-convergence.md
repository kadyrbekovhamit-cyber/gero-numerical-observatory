> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/pyxirr-rate-vector-convergence.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# PyXIRR vectorized RATE: unconverged values and zero-rate instability

Independent GERO investigation, 13 September 2026. Synthetic inputs only. Analysis, code preparation and validation assisted by AI.

## Confirmed scope

Two defects in the vectorized periodic `rate` solver reproduce on PyPI **0.10.8**, and on a fresh native build of upstream `main`, commit **5f31a85684051fb6badb542afba0ac1af5a68f9c**. GitHub main and PyPI's latest version were checked during this investigation. No direct duplicate was found in the bounded screening described in `DUPLICATE_REVIEW.md`.

The [official API documentation](https://anexen.github.io/pyxirr/functions.html#rate) accepts a scalar or array of periods, has a `guess` parameter, and documents failure as `None`. The [vectorization documentation](https://github.com/Anexen/pyxirr#vectorization) describes elementwise/broadcast results. Rates are per payment period, not automatically annual percentages.

## 1. A negative final Newton step bypasses the failure test

```python
import pyxirr as px
px.rate([360], -1000, 200000, guess=0.5)
# Released/native baseline: [0.13584408469291567]
# Candidate: [None]
```

For a synthetic loan of 200,000 and 360 end-of-period payments of 1,000, the unique economically admissible rate solves

`200000 = sum(1000 / (1 + r)**k for k in range(1, 361))`.

Independent 70-digit Decimal discounting and bisection give **r = 0.00365592795236270985246…**, or **0.3655927952% per period**. The reported value is **13.5844084693% per period**. At that value, the discounted payments total only **7,361.38052871838**, not 200,000. This discrepancy is a synthetic valuation residual, not an observed customer loss.

The explicitly supplied starting guess is 0.5, meaning 50% per period. With the same start, the scalar API returns `None`; with the default starting guess it finds the valid rate. The defect is not that a difficult start must always converge. The defect is that the vector implementation returns an unfinished iterate as a result after 100 iterations. A separate case with 1,200 periods also reproduces with the default starting guess 0.1; see the retained grid.

In [`src/core/periodic.rs`](https://github.com/Anexen/pyxirr/blob/5f31a85684051fb6badb542afba0ac1af5a68f9c/src/core/periodic.rs), the loop tests `abs(diff) < 1e-6`, but its final rejection mask tests `diff > 1e-6` without an absolute value. A large negative step therefore escapes rejection. The candidate uses a finite absolute step. It returns an explicit failure for the five difficult cases; it does **not** claim to solve them from those starting guesses.

## 2. The vector derivative is singular at zero and unstable nearby

```python
px.rate(1, -1100, 1000, guess=0)       # about 0.1
px.rate([1], -1100, 1000, guess=0)     # baseline [None], candidate about [0.1]
px.rate(12, -100, 1200, guess=0)       # 0.0
px.rate([12], -100, 1200, guess=0)     # baseline [None], candidate [0.0]
```

The first equation is simply `1000*(1+r) - 1100 = 0`; the unique answer is 10%. A zero starting guess introduces divisions by `r` and `r**2` in the vector derivative, producing NaN before it can take a useful step. The scalar path uses a different derivative and succeeds on these examples.

The fixed control grid also exposed a near-zero intermediate failure with the default guess:

```python
px.rate([2], -497.48743718592965, 1000, pmt_at_beginning=True)
# Baseline: [4.961309141293861e-15], approximately zero
# Independent root / candidate: approximately [-0.01]
```

There are two equal payments, at time zero and time one. Their discounted sum is monotone in the rate, so the negative root is unique. An ordinary Newton path passes close to zero, where cancellation in the derivative creates a false stopping condition. This additional failing control was retained, not removed to improve the test score; see the protocol amendment and `release-initial.json`.

The candidate evaluates the annuity factor `A(r) = ((1+r)**n - 1)/r` through its fourth-order binomial expansion when `abs(r)*max(abs(n),1) <= 1e-4`, and differentiates the same expansion. At zero it gives the exact limits `A(0)=n` and `A'(0)=n*(n-1)/2`. Terms are formed with powers of the small scaled rate rather than large binomial coefficients. Other parts of the solvers are unchanged.

## Verification

| Public API observations | PyPI wheel | Native baseline | Native candidate |
|---|---:|---:|---:|
| Reject unfinished iterates, five inputs in three shapes | 0/15 | 0/15 | 15/15 |
| Zero starting guess, 40 inputs in three shapes | 0/120 | 0/120 | 120/120 |
| Default-start controls, same 40 inputs in three shapes | 117/120 | 117/120 | 120/120 |
| Total | 117/255 | 117/255 | **255/255** |

The released and native-baseline observed rates matched exactly in this grid, including all failure states. Controls include negative, zero and positive rates; end/beginning payments; principals of 1,000 and 1,000,000; lists, NumPy arrays and two-dimensional inputs. One immediate payment with no later cash flow is excluded because it does not identify a rate.

- **242/242** tests passed in the candidate's complete `cargo test --release` suite, including five new native public-API regressions; no ignored or filtered tests in that full run.
- Restoring only the signed failure mask makes its regression fail. Restoring only the original derivative makes all four zero/near-zero regressions fail. Both failures are numerical assertion failures, not build errors. Candidate source was restored and hash-checked afterward.
- **154/154** additional native API checks passed around both sides of the series threshold, for 1–1,200 periods, positive/negative/zero rates and both payment timings. Each was checked by direct Decimal discounting; present-value residuals are below 0.00001 on principal 1,000. See `series-boundary.json` for actual values and tolerances.
- `cargo fmt --check` passes. Only `src/core/periodic.rs` and `tests/test_periodic.rs` differ from the upstream snapshot.
- Rust builds and numerical runs were sequential with one worker. Tests executed the actual compiled Rust module, not a Python replacement of the faulty algorithm.

These counts describe test observations, not numbers of independent defects or affected users. No timings, market prevalence, production deployments, transactions or customer losses were measured. The correction does not promise global convergence, resolve every scalar-solver edge case, validate all extreme magnitudes, or define selection among multiple roots.

## Build provenance and reproduction

Recorded environment: macOS arm64, Python 3.12.13, Rust 1.98.1, NumPy 2.5.2, Maturin 1.15.0, pandas 2.3.3, numpy-financial 1.0.0. Full execution commands are retained in `commands.json`; source, wheel and result hashes are recorded in the package manifest.

Both native wheels were compiled with the same toolchain and locked dependency file. The local `rust-objcopy` could not load its LLVM library while stripping debug information; Cargo continued with a warning, built usable binaries, and all numerical tests ran. This was not a compile failure or a skipped test. The logs retain the warning.

Minimal released-wheel reproduction:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install pyxirr==0.10.8
.venv/bin/python minimal_repro.py
```

For the complete numerical grid, install `requirements.txt` and run:

```sh
.venv/bin/python reproduce.py --output results/release.json --expect affected
```

To build either source snapshot, enter `pyxirr-baseline` or `pyxirr-corrected` with Cargo on PATH and use `maturin build --release --locked --interpreter /absolute/path/to/.venv/bin/python`. Install the resulting wheel into a separate environment and run `reproduce.py --expect affected` for the baseline or `--expect corrected` for the candidate. Use `verify_series_boundary.py` with the candidate installed. The bundled native wheels are for macOS arm64/CPython 3.12; other platforms require a fresh build.

`validate_native.py` is the exact local orchestration script and includes the local toolchain/cache paths; adapt those paths on another machine. The evidence contains complete source snapshots and `candidate.patch`. The frozen evidence archive was prepared before publication. This public edition accompanies that unchanged archive. The candidate correction has not been submitted or accepted upstream.


## Public files

[Complete evidence ZIP](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/gero-pyxirr-rate-evidence-2026-09-13.zip) · [English video](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/pyxirr-rate-vector-convergence.mp4) · [Video sources](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/pyxirr-rate-video-source.zip) · [GitHub reproducer](https://github.com/kadyrbekovhamit-cyber/gero-pyxirr-rate-audit).

Original graphics, synthetic Microsoft Andrew English narration and AI-assisted preparation. [Read the GERO article and watch the embedded video](https://www.gero.uz/research/articles/pyxirr-rate-vector-convergence.html).
