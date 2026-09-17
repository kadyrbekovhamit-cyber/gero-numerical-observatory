# actuarialmath Beta: a missing survival factor changes expected insurance benefits

**One synthetic term benefit produces 150,000 through `term_insurance` and 100,000 through the equivalent `A_x` call. The independently derived expected value is 75,000.** The root cause is the constant density supplied by `Beta`: it omits the time-dependent factor required by the class's survival law.

Independent research by Xamit Kadirbekov / GERO. First reported **15 September 2026**, re-executed and prepared for archival distribution **17 September 2026**. This is the archive of existing [issue #5](https://github.com/terence-lim/actuarialmath/issues/5), not a second discovery. The issue was open with no maintainer response at review; the correction below is a local proposal, not an accepted release.

## Exact example

Let the remaining lifetime follow the library's Beta mortality law with `omega=70`, `alpha=2`, current age 30, no selection offset, zero interest, a 20-year term and a benefit of 100,000 payable on death within the term. Remaining support is 40 years. Survival to year 20 is `(1−20/40)^2 = 1/4`, so the death probability is 3/4. With no discounting, the expected benefit is exactly **100,000 × 3/4 = 75,000**.

| Quantity | Original source and release 1.1.0 | Independent expectation | Local candidate |
|---|---:|---:|---:|
| Survival to year 20 | 0.25 | 0.25 | 0.25 |
| Density at year 20 | 0.05 | 0.025 | 0.025 |
| Whole-life expected benefit | 200,000 | 100,000 | 100,000.00000000001 |
| 20-year term via `term_insurance` | 150,000 | 75,000 | 75,000.00000000001 |
| Same term via `A_x` | 100,000 | 75,000 | 75,000.00000000001 |

These are expected benefits under explicitly chosen mathematical assumptions, **not commercial insurance quotations or measured losses**. A probability density may exceed one; the relevant invariant here is that its integral over the full support must equal one.

The actual public API calls are:

```python
from actuarialmath import Beta
life = Beta(omega=70, alpha=2).set_interest(i=0)
print(life.p_r(30, t=20))
print(life.f_r(30, t=20))
print(life.whole_life_insurance(30, b=100000, discrete=False))
print(life.term_insurance(30, t=20, b=100000, discrete=False))
print(life.A_x(30, t=20,
               benefit=lambda age, t: 100000, discrete=False))
```

`minimal_repro.py` evaluates these calls and obtains the expected values independently with Python `Fraction`.

## Contract and implementation

For remaining support `L = omega − (x+s+r) > 0` and interior `0 ≤ t < L`, the survival law is

`S(t) = (1 − t/L)^alpha`.

Differentiation gives

`f(t) = −S′(t) = alpha/L × (1 − t/L)^(alpha−1)`.

This also satisfies `f(t) = S(t) × mu(t)` and integrates to one. At zero interest a constant death benefit has term expectation `b × (1−S(t))` and whole-life expectation `b`.

The nested `_f` in `Beta.__init__` instead returns only `alpha / (omega − (x+s))`. Its integral over the remaining lifetime is **alpha**, not one. It happens to be correct for the uniform special case `alpha=1` and at `t=0`. Both the equivalent-method disagreement and the wrong whole-life value follow from this same missing factor; they are not counted as separate defects. The public fractional-age wrapper already passes the appropriate shifted selection age.

The candidate changes only that nested density function:

```python
def _f(x: int, s, t: float) -> float:
    remaining = omega - (x+s)
    return (alpha / remaining
            * ((remaining - t) / remaining)**(alpha - 1))
```

No insurance shortcut, survival function, hazard function, variance formula or oracle is patched.

## Executed evidence

The fixed grid was specified before the correction and is retained in `VALIDATION_PROTOCOL.md`. Four separate processes execute pristine source, the candidate, the source with the original formula restored, and the official wheel. This uses the real extracted package with its import location checked. The primary expectations use **80-digit mpmath** with the exact binary64 input values. Numerical integration of the density is supplementary.

| Check group | Cases | Original failures | Candidate failures | Restored failures | Release failures |
|---|---:|---:|---:|---:|---:|
| Pointwise density | 960 | 640 | 0 | 640 | 640 |
| Full-support density integral | 96 | 80 | 0 | 80 | 80 |
| Insurance scenario: whole, term and direct values | 864 | 720 | 0 | 720 | 720 |

The groups overlap and describe **one implementation defect**. The insurance count is scenarios with at least one failed value, not the sum of individual assertions. Equivalent term methods also disagree in the same 720 original scenarios and agree after the correction.

All 320 pointwise density controls (`alpha=1` or `t=0`) and all 144 uniform insurance scenarios remain exactly unchanged. All 960 survival/hazard controls and all 864 insurance survival/pure-endowment controls are unchanged. Every previously passing row still passes. SciPy reported no integration warnings in these recorded runs.

The grid includes shape parameters 0.5, 1, 1.5, 2, 3 and 5; remaining supports 4, 20, 40 and 80; ages 0 and 30; selection offsets 0 and 5; fractional-age offsets 0 and 0.5 for the pointwise checks; and benefits 1, 100 and 100,000. Density times include 0, 0.125, 0.5, 0.875 and 0.99 of the fractional remaining support. Insurance terms are 0.25, 0.5 and 0.75 of the remaining support, with zero interest throughout.

Pointwise comparisons use relative tolerance `2e-13`, absolute `1e-14`; the mass check uses absolute `2e-10`; insurance values use relative and absolute `2e-10`. The finite grid is not a proof of uniform numerical accuracy over all admissible parameters.

## Versions and reproduction

- Default-branch source, verified again September 17: [`7d18f11ad304898f177b7922b3c53f70e4c2b4f4`](https://github.com/terence-lim/actuarialmath/tree/7d18f11ad304898f177b7922b3c53f70e4c2b4f4). All 91 source-file identities match the official Git tree. The source's packaging metadata says 1.0.1.
- Official PyPI wheel: [actuarialmath 1.1.0](https://pypi.org/project/actuarialmath/1.1.0/), SHA-256 `b19990e4378aaa19fe6bc1182b4269faec6617cb62b0677fea1e624fbbb3ff6f`. PyPI hash and wheel RECORD entries verified. Its target source is identical.
- Source archive SHA-256: `0a0e98700ae483a390455251d9d4165d1744a2bac35c80a425381785c69333d3`.
- Target file SHA-256: `ae0a9757a1bb580b4f3a62b666afc28a387f02d8f3eec0b6475b86e35d885833`.
- Candidate patch SHA-256: `dfca779d2294aa3eb69912c1f44ba1a853af4a1f29916485c3227a5732e78f2f`.
- Python 3.12.14; NumPy 2.3.5, SciPy 1.16.3, pandas 2.3.3, matplotlib 3.10.8, mpmath 1.3.0, IPython 9.17.1. Configured single numerical thread; no GPU.

The archive includes the original source tarball and wheel, source hashes, candidate patch, minimal reproducer, fixed grid, raw results, comparison receipts, bounded prior-work review and a portable `reproduce.py`. The portable runner was separately executed from the packaged inputs; all numeric rows and summaries match the fresh recorded run. The September 17 numeric results also match the original September 15 experiment. Machine-specific path/timestamp metadata differs and is not claimed byte-identical.

See `README.md` for exact dependency installation and replay commands. Source and publication licenses remain separate in `LICENSES.md`.

## Prior work and limitations

All five public issue/PR records, target history, official release, and GERO/GitHub/Hugging Face catalogs were reviewed September 17. Our existing Beta [issue #5](https://github.com/terence-lim/actuarialmath/issues/5) is the same finding. The earlier [UDD density correction, PR #2](https://github.com/terence-lim/actuarialmath/pull/2), concerns a different class/formula; variance and ConstantForce findings are also separate. No second exact Beta report was found in this bounded review. Our Zenodo uploads search and both pages of the 50-item technical Shorts list showed no Beta publication. Search coverage is not proof that no unindexed report exists.

Direct evaluation at the terminal density singularity is excluded; for `alpha=0.5` the full-support integral is improper but converges in the recorded quadrature. Invalid parameters, extreme magnitudes, nonzero interest, discrete benefits, higher moments, variances, full upstream-suite compatibility, performance, calibrated mortality tables, production insurers and policyholder impact were not measured. The candidate is not presented as a certified actuarial engine or universally stable implementation.

The English Short explains these measurements with original diagrams and disclosed synthetic Jenny narration via edge-tts. Production checks cover timing, static decoded frames and full file decoding; no audio/video was played aloud and no listening or speech-recognition verification is claimed.

## Public evidence links

[GERO report](https://www.gero.uz/research/articles/actuarialmath-beta-density.html) · [GitHub evidence](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/actuarialmath-beta-density) · [Hugging Face mirror](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/actuarialmath-beta-density.md) · [Archival DOI](https://doi.org/10.5281/zenodo.22810874) · [29-second English video](https://youtube.com/shorts/faLB656CM4o).
