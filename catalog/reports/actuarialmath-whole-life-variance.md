> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/actuarialmath-whole-life-variance.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# actuarialmath whole-life variance squares the mean twice

Xamit Kadirbekov / GERO Research · 14 September 2026

The discrete whole-life variance path returns an incorrect variance because it squares the first moment before passing it to a helper that squares it again. A one-line candidate correction removes that extra square. This report describes executed public source code and synthetic inputs; no insurer deployment, customer loss or production exposure was measured.

## Version and scope

Repository: [terence-lim/actuarialmath](https://github.com/terence-lim/actuarialmath). Pinned main: `7d18f11ad304898f177b7922b3c53f70e4c2b4f4`, project metadata 1.0.1. Current main was rechecked before publication on 14 September and was unchanged. This was a fresh source checkout imported through its `src` directory, not a separately tested PyPI release. Dependency versions and actual import paths are retained.

## Two outcomes, an exact oracle

Take a synthetic lifetime table with death probability 0.25 in year one and certainty of death in year two conditional on surviving year one. Pay a unit benefit at the end of the year of death; the annual effective interest rate is 5%.

Let `v = 1 / 1.05`. The discounted benefit is `v` with probability 0.25 and `v²` with probability 0.75. Its variance is exactly `0.25 × 0.75 × (v − v²)²`.

```python
from actuarialmath import LifeTable
life = LifeTable().set_table(q={40: 0.25, 41: 1}).set_interest(i=0.05)
print(life.whole_life_insurance(40, moment=life.VARIANCE))
```

| Quantity | Expected | Original source |
|---|---:|---:|
| Mean | 0.9183673469387755 | 0.9183673469387754 |
| Second moment | 0.8437842257084238 | 0.8437842257084238 |
| Variance | 0.000385641785058695 | 0.13246305434448635 |

The incorrect variance also exceeds `(maximum − minimum)² / 4`, the maximum possible variance for a random variable confined to that interval. This is a mathematical inconsistency under ordinary finite inputs, not a claim about real mortality experience.

## Cause and candidate correction

[Insurance.whole_life_insurance, line 111](https://github.com/terence-lim/actuarialmath/blob/7d18f11ad304898f177b7922b3c53f70e4c2b4f4/src/actuarialmath/insurance.py#L111) computes `A1 = whole_life_insurance(...)**2`. The helper then calculates `b**2 * max(0, A2 - A1**2)`. For the unit discounted benefit Y, the composition uses `E[Y²] − E[Y]⁴`, rather than `E[Y²] − E[Y]²`, before applying the squared benefit scale and the nonnegative clamp.

The candidate removes only the caller's extra `**2`. It does not change the helper, mean, second moment or nonnegative clamp. The attached patch also adds 16 independently derived regression cases. No upstream acceptance is claimed.

## Executed checks

An independent `fractions.Fraction` oracle enumerates 96 cases: six probabilities (0, 0.01, 0.25, 0.5, 0.9, 1), four rates (0, 0.01, 0.05, 0.2), and four benefits (0, 1, 100, 100000). The absolute tolerance, fixed before evaluating the candidate, is `32 × machine_epsilon × max(1, benefit²)`.

| Check | Original | Candidate |
|---|---:|---:|
| Incorrect variances / 96 | 54 | 0 |
| Variance range-bound violations / 96 | 54 | 0 |
| Incorrect means / 96 | 0 | 0 |
| Incorrect second moments / 96 | 0 | 0 |
| New regression tests passing / 16 | 0 on mutation replay | 16 |

Restoring the original production file reproduced all 96 original result rows exactly and failed all 16 new tests by numerical assertions. Restoring the candidate made all 16 pass again. The existing `tests/test_changes.py` example program also completed successfully; this is not a claim that the project has a full passing test suite. One existing SciPy deprecation warning was recorded.

## Duplicate review and limits

The public upstream index contained two open/closed issue or PR records: #1 concerns UDD m-thly documentation; #2 is the prior fractional-age UDD density correction. Neither is this cause. A GitHub issues search for `"actuarialmath" "whole_life_insurance" variance` returned zero matches with `incomplete_results=false`; the guide repository and local GERO registry were also checked. The source and upstream index were refreshed before publication. This bounded search found no direct duplicate; it does not prove priority or cover private/deleted reports.

The measured scope is the discrete whole-life path via LifeTable, on a two-year finite support. Other insurance classes, real reserves, premiums and actual insurer systems were not evaluated. Existing dependency environments were reused; no fresh-machine validation is claimed. Static observations about other methods are not included as findings.

## Reproduction and rights

The frozen ZIP contains the upstream source subset and MIT license, candidate patch, exact oracle, 96-case outputs, regression XML and mutation evidence. Extract it and follow its README. Preparation-stage local/unpublished labels in that immutable archive describe the earlier research state; this is the public report edition. The original ZIP is unchanged, SHA-256 `678b8a2efad2a7db6c31ab416fa2dbeaa4d9406d16a5235a5bc08ed9ebf1c52b`.

Independent GERO research, assisted by AI for analysis and preparation. Video diagrams are original; narration uses the synthetic Microsoft Jenny English preset through [edge-tts](https://github.com/rany2/edge-tts). No affiliation with the project author or any insurer is implied.
