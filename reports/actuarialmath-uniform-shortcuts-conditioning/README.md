# `actuarialmath`: Uniform shortcuts use the wrong conditioning age and reverse limited-life weights

Status: **confirmed locally; reported to the maintainer**. The bounded novelty review does not establish absolute priority. Developer report: https://github.com/terence-lim/actuarialmath/issues/9.

## Affected source

- Upstream: `terence-lim/actuarialmath`
- Current remote `main`: `7d18f11ad304898f177b7922b3c53f70e4c2b4f4`
- Official release copy: PyPI 1.1.0, taken from the independently hash-verified
  wheel retained by the earlier actuarial audit
- File: `src/actuarialmath/mortalitylaws.py`
- The current and release target files are byte-identical, SHA-256
  `ae0a9757a1bb580b4f3a62b666afc28a387f02d8f3eec0b6475b86e35d885833`.

## Mathematical discrepancy

For De Moivre's law at selection age `x`, duration `s` and limiting age
`omega`, the remaining lifetime is uniform on `[0, L]`, where

```text
L = omega - (x + s).
```

Therefore:

```text
E[T]              = L / 2
Var[T]            = L^2 / 12
E[min(T, t)]      = t - t^2/(2L),                 0 <= t <= L
Pr[T > t]         = (L - t)/L.
```

The current `Uniform.e_x` complete shortcuts use `omega - x`, discarding `s`.
Its limited expectation computes

```python
t_p_x = t / (omega - x)
return t_p_x * t + (1 - t_p_x) * (t / 2)
```

but `t/(omega-x)` represents death within the term.  The death branch should
receive the conditional mean `t/2`, while the survival branch should receive
the full term `t`.  The two weights are reversed.  `Uniform.E_x` also uses
`omega-x` instead of `omega-(x+s)`.

## Minimal public examples

At `omega=100`, `x=40`, `s=0`, `t=10`, zero interest and a continuous unit
payment, the exact value is

```text
integral_0^10 (1 - u/60) du = 10 - 100/120 = 55/6
                              = 9.166666666666...
```

Observed results:

| Public call | Current / 1.1.0 | Exact | Candidate |
|---|---:|---:|---:|
| `e_x(40, t=10, curtate=False)` | 5.833333333333334 | 9.166666666666667 | 9.166666666666668 |
| `temporary_annuity(40, t=10, discrete=False)` | 5.0 | 9.166666666666667 | 9.166666666666664 |
| `E_x(40, s=15, t=10)` | 0.8333333333333334 | 0.7777777777777778 | 0.7777777777777778 |

The first two failures occur even at `s=0`; they are therefore distinct from
the previously reported continuous `Annuity.a_x` selection-duration defect.

## Candidate correction

The isolated candidate:

1. computes `remaining = omega - (x+s)` in `Uniform.e_x` and `Uniform.E_x`;
2. uses `remaining/2` and `remaining^2/12` for complete moments;
3. assigns the `t/2` value to the death-within-term probability and `t` to
   survival in the limited expectation;
4. computes the pure-endowment survival probability as
   `(remaining-t)/remaining`.

The exact patch is [candidate.patch](candidate.patch).

## Executed verification

Each variant was imported in a separate process through the actual public
classes.  The grid covers:

- `x = {10, 40, 60}`;
- `s = {0, 5, 15}` with `x+s < 100`;
- `t = {0, 1, 5, 10, 20}` inside the lifetime support;
- force of interest `delta = {0, 0.01, 0.05}`;
- pure-endowment raw moments 1 and 2;
- continuous temporary annuities with unit payments.

Independent 80-decimal closed forms produced 468 comparisons per variant:

| Check family | Checks | Current failures | Candidate failures | Restored | Release 1.1.0 |
|---|---:|---:|---:|---:|---:|
| Complete expectation | 9 | 6 | 0 | 6 | 6 |
| Complete variance | 9 | 6 | 0 | 6 | 6 |
| Limited expectation | 45 | 35 | 0 | 35 | 35 |
| Pure-endowment raw moments | 270 | 144 | 0 | 144 | 144 |
| Continuous temporary annuity | 135 | 84 | 0 | 84 | 84 |
| **Total** | **468** | **275** | **0** | **275** | **275** |

The full grid was repeated at 120 decimal digits with the same counts.  All
468 restored outputs and all 468 release outputs are bit-identical to current.
Of 193 originally passing controls, 174 remain bit-identical under the
candidate; the remaining 19 change only through algebraic evaluation order,
by at most `3.552713678800501e-15`, and all still match the oracle tolerance.
The saved patch also applies to a fresh copy and reverses to the original
SHA-256 exactly; see `evidence/PATCH_REPLAY.json`.

## Duplicate review

The review covered all eight current upstream issue/PR records, their five
comments, the relevant source history and focused GitHub issue/code searches.
No issue or correction for these expressions was found.  Exact code search for
the limited-expectation denominator returned only the upstream source file.
The expression dates to commit `a5d2797ac722b6abcb2d1459311b85c20130a018`
from 4 June 2023, so this is not described as a newly introduced regression.

Issue #7 reports a different `Annuity.a_x` call-site defect and explicitly left
`Uniform.temporary_annuity` untriaged.  The earlier published report likewise
excluded this shortcut.  Absolute novelty is not claimed.

## Limits

- This verifies `Uniform` shortcut formulas for the stated finite domain; it is
  not a proof for every mortality law or contract method.
- Unit annuity benefits were used to isolate this defect.  A separately observed
  zero-interest benefit-scaling discrepancy is excluded.
- The discrete zero-interest path and pure-endowment variance selector expose
  separate exceptions and are excluded pending independent analysis.
- Optional plotting and notebook-display modules were stubbed at import time;
  the tested numerical methods do not call them.
- The complete upstream test suite was not run.
- No insurer deployment, policy record, reserve, customer loss, regulatory
  breach or bounty eligibility was tested.

## Developer disclosure

The reproducer, formulas, candidate diff and validation totals were submitted as [actuarialmath issue #9](https://github.com/terence-lim/actuarialmath/issues/9) before broad distribution. No maintainer response, accepted patch or released correction is claimed. The investigation and report were AI-assisted; executable evidence and independent closed-form checks are retained.

## Public records

- [Developer issue](https://github.com/terence-lim/actuarialmath/issues/9)
- [GitHub report](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/actuarialmath-uniform-shortcuts-conditioning.md)
- [Complete evidence archive](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/raw/refs/heads/main/reports/actuarialmath-uniform-shortcuts-conditioning/gero-actuarialmath-uniform-shortcuts-conditioning-evidence-2026-09-30.zip) — SHA-256 `ec9f55b79b3a9f35e16ed984ec923536b0b128ba2adbde0eedd3b37c36b4b36f`

GERO, Hugging Face, Zenodo, LinkedIn and media links are added only after each destination is independently verified.
