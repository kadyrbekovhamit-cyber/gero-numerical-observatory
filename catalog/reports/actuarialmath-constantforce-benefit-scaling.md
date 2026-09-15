# ConstantForce whole-life insurance ignores the benefit amount

Research result verified 15 September 2026. Evidence edition prepared for publication on 15 September 2026.

`ConstantForce.whole_life_insurance(..., discrete=False, moment=1 or 2)` in
`actuarialmath` returns the moment for a unit benefit even when the caller passes
a different benefit `b`. The same class's `term_insurance(t=WHOLE, ...)` includes
the benefit and returns the expected result.

The recorded current-source pin is
`7d18f11ad304898f177b7922b3c53f70e4c2b4f4`. The published PyPI wheel is version
**1.1.0**, SHA-256
`b19990e4378aaa19fe6bc1182b4269faec6617cb62b0677fea1e624fbbb3ff6f`.
Its `constantforce.py` is byte-identical to the pinned current source. Both were
executed, and their 144 main observation rows match exactly. The current source's
project metadata still says 1.0.1; the source pin and wheel identity distinguish
the two distributions.

## Minimal example

```python
from actuarialmath import ConstantForce

life = ConstantForce(mu=0.02).set_interest(delta=0.03)
print(life.whole_life_insurance(35, b=100000, discrete=False))
print(life.term_insurance(35, t=life.WHOLE, b=100000, discrete=False))
```

Recorded outputs are approximately **0.4** and **40,000**. These calls describe
the same continuous whole-life benefit under the class's constant mortality
assumption. Changing age from 35 to 70 leaves the results unchanged, as this
memoryless lifetime model requires.

For the second moment at the same parameters and benefit, the whole-life call
returns approximately **0.25**, while the expected answer is **2,500,000,000**
in squared monetary units. A zero benefit also incorrectly returns a positive
unit-benefit moment.

## Independent oracle and candidate

With a lifetime `T ~ Exponential(mu)` and discounted benefit
`Z = b * exp(-delta*T)`, direct integration gives

```
E[Z**m] = b**m * mu / (mu + m*delta).
```

The main oracle uses exact rational arithmetic for the declared decimal inputs.
The source's shortcut omits `b**moment`. The candidate adds this single factor
in this branch. It does not change the generic variance branch or mortality
assumptions.

## Executed checks

- 144 observations: three positive mortality forces, four nonnegative interest
  forces, six benefit amounts and the first/second moments.
- Oracle failures: **120 original → 0 candidate → 120 original-formula mutation**.
  The released wheel reproduces the same 120 failures.
- All 144 `term_insurance(t=WHOLE)` controls satisfy the independent oracle.
- 288 finite-term controls, at terms 1 and 10, pass their analytical oracle and
  remain byte-for-byte identical in all four result sets.
- All 24 unit-benefit main observations remain unchanged. The age check passes
  throughout. These controls overlap in purpose; they are not separate customer
  trials.
- Fixed tolerances: relative `2e-12`, absolute `2e-14`; unchanged between runs.
- All 91 original source files retain their recorded hashes. Only
  `src/actuarialmath/constantforce.py` differs in the candidate. The candidate was
  restored after mutation replay.

Runtime: Python 3.12.14; NumPy 2.3.5, SciPy 1.16.3, pandas 2.3.3, matplotlib
3.10.6. The source also imports IPython, which was installed in this isolated
environment; the full resolved dependency set is in `environment-requirements.txt`.
Numerical library thread counts were set to one. The full upstream test suite
was not run. A separate portable runner verified the bundled source/wheel hashes,
extracted fresh copies and reproduced every recorded main and finite-term numeric
row exactly. The original evidence files were preserved.

## Duplicate review and limits

Five saved GitHub issue/PR searches and the canonical 95-publication GERO
catalog were checked on 15 September 2026. The related existing report,
[issue #3](https://github.com/terence-lim/actuarialmath/issues/3), concerns the
generic whole-life variance formula squaring the first moment twice. This
finding concerns the positive-moment shortcut in `ConstantForce` and a missing
benefit factor; it is not a new version of that variance finding. No exact
duplicate was found within the recorded search scope. This is not a guarantee
of global novelty.

These are synthetic actuarial calculations on a public educational library.
No insurer deployment, premiums charged, reserves booked, customer losses or
production exposure were measured. Variance and other shortcut candidates are
outside this correction's scope.

Primary contract references:
[source](https://github.com/terence-lim/actuarialmath/blob/7d18f11ad304898f177b7922b3c53f70e4c2b4f4/src/actuarialmath/constantforce.py),
[official guide](https://actuarialmath-guide.readthedocs.io/en/latest/constantforce.html),
[PyPI release](https://pypi.org/project/actuarialmath/1.1.0/).

Research: Xamit Kadirbekov / GERO. AI-assisted preparation, with executed public
Python code and an independent mathematical oracle. Upstream MIT license is
preserved in the source archive.

## Reproduction and evidence

[Portable source, raw results, independent oracle and candidate patch](../artifacts/gero-actuarialmath-constantforce-benefit-evidence-2026-09-15-v1.0.1.zip).

Evidence SHA-256: `f9c41cdf7e788b0aab2f3c1c4c52b26a6b63959a59cbda0d68893fb202bf3d1b`.

[Maintainer report #4](https://github.com/terence-lim/actuarialmath/issues/4) was submitted; upstream acceptance is not claimed.
