# actuarialmath continuous annuity: dropping elapsed years changes a value and a decision

20 September 2026 · Independent GERO research by Xamit Kadirbekov · AI-assisted investigation and preparation.

A continuous annuity factor that should be **8.75** becomes **9.166666666666666** in the public `actuarialmath` implementation. The function accepts years since selection, `s`, but substitutes zero in its continuous survival calculation. An equivalent attained-age input returns 8.75. This report documents one implementation defect, a narrow local candidate and an executed synthetic document/decision consequence.

## Minimal public-API example

```python
from actuarialmath import Uniform
life = Uniform(omega=100).set_interest(i=0)
print(life.a_x(40, s=20, t=10, discrete=False))  # 9.166666666666666
print(life.a_x(60, s=0, t=10, discrete=False))   # 8.75
```

The [annuity API](https://actuarialmath.readthedocs.io/en/latest/actuarialmath.annuity.html) defines `x` as selection age and `s` as elapsed years. The [survival API](https://actuarialmath.readthedocs.io/en/latest/actuarialmath.survival.html) carries both quantities into the survival function.

For this ultimate uniform-lifetime model, both inputs describe a person aged 60. Remaining lifetime is uniform from 0 to 40. At zero interest, a **continuous payment rate** of one unit per year, for up to ten years while alive, has expected present value

```text
integral from 0 to 10 of (1 - y/40) dy
= 10 - 100/80
= 8.75.
```

Using selection age 40 without its elapsed 20 years instead gives remaining lifetime 60 and `10 - 100/120 = 55/6`. Same-attained-age equivalence is asserted for this ultimate model, not for arbitrary select mortality models.

## Implementation and narrow candidate

[The pinned source](https://github.com/terence-lim/actuarialmath/blob/7d18f11ad304898f177b7922b3c53f70e4c2b4f4/src/actuarialmath/annuity.py) passes `s` in the discrete branch but calls `self.S(x, 0, t=t+u)` in the continuous integrand. The candidate changes only that survival argument:

```diff
- self.S(x, 0, t=t+u)
+ self.S(x, s, t=t+u)
```

Executed source: main `7d18f11ad304898f177b7922b3c53f70e4c2b4f4`. Separately executed release: official [PyPI 1.1.0](https://pypi.org/project/actuarialmath/1.1.0/), wheel SHA-256 `b19990e4378aaa19fe6bc1182b4269faec6617cb62b0677fea1e624fbbb3ff6f`. All 22 copied release Python modules were compared with that wheel. The affected file is identical in current and release snapshots.

## Independent oracle and checks

The main grid calls the real `Annuity.a_x` with a supplied uniform survival function. With `L = 100 - x - s`, payment rate `B`, deferral `u`, term `t` and force of interest `delta`, the continuous reference is

```text
B * integral from u to u+t of exp(-delta*y) * (1 - y/L) dy.
```

At zero force it is `B * [t - (u*t + t*t/2)/L]`. At positive force an antiderivative is `exp(-delta*y) * [(y-L)/delta + 1/delta²] / L`, multiplied by B. Discrete controls use an independent discounted finite sum. The reference uses 80-digit Decimal arithmetic, with 59 controls per variant repeated at 120 digits. Nominal decimal rates are compared with the library's float64 counterparts at fixed absolute tolerance `2e-12 * max(1, B*t)`; the reference does not represent exact binary64 input arithmetic.

The 2,160 cases per variant cover selection ages 20/40/60; elapsed years 0/1/5/10/20; terms 0/1/5/10; deferrals 0/2/7; forces 0/.01/.05; constant rates 1/1000; continuous and discrete branches. All tested intervals stay inside lifetime support.

| Executed variant | Failures / 2,160 |
| --- | ---: |
| Pinned current source | 648 |
| Official release 1.1.0 | 648 |
| Local one-line candidate | 0 |
| Original argument restored | 648 |

All 1,512 previously passing outputs are exactly unchanged in the tested environment. Separate executions of the package's `Uniform` class verify the minimal example. A fresh portable packet replay applies and reverses the patch, verifies the source/wheel hashes, and reproduces all four numerical grids, Uniform controls, written memos and read-back decisions.

Environment: Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6, matplotlib 3.10.8 and IPython 9.17.1; one configured numerical worker, CPU only. No full upstream test-suite or performance claim.

## Executed formula → document → decision chain

For the same example with a continuous payment rate of 1,000 units per year, a GERO adapter writes a JSON expected-value memo. A separate consumer reads the stored rounded amount and compares it with a synthetic ceiling of 9,000.

| Stage | Original / release / restored | Local candidate |
| --- | ---: | ---: |
| Calculated expected present value | 9,166.666666666668 | 8,750.0 |
| Amount written in JSON | 9,166.67 | 8,750.00 |
| Read-back ceiling decision | OVER_LIMIT | WITHIN_LIMIT |

The displayed amount changes by **416.67 units**. The pricing formula is real library code. Memo creation, rounding, the ceiling and the consumer are explicitly GERO demonstrations. This is **not** an insurer's quote, a required reserve, an actual underwriting decision or evidence of customer loss.

## Prior art, disclosure and limits

The bounded duplicate review inspected all pre-existing six issue/PR bodies and four comments, the complete PR #2 diff, eight `annuity.py` history records, the guide repository's issues, focused searches and GERO catalogues. A fresh check before distribution found unchanged main, seven issue/PR records including our own #7, and no additional PR or response. No exact earlier report or correction was found in that scope; no global priority claim is made. The source pattern dates to 2023, so this is not presented as a recently introduced regression. The internal static lead preceded its executed confirmation.

[Developer issue #7](https://github.com/terence-lim/actuarialmath/issues/7) was submitted and its exact body verified before publication. No upstream acknowledgment, accepted patch or released fix is claimed.

The correction is limited to this argument in `Annuity.a_x`. Other annuity convenience methods, arbitrary survival laws, variable benefits, whole-life limits and negative forces are not validated here. In particular, `Uniform.temporary_annuity` gives a separate unexpected zero-rate result that remains unchanged; it is not used as an oracle or described as repaired. Other initial candidates are excluded. No real insurer system, policyholder record, deployment, loss or regulatory breach was tested.

Original report: CC BY 4.0. GERO scripts and candidate: MIT. Upstream source retains Terence Lim's MIT license. See the complete evidence archive, source ledger, executable runner, raw outputs and explicit environment requirements.
