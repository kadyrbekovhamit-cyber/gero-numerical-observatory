# actuarialmath: the variance selector rejected before its own branch

Xamit Kadirbekov · GERO · 22 October 2026. Investigation and writing were AI-assisted.

## A minimal contradiction in the public API

`Uniform.E_x` contains a branch for `moment=VARIANCE`, whose sentinel is −2. An earlier assertion requires `moment > 0`, so a valid variance request never reaches the branch.

```python
from actuarialmath import Uniform
life = Uniform(omega=100).set_interest(i=0)
life.E_x(x=40, s=0, t=30, moment=life.VARIANCE)
# Actual: AssertionError. Independent reference: 0.25.
```

Under a uniform future lifetime with 60 years remaining, survival for another 30 years has probability 1/2. With unit benefit and zero interest, the pure-endowment value is a Bernoulli random variable: its variance is p(1−p) = 1/4.

## Version and verification

The current repository revision remains `7d18f11ad304898f177b7922b3c53f70e4c2b4f4` on 22 October. [Issue 11](https://github.com/terence-lim/actuarialmath/issues/11), opened 3 October, remains open with no comments. The stated 14-day interval ended 17 October; this broader publication follows a fresh source and response check.

The actual package API was replayed with real dependencies, without graphical or numerical stubs. Three remaining lifetimes, three interest rates and four terms produce **36 variance checks**; raw moments 1, 2 and 3 give **108 controls**.

All 36 variance checks fail before the narrow candidate and pass after it. All 108 controls pass in both versions. The candidate admits the variance sentinel and returns zero variance for a zero-term certain payment. The reference is independent 80-digit Decimal arithmetic: v^(2t)p(1−p) for variance, and v^(kt)p for raw moment k.

There is a separate base-class boundary: `Insurance.E_x` still returns 1 for zero-term variance. Consequently the full diagnostic reports **37 failures before and 1 after**, not zero. The candidate does not repair that distinct branch.

## Scope and repair status

All cases use selection duration s=0 to isolate this problem from the earlier [Uniform conditioning report](https://github.com/terence-lim/actuarialmath/issues/9). The full upstream suite and production applications were not tested. The candidate patch is available for review but has not been accepted or released. There is no claim of observed financial loss, security impact or global novelty.

The archive includes the runnable harness, candidate patch and fresh outputs. Original upstream code is MIT licensed; the report is CC BY 4.0 and original reproduction code is MIT.
