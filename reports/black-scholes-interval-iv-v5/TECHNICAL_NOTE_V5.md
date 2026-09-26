# Black-Scholes implied volatility as a range: exact quotes, rounding cells and auditable bounds

**Khamit Kadyrbekov and Daniyal Kadirbekov**

Working paper v5.0 | 26 September 2026 | DOI: [10.5281/zenodo.22976977](https://doi.org/10.5281/zenodo.22976977)

## 1. Result and scope

A rounded option price need not identify one volatility. We provide a small
Python research utility that maps an exact price interval or a binary64
rounding cell to an outer enclosure of the compatible Black-Scholes-Merton
(BSM) volatilities. Its output distinguishes an empty set, zero volatility,
unbounded ranges, expiry and insufficient computational budgets. It retains
exact rational endpoints and dyadic witnesses for definite comparisons.

For a call with S=200, K=100, T=1 and r=q=0, the binary64 observation 100 is
compatible with every volatility from zero to approximately **8.9407687355%**.
We enclose the upper endpoint in an interval narrower than 10^-24 in sigma.
This refines the illustrative information-loss example in our
[preceding working paper](https://doi.org/10.5281/zenodo.22976212).

The final fresh confirmation has **64/64 scenarios passing all declared
gates**: 56 resolved outcomes and eight deliberately insufficient budgets
handled explicitly. Earlier confirmation runs scored **158/160 and 82/96**
because of test-generator/reference errors; those results are retained, not
relabelled. The candidate remained unchanged throughout all three runs.

This is an engineering artifact using standard BSM mathematics and Arb
enclosures. It is not a new market model, priority claim, performance record
or independently verified formal proof. Quoted ranges are conditional on
exact model parameters and the specified interpretation of the price.

## 2. Why inversion produces a set

Put A=S exp(-qT), B=K exp(-rT). For T>0, the call has intrinsic lower limit
L=max(A-B,0) and unattained upper limit U=A; for the put, L=max(B-A,0) and U=B.
For positive sigma, model Vega is strictly positive:

    dP/dsigma = A sqrt(T) phi(d1) > 0.

Thus P is continuous on sigma>=0, strictly increasing, with attainable range
[L,U). Given an exact quote interval Q, the quantity of interest is

    Sigma(Q) = { sigma >= 0 : P(sigma) belongs to Q }.

Intersect Q with [L,U). An empty intersection means no finite compatible
volatility. A sole included price L gives {0}. Every interior endpoint has a
unique inverse, and its open/closed status is inherited. If the quote extends
to U, the volatility set has no finite upper bound even when U itself is
excluded. At T=0, the payoff is independent of volatility: either all
nonnegative volatilities are compatible or none are.

For a positive finite binary64 observation p, the real inputs rounding to p
are bounded by midpoints with its adjacent representable values. Both ties
belong to the cell exactly when p has an even least-significant significand
bit. At zero, the nonnegative cell is [0, 2^-1075]. At the largest finite
number, the overflow midpoint uses the virtual successor 2^1024 and is
excluded. These cells assume round-to-nearest, ties-to-even. Decimal quote
intervals and binary64 cells are different input contracts.

## 3. Computation and meaning of the certificate

The implementation evaluates the ordinary unspecialized BSM erfc expressions
using python-flint 0.8.0's Arb arithmetic. Exact rational inputs are converted
to enclosing balls; exponential, logarithm, square root and erfc retain
enclosures. A comparison is used only when the entire price-minus-target
ball is below zero, above zero, or exactly zero. Unknown comparisons never
become equality. Arb's documented enclosure contract is an explicit
dependency; we have not independently verified its native implementation.

Root search doubles an upper trial until it proves a bracket, then bisects
using exact rational midpoints. Monotonicity and the two proved endpoint
signs imply root containment. Precision increases along a declared schedule.
An uncertain trial during initial bracketing is not a valid finite upper
bound: the result preserves a conservative half-line and an unresolved
status. After a bracket is established, exhaustion retains that bracket.

The JSON output stores exact rational outer bounds and the brackets locating
the exact mathematical endpoints. Endpoint closure flags describe those
mathematical endpoints, not rounded bracket endpoints. `resolved=true`
means endpoint location meets the requested absolute sigma tolerance; it
does not mean the quote uniquely identifies sigma. A finite mathematical
range may have an infinite *reported outer bound* if the search budget is
exhausted. The separate `unbounded` field describes the mathematical case.

Resource limits include rational bit size, precision at most 4096 bits,
at most 2048 bisections and a sigma search cap. Default endpoint tolerance
is 10^-12. A single process and `ctx.threads=1` are required because precision
context is global. These limits are not a universal runtime or memory SLA.

## 4. Worked examples

The four post-confirmation illustrations have their own receipt and are not
counted as new confirmation. All their retained witnesses pass the separate
high-precision control. For the rounded ITM call of 100, the exact upper
endpoint lies strictly between these outward decimal bounds:

    0.0894076873552443926268368
    0.0894076873552443926268378

The underlying rational bracket has width below 10^-24 and required up to
256-bit arithmetic. Hence the six volatilities 3%, 4%, 5%, 6%, 7% and 8% in
v4 are part of a continuum of compatible answers, not six isolated cases.

For the corresponding OTM put, a binary64 observation generated at sigma=5%
still retains the tiny positive time value. Its returned outer enclosure is
contained between 0.0499999999999999999769 and
0.0500000000000000000312. This contrasts two storage representations of
synthetic model prices; it does not imply that real markets quote such tiny
prices or that put and call bid/ask spreads are interchangeable.

A simpler call example S=K=100, T=1, r=q=0 and exact price band [7,9] yields
sigma approximately [0.175689675792, 0.226077081290]. Tightening numerical
tolerance cannot remove the width caused by that price band. A fourth
illustration near sigma=0.5 forces escalation from 64 to 512 bits. The main
confirmation corpora needed at most 128 bits; precision escalation is tested
separately, not implied by their counts.

## 5. Evidence, including the failed checks

Every confirmation corpus and its candidate, generator, runner, reference,
protocol and dependency hashes were fixed locally before viewing candidate
outputs on that corpus. The cases are deterministic synthetic tests, not
market samples, random population estimates or external blind evaluation.
The primary implementation hash is
`2519288a7ee616274ad7f87f18ec2d6021e4407b4bc7272b1674bcb7b252985a`.

| Frozen run | Scenarios | All original gates pass | Retained definite witnesses |
|---|---|---|---|
| IV v1 | 160 | 158 | 1108 |
| IV v2 | 96 | 82 | 673 |
| IV v3 | 64 | 64 | 417 |

In v1, converting an mpmath value directly to float selected a neighbouring
subnormal rounding cell in one generator case. The discrepancy persisted
at 180, 260 and 360 decimal digits; Fraction-mediated conversion and a
1024-bit Arb check diagnosed it. Another scenario incorrectly expected
boundary ambiguity, while Arb proved a sign; two control convergence checks
also suffered cancellation of a tiny intrinsic value. The frozen failures
remain visible. No maintainer defect or novelty claim is made here.

The v2 control used expm1 for tiny intrinsic differences, but unnecessarily
passed exact zero-carry differences through log/exp. Tiny oracle rounding
errors then failed exact, zero-width witness checks in 14 scenarios. The v3
control preserves exact rational S-K in those cases. Rechecking the earlier
1781 definite witnesses with this control passed, explicitly as **post hoc
diagnosis**, not as replacement confirmation. No solver change was needed.

The final 64 new scenarios are disjoint from both earlier corpora: 32 rounded
quote recovery cases, 16 rounded boundaries, eight explicit budget limits
and eight expiry cases. All 417 retained definite witnesses have valid
dyadic signs and contain the separately computed 260-digit value, with the
declared 180/260 convergence checks. All 32 latent sigmas lie in the returned
outer ranges. The eight budget cases remain unresolved as intended.
Development has 24 solver test methods plus three reference-identity methods;
v1 also passes eight parity/scaling checks. Failed development logs remain.

## 6. Reproduce and inspect

Use Python 3.12 with `python-flint==0.8.0` and `mpmath==1.3.0`; no GPU is
required. The distributed package contains original source and evidence,
not native dependency binaries. Set numerical thread counts to one. From
the extracted release, using fresh output paths:

```bash
python -m unittest discover -s iv-tests -v
python -m unittest discover -s iv-tests-v3 -v
python -m lab.run_iv_v3 --portable --output /tmp/iv-replay.json
python iv_range.py --spot 200 --strike 100 --time 1 --rounded-price 100
```

The portable runner checks source/input hashes and dependency versions;
platform-specific native library hashes may differ. The original native
hashes are retained for audit. Replaying `lab.run_iv` and `lab.run_iv_v2`
reproduces their original failed gates and intentionally exits nonzero.
Reproduction on this host is not independent human review.

Code and complete evidence: [GERO Numerical Observatory, interval-IV v5](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/reports/black-scholes-interval-iv-v5).

## 7. Limitations, related work and authorship

The set inversion follows elementary continuity and monotonicity; enclosure
arithmetic and bracketing are established methods. Arb is described by Fredrik
Johansson, *Arb: Efficient Arbitrary-Precision Midpoint-Radius Interval
Arithmetic*, [arXiv:1611.02831](https://arxiv.org/abs/1611.02831).
[python-flint documentation](https://python-flint.readthedocs.io/en/latest/arb.html)
describes its real-ball operations; experiments pin version 0.8.0, while the
current documentation displays 0.9.0. We do not claim to have reproduced the
benchmarks in Wolfgang Schadner's [explicit IV representation](https://arxiv.org/abs/2604.24480v4).
That work studies a representation of the inverse; our focus is retaining
uncertainty in the observed price. No superiority comparison is made.

Uncertain spot, rates, dividends, time, exercise features, volatility smiles,
transaction costs and model misspecification are outside this API. A bid/ask
interval is not automatically a statistical confidence interval. Production
integration, specialist solver comparisons and external review remain open.

Khamit Kadyrbekov initiated and directs this programme. Daniyal Kadirbekov
is coauthor and designated lead for continuing software development; this
does not assert independent implementation or validation by him. OpenAI
Codex materially assisted derivation, implementation, experiment design,
execution, diagnosis and drafting. No university affiliation, endorsement
or independent human peer review is claimed. Computation was sequential on
CPU with one numerical worker. Original text/data: CC BY 4.0; code: MIT.
