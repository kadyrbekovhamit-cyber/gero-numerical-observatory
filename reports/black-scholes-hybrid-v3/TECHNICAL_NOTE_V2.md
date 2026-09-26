# Positive-payoff quadrature for Black–Scholes: a frozen binary64 prototype

Khamit Kadyrbekov and Daniyal Kadirbekov

Independent research · Working paper v2.0 · 26 September 2026

## Result and scope

We replaced the cancellation-prone branches of our experimental v0 evaluator
with a separate binary64 positive-integral implementation, candidate v2. On
**288 new confirmation inputs**, all predeclared practical price/log gates
passed, with no evaluation exception, wrong OTM side, negative/nonfinite price,
or unexpected zero. Among **202** prices whose converged reference rounds to a
nonzero binary64 value, v2 was closer than v0 in 192 cases, farther in 3, and
equal in 7. Against the unchanged historical Jäckel source, it was closer in
146, farther in 30, and equal in 26.

This is an improvement over our previous numerical prototype on these tests.
It is **not** a new BSM model, a proof of universal superiority, a novelty claim,
or a demonstrated improvement in market prediction. The corpus deliberately
emphasizes near-ATM cancellation and range limits; these counts are not market
failure probabilities. A small timing check found the Python v2 detailed API
about **15.4 times slower** than the Python v0 detailed API on the measured
development subset. Their achieved accuracy differs, and C++ speed was not timed.

The preceding audit and its negative result remain in
[`TECHNICAL_NOTE_V1.md`](TECHNICAL_NOTE_V1.md).

## 1. Shared mathematical representation

For the same exact-binary64 input contract as v1, put
\(m=\log(S/K)+(r-q)T\), \(v=\sigma\sqrt{T}\), and
\(z=v/2-|m|/v\). The OTM leg is call for \(m\le0\), put otherwise;
its discounted cash leg is \(A=S e^{-qT}\) or \(A=K e^{-rT}\), respectively.
For positive total volatility,

\[
V_{OTM}=\frac{A}{\sqrt{2\pi}}
\int_0^\infty e^{-(u-z)^2/2}(1-e^{-vu})\,du.
\]

The integrand is nonnegative. This representation removes the subtraction of
two nearly equal CDF/Mills terms. It was already used as a cross-formula check
in v1; here it is implemented as an ordinary binary64 candidate. No
multiprecision arithmetic or oracle fallback occurs inside v2.

For \(z\le0\), set \(L=\max(1,-z)\), \(b=v/L\), and \(u=w/L\).
We integrate

\[
I=\int_0^{64}
e^{(z/L)w-w^2/(2L^2)}
\frac{-\operatorname{expm1}(-bw)}{b}\,dw
\]

and reconstruct \(V=A\varphi(z)(v/L^2)I\). When rounded \(b=0\), the
integrand factor takes its limit \(w\); the external \(v\) is retained.
The omitted integral is bounded in exact arithmetic by \(65e^{-64}\) for
\(z\le-1\), or \(e^{-64^2/2}\) for \(-1<z\le0\).

For \(z>0\), we integrate the completed-square Gaussian on
\([\max(0,z-12),z+12]\), dividing the payoff factor by \(v\) and restoring
\(v\) outside. The integrand is bounded by \(u e^{-(u-z)^2/2}\), giving an
omitted-integral bound \((1+c z/12)e^{-72}\), where \(c=1\) if \(z\le12\)
and \(c=2\) otherwise. Bounds are evaluated in log form and checked relative
to the estimated integral. Floating-point evaluation of these bounds is not
directed interval arithmetic.

Both branches use adaptive Gauss–Legendre 16/32 panels. We retain positive
panel sums, allocate an absolute budget across panels, and include a binary64
roundoff floor. Depth and evaluation budgets are finite; exhaustion raises an
explicit exception. The node-rule difference estimates integration error; it
does not bound all input, elementary-function, or rounding errors.

## 2. Input preparation and reconstruction

Near ATM, v2 computes `log1p((S-K)/K)`, retaining the small input difference.
For separated values it uses `frexp` mantissas/exponents, avoiding overflow in
`S/K`. Carry uses `(r-q)*T` and `fsum` when adding log moneyness. These
operations still round; we make no general sign guarantee at arbitrary
near-exact carry cancellation.

Ordinary and log prices use the same computed integral and the same side.
The ordinary result retains the original monetary mantissa and accumulates
power-of-two exponents before final `ldexp`; an intermediate normalized price
is never required to fit into binary64. The log result is a second view of
these factors, not a separately selected pricing formula. We do not require
bitwise equality to `exp(rounded_complete_log_price)`, which can itself lose
accuracy from rounding a large monetary logarithm.

Intrinsic value uses `expm1` or the exact near-input cash difference where
applicable; the ITM leg is reconstructed by positive addition. The source
does not import mpmath, call native Jäckel, or read reference data.

The API explicitly rejects a numerically unsupported range: positive
`sigma,T` with `sigma*sqrt(T)` rounded to zero. For example, with
`S=K=1e200`, `sigma=1e-300`, `T=1e-100`, the true ATM time value is approximately
3.99e-151 even though ordinary binary64 total volatility rounds to zero.
Returning a deterministic payoff would be wrong. True zero volatility or
zero maturity remains supported. Nonfinite intermediate log arithmetic and
insufficient integration budgets also fail explicitly.

## 3. Freeze and evidence

[`EXPERIMENT_V2.md`](EXPERIMENT_V2.md) defines the construction and gates.
All 316 old v1 cases were development data. A first development-only quadrature
attempt exhausted a purely local relative tolerance in 105 cases; that attempt
is preserved in `evidence/archive-v2-draft-01`. The global absolute budget fixed
this integration-control problem before confirmation.

The final candidate, its unit-test module, protocol and oracle were hashed
before any v2 confirmation output was computed. The candidate SHA-256 is
`536509e79dc52cd22af5b6dadbc4aeac9c46f573df30e522d91e6ba5a3572b78`.
The corpus SHA-256 is
`019016a0e6a8256c4290033780d43c1226ae1ffb7bcdc150b75ab23daf57d978`.
The freeze receipt is [`evidence/freeze-v2.json`](evidence/freeze-v2.json).
No candidate changes were made after observing confirmation results.

The freeze receipt did not include the complete executable pipeline: the final
runner hash is recorded in its output, while some imported helper modules were
not in the pre-view receipt. Thus we claim a locally frozen candidate and corpus
with an auditable final runner, not a pre-view cryptographic freeze of every
executed byte or independent timestamped preregistration.

The primary reference is the unchanged exact-binary mpmath erfc formula,
evaluated independently at 100/180 digits with the registered 260-digit retry
available. All 288 cases passed without that retry. Ten predetermined points
also passed a 100-digit positive-integral check, with maximum absolute log
disagreement about 1.30e-91. This supplementary check shares the analytic
integral with v2, while its arithmetic and integration routine differ; it
must not be portrayed as an entirely independent mathematical formula for v2.
The primary erfc reference remains the cross-formula comparison.

All **22 unit/property tests** passed. They include polynomial moments for
the quadrature rules, an exact ATM erf identity at several monetary scales,
parity, symmetry, monotonicity, range reconstruction, v1 regressions and the
explicit total-volatility-underflow rejection. The frozen candidate also
reproduces all 316 measured development prices/logs and passes their gates.

## 4. New confirmation results

There are 201 normal, 1 nonzero subnormal, and 86 zero-rounding reference
prices. The table excludes the latter 86 trivial zero comparisons.

| New input family | Nonzero cases | v2 closer than Jäckel | Jäckel closer | Equal error |
|---|---:|---:|---:|---:|
| Near-ATM monetary scales | 96 | 88 | 2 | 6 |
| Normalized mixture | 82 | 41 | 21 | 20 |
| Nonzero carry | 13 | 9 | 4 | 0 |
| Tail and monetary range | 11 | 8 | 3 | 0 |
| **Total** | **202** | **146** | **30** | **26** |

Counts compare absolute monetary error against the converged reference.
Nonzero-carry rows remain end-to-end BSM comparisons; rounded-forward mapping
and subsequent evaluation errors are recorded separately. For zero carry
alone, the comparison is 137/26/26 across 189 nonzero cases. This is still not
an isolated normalized-kernel comparison.

| Diagnostic | Frozen v0 | Candidate v2 | Historical Jäckel |
|---|---:|---:|---:|
| Unexpected zero among 202 nonzero cases | 1 | **0** | 1 |
| Matches rounded reference among those cases | 8 | **73** | 27 |
| Largest relative error, normal prices | 1 | **1.19e-13** | 1 |
| Largest relative error, all nonzero prices | 35.22 | **2.45e-13** | 1 |

Rounded-reference matches use an empirical converged reference, not a formal
correct-rounding certificate. All v2 outputs passed the predeclared monetary
budget `max(1e-10*reference, one ULP at rounded reference)`. All log outputs
passed `max(2e-11,16 ULP at rounded reference log)`; this is an absolute budget
with a representational allowance, not uniformly small relative log error.

For example, at `v2-0163` the log error is only about 1.20e-16 in absolute
units, but roughly 2.26 million ULP because the reference log is near zero.
Conversely, in an extremely deep zero-price tail (`v2-0236`), absolute log
error is about 0.00237 while the log's own spacing is coarse. Passing the
mixed budget must not be summarized as all log errors being below 2e-11.

In `v2-0272`, the reference OTM price is about 2.550803296012454e-241. Both
v0 and historical Jäckel return zero; v2 returns approximately
2.550803296012165e-241, relative error 1.14e-13. In `v2-0274`, v2 matches the
rounded subnormal reference around 4.480857576181261e-312; v0's relative error
is about 35.22 and Jäckel's about 0.00108. These are synthetic range probes,
not demonstrated losses on real positions, and novelty has not been established.

The 3 cases where v0 is closer (`v2-0138`, `v2-0155`, `v2-0183`) and all 30
Jäckel wins are retained. Nothing was removed to improve a score.

## 5. Cost and practical role

A post-accuracy timing check used the 106 old fixtures at indices 0,3,6,...,
one warm-up batch per implementation, five timed batches and alternating
method order. Median per-call times on this macOS arm64 machine were about
5.93 microseconds for Python `stable_detailed` v0 and 91.34 microseconds for
Python v2, ratio 15.4. This compares the actual detailed APIs at unequal
achieved accuracy. It excludes reference calculations, compilation and file I/O;
it says nothing about native C++ throughput or production latency.

The most expensive confirmation call used 480 integrand evaluations. All
computation was sequential on one numerical CPU thread, with no GPU. This v2
is a useful accuracy-oriented independent implementation/fallback candidate,
not yet a replacement for a high-throughput pricing library.

## 6. Limitations and next step

The Jäckel baseline remains the unchanged historical
[vollib commit 83ae882](https://github.com/vollib/lets_be_rational/tree/83ae882df8e19323798c7ebfb8898f94d2d92ade/src),
associated with [*Let's Be Rational*](https://doi.org/10.1002/wilm.10395).
The latest official implementation has not been verified. This note makes no
new discovery claim about that source or the established positive integral.

The reference is empirical rather than interval-certified. Gates cover this
corpus and tolerate much more than one ULP for some normal prices. The
near-zero log regime and tiny-total-volatility range remain explicit limitations.
We have not established full binary64-domain coverage, smooth Greeks, reliable
IV inversion, model calibration, hedging gains, or production suitability.
Internal AI review is not independent human peer review.

The next research step is independent reproduction of this frozen package,
then a separately versioned, condition-controlled fast path with the integral
as a fallback. Both v1 and v2 corpora are now consumed; a new accuracy claim
for another candidate requires a new confirmation set. Keep the slow v2 as
an inspectable reference rather than silently changing it for a better score.

Full rows: [`evidence/benchmark-v2.json`](evidence/benchmark-v2.json).
Timing: [`evidence/python-timing-v2.json`](evidence/python-timing-v2.json).
Source: [`lab/black_scholes_v2.py`](lab/black_scholes_v2.py).
Reproduction commands are in [`README.md`](README.md).

## Authors and research transparency

Khamit Kadyrbekov and Daniyal Kadirbekov. Independent research, Tashkent, Uzbekistan.
Correspondence: kadirbekov@gmail.com. Software contact: daniyal.kadirbekov@gmail.com.

Khamit initiated and directed the research programme. Daniyal is a coauthor and
the designated lead for its continuing software development. That prospective
responsibility does not assert that he independently implemented or validated
the frozen v2 code. The recorded experiment was prepared and executed with
OpenAI Codex assistance, including mathematical analysis, implementation,
test generation, result checking, drafting and editorial review. AI-role
reviews are not human peer review. The work is an interim, non-peer-reviewed
computational study; independent reproduction and human technical review are
invited. No academic or financial institution endorsement is claimed.

## References

1. Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy 81(3), 637–654. https://doi.org/10.1086/260062
2. Jäckel, P. (2015). Let’s Be Rational. Wilmott 2015(75), 40–53. https://doi.org/10.1002/wilm.10395
3. Historical implementation: vollib/lets_be_rational, commit 83ae882df8e19323798c7ebfb8898f94d2d92ade. https://github.com/vollib/lets_be_rational/tree/83ae882df8e19323798c7ebfb8898f94d2d92ade
4. mpmath 1.3.0, arbitrary-precision reference arithmetic. https://mpmath.org/
