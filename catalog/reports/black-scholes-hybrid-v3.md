# A hybrid Black–Scholes evaluator: recovering speed with bounded synthetic accuracy evidence

**Khamit Kadyrbekov and Daniyal Kadirbekov**  
Independent research, Tashkent, Uzbekistan  
Working note v3.0 — 26 September 2026 — not independently peer reviewed

## Abstract

The published v2 prototype evaluates a positive-payoff representation of the
Black–Scholes–Merton price by quadrature. It avoids problematic subtraction
in extreme synthetic examples, but the detailed Python API is slow. We replace
quadrature in selected regions by an exact-input ATM identity, a small-volatility
moment series, an asymptotic difference series, or a conditioned direct CDF
evaluation. The unchanged v2 integrator remains a fallback. On 480 new locally
frozen exact-binary64 input cases, both v2 and v3 pass all predeclared mixed
price/log tolerances. Median batch time falls from 90.143 to 19.668 microseconds
per call, a 4.583-fold speedup on this machine and synthetic mixture. On 426
nonzero rounded reference prices, v3 is closer to the reference than v2 in 76
cases, farther in 99, and tied in 251. Thus speed is recovered without breaking
the declared tolerances, not without every change in numerical error. Analytic
Taylor remainder arguments concern exact-arithmetic truncation; total floating
error is not certified. No new market model, global superiority, historical
novelty, or economic return is established.

Substantial AI assistance (Codex) was used for derivation, implementation,
experiment design, execution, analysis, and drafting. Human independent
reproduction has not yet occurred. Khamit initiated and directed the research;
Daniyal is coauthor and the designated lead for continuing software work.
This does not assert that Daniyal independently implemented or validated this
frozen prototype. Historical-mathematician AI roles are not human consultants.

## 1. Contract and positive-payoff representation

All six inputs S,K,T,r,q,sigma are interpreted as their exact binary64 values
by the high-precision reference. Let X=S exp(-qT), Y=K exp(-rT),
m=log(S/K)+(r-q)T, and v=sigma sqrt(T)>0. Let L=X if m<=0 (OTM call), or
L=Y if m>0 (OTM put), and z=-|m|/v+v/2. The positive OTM price is

\[
P=L\,\varphi(z) B(z,v),\qquad
B(z,v)=\int_0^\infty e^{zu-u^2/2}(1-e^{-vu})\,du.
\]

The opposite option follows from put–call parity. v2 and v3 use the same
log1p/frexp moneyness, split-exponent monetary reconstruction and parity
calculation. Factoring monetary scale before reconstruction avoids prematurely
rounding a normalized price to zero. The candidate uses ordinary Python/libm
binary64 only, with no oracle or multiprecision calls.

## 2. Small-volatility moment expansion

Define I_n(z)=integral from 0 to infinity of u^n exp(zu-u^2/2) du. Then

\[
\frac{B(z,v)}v=\sum_{n=1}^{N}\frac{(-v)^{n-1}}{n!}I_n(z)+R_N,
\qquad |R_N|\le\frac{v^N}{(N+1)!}I_{N+1}(z).
\]

This finite expansion and bound follow by integrating the pointwise Taylor
remainder of exp(-vu) against a nonnegative weight. No exchange with an
unbounded infinite series is required. Integration by parts yields

\[
I_0=\sqrt{\pi/2}\,e^{z^2/2}\operatorname{erfc}(-z/\sqrt2),\quad
I_1=1+zI_0,\quad I_n=(n-1)I_{n-2}+zI_{n-1}.
\]

The implementation restricts -1<=z<=1/16 and 0<v<=1/8 to avoid the severe
cancellation of this forward recurrence at large negative z. It stops when
the first omitted term is <= one binary64 epsilon times the partial sum,
with at most 17 included moments. Failed positivity or convergence uses v2.
The computed stopping estimate does not include recurrence or libm rounding.

## 3. Tail expansion of the difference

Write a=-z>0 and rho=v/a. Expanding exp(-u^2/2) inside the positive integral,
then integrating each finite term against exp(-au)(1-exp(-vu)), gives

\[
B(-a,v)=\frac{v}{a^2}
\left[\sum_{n=0}^{N-1}(-1)^n(2n-1)!!\,a^{-2n}H_{2n+1}(\rho)
+E_N\right],
\quad H_p(\rho)=\frac{1-(1+\rho)^{-p}}{\rho},
\]

where (-1)!!=1 and H_p(0)=p. The finite Taylor remainder gives

\[
|E_N|\le (2N-1)!!\,a^{-2N}H_{2N+1}(\rho).
\]

This is a bound for the *difference* under a positive weight; subtracting
two separately approximated Mills ratios is unnecessary. Evaluate H_p by
`-expm1(-p*log1p(rho))/rho`, with its limiting value when rho rounds to zero.
The implementation tries this only for a>=12, uses at most 64 omitted-term
checks, and abandons a nondecreasing term sequence before convergence. v/a^2
remains factored until monetary reconstruction. General erfc asymptotics and
their remainder behavior are standard; see [NIST DLMF §7.12](https://dlmf.nist.gov/7.12).
The particular implementation and experiment are not claimed historically new.

## 4. ATM, conditioned direct path, and fallback

For the exact-input case S=K and r=q,

\[
P=L\,\operatorname{erf}\!\left(\frac{v}{2\sqrt2}\right).
\]

For v<1e-4 use v/sqrt(2pi) times (1-v^2/24+v^4/640); the next normalized
term has magnitude v^6/21504. Keeping v factored handles the smallest positive
subnormal volatility when a large monetary scale makes the final price normal.
This extreme is a separate unit fixture, not part of the 480-case confirmation.
Rounded m=0 alone is not sufficient to enter the exact-input ATM branch.

Otherwise the direct normalized price is Phi(z)-exp(|m|)Phi(z-v). A heuristic
screen combines subtraction condition, squared normal arguments and |m|.
It requires arguments in [-26,26], |m|<=600, and score <=1e-13. This is not
a libm error certificate. Every region not accepted uses the unchanged v2
integral; v3 still explicitly rejects unsupported ranges. Greeks, branchwise
derivatives and implied-volatility inversion are outside this experiment.

## 5. Experimental integrity and accuracy

All 604 v1/v2 inputs were development data. They passed the existing mixed
price/log/side gates using saved 35-digit reference strings. Twenty-eight unit
test methods passed, including independent high-precision integral checks of
the series, seams, scaling, parity, monotonicity and reciprocal symmetry.

Before the first new outputs, the candidate, all local lab modules/tests,
protocol, timing code, native sources/binary, dependency hashes and 480 new
inputs were frozen locally at 2026-09-26 07:53:38 UTC. Candidate SHA-256:
`4bf8c51ca33082665b9c6902bb2a114604d937b6dd0ed67e0c17ece46b4547bc`.
This improves the earlier incomplete pipeline freeze; it remains local
evidence, not an externally preregistered or blinded experiment. Source and
candidate hashes remained unchanged after outputs. The corpus contains
96 small-v, 80 tail, 80 regular-with-carry, 96 switching-neighborhood,
80 monetary-range and 48 exact-ATM cases. Thirteen nextafter steps removed
input duplicates with old data; prices were not used in corpus generation.

The unchanged erfc oracle converged at 100/180 digits in every case; none
needed the planned 260-digit retry. All 12 fixed supplementary integral
checks passed at 100 digits. They use another numerical integrator but a
mathematical representation related to the candidate; they are not an
independent mathematical formulation. All 80 carry-adapter oracle checks
also converged.

Both v2 and v3 pass all 480 gates: finite nonnegative call/put, correct OTM
side, no unexpected zero, price error <=max(1e-10 times reference, one ULP
of rounded reference), and log error <=max(2e-11,16 ULP of rounded reference
log). These are mixed practical tolerances, not universal correct rounding.

| Quantity | Result for v3 |
|---|---:|
| Normal / subnormal / rounds-to-zero references | 375 / 51 / 54 |
| Maximum relative error, normal prices only | 3.1224e-13 |
| Maximum absolute log error, all 480 | 6.0008e-13 |
| Rounded-reference matches, 426 nonzero prices | 134 |
| Rounded-reference matches, 51 subnormals | 40 |
| Exact ATM / small-v / tail / direct / fallback counts | 48 / 125 / 176 / 63 / 68 |

Very small subnormals require a separate interpretation: in case v3-0353,
the exact price is about 2.7081e-324 and the correctly rounded result is the
smallest positive binary64 number, about 4.9407e-324. Its relative error is
82.44% despite correct rounding. Do not advertise 3.1224e-13 for all nonzero
prices. Similarly, v3-0455 has log error about 2.0e-17 but 1.58e12 log-ULPs
because the log is near zero; do not advertise uniformly small log-ULP error.

On the 426 nonzero rounded prices, smaller exact absolute error counts are:

| Comparator | v3 closer | v3 farther | Equal error |
|---|---:|---:|---:|
| Own original v0 | 398 | 12 | 16 |
| Frozen integral v2 | 76 | 99 | 251 |
| Historical native Jaeckel | 329 | 61 | 36 |

The historical comparator is mirror commit
83ae882df8e19323798c7ebfb8898f94d2d92ade, not verified latest official upstream.
Its comparison includes the forward/discount adapter, whose distortion is
recorded per carry case. These results say nothing about a newer version or
the frequency of such inputs in trading. v0 produces 50 unexpected zeros,
the historical adapter/native combination 56, and v2/v3 none on these 426
prices. Both v2 and v3 retain finite logs when monetary prices round to zero.

## 6. Timing and the explicit compromise

Timing used the detailed Python APIs, one sequential CPU worker, no GPU,
one warm-up and seven alternating-order batches with five loops each. The
table reports ratios of median batch times, not medians of per-call ratios.

| Fixed stratum | Cases | v2 microseconds/call | v3 microseconds/call | v2/v3 |
|---|---:|---:|---:|---:|
| All new inputs | 480 | 90.143 | 19.668 | 4.583 |
| Exact ATM | 48 | 75.445 | 3.238 | 23.304 |
| Small volatility | 96 | 93.113 | 6.174 | 15.082 |
| Negative tail | 80 | 92.562 | 8.019 | 11.543 |
| Monetary range | 80 | 92.998 | 8.089 | 11.497 |
| Regular with carry | 80 | 84.384 | 27.526 | 3.066 |
| Switching neighborhoods | 96 | 93.469 | 54.087 | 1.728 |
| Fallback subset, selected by v3 path | 68 | 94.025 | 96.706 | 0.972 |

The fallback subset is about 2.85% slower because the selection work precedes
integration; it is not an additional independent stratum. Its v3 price and
log outputs are identical to v2. Across all inputs v0 takes 5.307 microseconds:
v3 remains 3.706 times slower than that less accurate baseline. No C++ speed
comparison or production-throughput claim is made. The predeclared criterion
(all gates plus >2x all-case speedup over v2) is met. The main result is a
measured accuracy/speed tradeoff, not that v3 beats every comparator everywhere.

## 7. Limits, reproduction, and next work

Converged multiprecision and exact-arithmetic remainder formulas are not a
certified floating-point error proof. Binary64 input transformation, libm,
recurrence rounding, subnormal rounding and parity reconstruction remain
separate error sources. The stress mixture overweights extreme ranges. There
is no full-domain test, independent human validation, model calibration,
market-loss estimate, Greek accuracy assessment, or implied-volatility study.

Run from the project root with Python and mpmath 1.3.0, one numerical thread:

```bash
python3 -m unittest discover -s tests -v
python3 reproduce_v3.py --output /tmp/black-scholes-v3-reproduction.json
```

The local freeze pins this machine's native binary and shared one-worker
wrapper. The portable harness was added **after** the original result. It
validates original source/input and mpmath hashes, compiles the native comparator
sequentially in a temporary directory, and substitutes only its location and
the source verifier. Its report explicitly labels the new build/environment;
it does not rewrite the original freeze or claim external independent validation.
The native C++ wrapper hash is pinned transitively by the frozen native-build
receipt and is checked explicitly by this replay harness. Native sources and
license notices are preserved. All raw rows, timing batches and pipeline hashes are
in `evidence/benchmark-v3.json`, `evidence/python-timing-v3.json`, and
`evidence/freeze-v3.json`; the exact protocol is `EXPERIMENT_V3.md`.

The next useful investigation is Greek continuity/accuracy at branch seams
and robust conditioned IV inversion on a newly frozen corpus. A separate
study could replace heuristic direct-path admission with verified error
control. Existing v2 and v3 evidence must remain unchanged.

## Sources and predecessor

- [Published v2 working paper and evidence](https://doi.org/10.5281/zenodo.22972343).
- [NIST DLMF §7.12: erfc asymptotics](https://dlmf.nist.gov/7.12).
- Pinned historical Jaeckel/Cody source and provenance: `vendor/` and
  `JACKEL_SOURCE_REVIEW_2026-09-26.md`. Latest official Jaeckel PDF retrieval returned
  HTTP 502 during this continuation; no latest-version claim is made.
