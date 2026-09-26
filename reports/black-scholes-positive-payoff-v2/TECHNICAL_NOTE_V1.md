# Auditing Black–Scholes Evaluations: Exact Inputs, Tail Range, and a Frozen Prototype

GERO Numerical Research · Working technical note v1 · 26 September 2026

**Status:** local research report and reproducibility package. No external peer
review, publication, novelty claim, or claim of improved market prediction.

## Abstract

We audit a small binary64 Black–Scholes–Merton (BSM) pricing prototype against
a direct formula and an unchanged historical Jäckel implementation. The study
uses 117 development cases and 199 newly frozen confirmation cases, exact
binary64 input lifting, converged multiprecision references, and an independent
positive-payoff integral on a predetermined subset. Among the 170 confirmation
prices whose reference rounds to a nonzero binary64 number, the prototype is
closer than Jäckel in 15 cases, farther in 152, and equal in 3. This rejects
general numerical-superiority claims for this prototype on this corpus.

The audit also exposes three distinct failure mechanisms: cancellation at
small total volatility, loss of near-ATM moneyness during input preparation,
and underflow of an intermediate normalized price before monetary rescaling.
The last mechanism produces two unexpected zero prices in the historical
Jäckel wrapper. These synthetic extremes do not establish financial loss,
novelty, or a defect in the latest upstream implementation. The contribution
of this note is a reproducible diagnostic comparison, not a new pricing model.

## 1. Mathematical and computational contract

The inputs are finite binary64 values \((S,K,T,r,q,\sigma)\), with positive
\(S,K\) and nonnegative \(T,\sigma\). Each hexadecimal fixture denotes the
exact real value of that binary64 input. Desired grid coordinates and their
rounded encodings are not identified with each other.

Set \(X=S e^{-qT}\), \(Y=K e^{-rT}\),
\(m=\log(S/K)+(r-q)T\), and \(v=\sigma\sqrt T\). For \(v>0\),

\[
d_1=m/v+v/2,\qquad d_2=m/v-v/2,
\]
\[
C=X\Phi(d_1)-Y\Phi(d_2),\qquad
P=Y\Phi(-d_2)-X\Phi(-d_1).
\]

The evaluated leg is call when the high-precision \(m\le0\), otherwise put;
at ATM the choice is conventional. We call this the OTM leg throughout.
Formula evaluation error is separate from model misspecification, parameter
estimation, market fit, and hedging error. None of the latter are measured here.

## 2. Implementations and provenance

1. **Direct:** the textbook difference of normal-CDF terms, implemented with
   Python `math.erfc`. Its input preparation is shared with the prototype and
   uses `exp(log(S)-q*T)` and the corresponding strike expression. It is this
   exact implementation, not every possible direct-formula implementation.
2. **Frozen prototype v0:** an experimental choice between direct, `erf`, and
   Mills-ratio forms, with parity through `expm1`, plus a separate log-price
   interface. The old candidate was frozen before the new corpus was evaluated.
   Only instrumentation exposing the value before clipping was added; a test
   checks identical ordinary outputs against the archived code on all 316 cases.
3. **Jäckel native:** unmodified C++ files from
   [vollib/lets_be_rational, commit 83ae882](https://github.com/vollib/lets_be_rational/tree/83ae882df8e19323798c7ebfb8898f94d2d92ade/src).
   The source credits Peter Jäckel and W. J. Cody/Netlib and retains its notices.
   This is a pinned **historical mirror**, not a verified latest official release.
   The author's archive URL could not be retrieved in this session.

Jäckel's `black(F,K,sigma,T,sign)` returns an undiscounted Black price.
Our BSM adapter computes binary64 \(F=S\exp((r-q)T)\) and
\(D=\exp(-rT)\), then returns `D*black(...)`. For nonzero carry, the evidence
separately records the error caused by rounded \(F,D\) and the subsequent
evaluation error. This is an end-to-end price comparison, not an isolated
comparison of normalized kernels. The source and associated method are
described in [Jäckel, *Let's Be Rational*, 2015](https://doi.org/10.1002/wilm.10395).

Compilation used Apple clang, `-std=c++11 -O2 -DNDEBUG -ffp-contract=off`,
without fast-math, one translation unit at a time. A native probe confirmed
round-to-nearest mode and elementary gradual underflow on this machine.
Compiler details, flags and SHA-256 hashes are in
[`evidence/jackel-build-v1.json`](evidence/jackel-build-v1.json).

## 3. Protocol and reference checks

[`EXPERIMENT_V1.md`](EXPERIMENT_V1.md) and
[`evidence/corpus-v1.json`](evidence/corpus-v1.json) were fixed before the full run.
The corpus SHA-256 is
`2867930dc4cd8f0a280450dd48d566eb37144323d3d6bffd937f247a2cd5b296`.

| Partition | Cases | Construction |
|---|---:|---|
| Development | 117 | Previously used 13 × 9 normalized grid |
| Confirmation: normalized | 96 | Seed 20260926; prescribed mixture of central and tail regimes |
| Confirmation: monetary scale | 75 | Five scales from 1e-200 to 1e200; near-ATM and separated strikes |
| Confirmation: carry | 18 | Nonzero rates/dividends and three maturities |
| Confirmation: underflow boundary | 10 | Positive and negative moneyness near 37–39, total volatility 1 |

The primary reference uses mpmath 1.3.0, lifting each binary input directly
with `mp.mpf(float)`. Separate evaluations at 100 and 180 decimal digits must
agree to 1e-60 relative OTM price and 1e-55 absolute log price. A development
stress probe failed this strict 100/180 test before the full run; an explicit
180/260 retry was added to the protocol. Eight development cases required it.
All 316 cases passed the resulting convergence gate. Every attempt is retained.

The second representation is a positive-payoff integral. For OTM call,
\(z=d_1\), \(A=X\); for OTM put, \(z=-d_2\), \(A=Y\). Then

\[
V=A\varphi(z)\int_0^\infty e^{zu-u^2/2}(1-e^{-vu})\,du.
\]

With \(L=\max(1,-z)\), \(b=v/L\), and \(u=w/L\), we integrate
\(e^{zw/L-w^2/(2L^2)}[-\operatorname{expm1}(-bw)/b]\), and restore the
factor \(v/L^2\) in the logarithm. This avoids subtracting two tails and keeps
the quadrature integrand appreciably scaled even when the final price is tiny.

All **14 predetermined integral checks**, at 90 decimal digits, passed the
1e-50 absolute log-price agreement gate; the largest discrepancy was about
5.72e-68. Four failures selected after seeing the results were additionally
checked at 260/360 digits and by a 120-digit integral. Those are explicitly
post-hoc diagnostics, not another holdout. All four agreed; see
[`evidence/diagnostics-v1.json`](evidence/diagnostics-v1.json).

Both representations share mpmath arithmetic. Convergence and cross-formula
agreement are empirical reference checks, **not an interval certificate of
correct rounding**. No probabilistic coverage claim follows from this corpus.

## 4. Error measurements and results

Errors are formed inside a multiprecision context before serialization:
absolute monetary error, relative error, absolute error divided by one ULP at
the rounded reference, and integer distance from the rounded reference.
The latter two are different measurements. Log error uses the unrounded
multiprecision logarithm, not its binary64 conversion.

Let \(\eta=2^{-1074}\), the least positive binary64 number. Under
round-to-nearest-even, positive values at most \(\eta/2\) round to zero;
values between \(\eta/2\) and \(\eta\) can round to \(\eta\). The conversion
helper rounds in subnormal units explicitly to avoid double rounding there.

Across 316 cases, 213 reference prices round to normal values, 4 to nonzero
subnormals, and 99 to zero. The next table excludes the 99 zero-rounding
cases so that trivial zero-equals-zero comparisons do not dominate the counts.

| Comparison with Jäckel | Nonzero reference cases | Prototype closer | Jäckel closer | Equal absolute error |
|---|---:|---:|---:|---:|
| Development | 47 | 12 | 26 | 9 |
| Confirmation, zero carry | 156 | 13 | 140 | 3 |
| Confirmation, nonzero carry | 14 | 2 | 12 | 0 |
| All confirmation | **170** | **15** | **152** | **3** |

Against the direct implementation, the prototype wins/loses/ties 21/5/21 on
the 47 nonzero development cases and 57/49/64 on the 170 nonzero confirmation
cases. Thus the attractive development comparison against a simple formula
does not establish an improvement over a strong existing implementation.

| Diagnostic on all 316 cases | Direct | Prototype v0 | Historical Jäckel |
|---|---:|---:|---:|
| Nonfinite OTM outputs | 0 | 0 | 0 |
| Negative OTM outputs | 0 | 0 | 0 |
| Zero when reference rounds nonzero | 0 | 0 | 2 |
| Matches rounded reference among 217 nonzero cases | 15 | 19 | 53 |

The prototype clipped no negative OTM values on this corpus. Its OTM metadata
disagreed with the reference side in 12 near-ATM scale cases. These are interface
contract failures, excluded from same-leg log-error aggregates. Log outputs were
finite in all 316 cases, with 304 comparable same-leg values. Finiteness did not
imply good accuracy: the worst comparable
absolute log error measured in local ULP units was about 4.57e9 ULP, or
3.25e-5 in absolute log units, at `confirmation-0158`.

All counts describe these exact fixtures and this machine; they are not failure
rates for financial applications. Very small absolute prices can have large
relative or ULP errors. Economic importance requires a separate use-case test.

## 5. Four inspectable failures

### Small total volatility: `confirmation-0158`

The exact inputs include `S=0x1.00000003237e2p+0`, `K=T=1`, `r=q=0`,
`sigma=0x1.54e9621a41fc9p-33`. The OTM put reference is approximately
3.70033111407402524e-17. The prototype's relative error is 3.24947e-5;
Jäckel's is 1.90700e-15. This is direct evidence against accepting the
prototype's present Mills-ratio subtraction/branch selection as sufficiently
accurate. A replacement must be assessed as a new candidate on new data.

### Lost moneyness: `confirmation-0222`

`S=0x1.87e92154ef7aep-665`, `K=0x1.87e92154ef7acp-665`,
`sigma=0x1.5798ee2308c3ap-27`, `T=1`, `r=q=0`.
The exact moneyness is about 2.90083551985955747e-16, yet the prototype's
`log(S)-log(K)` returns exactly zero. It labels the call as the OTM leg although
the put is cheaper. This is an input-preparation failure before the main
formula. The monetary price comparison still evaluates the true OTM leg; the
log-interface discrepancy includes its incorrect side selection and is stored
separately rather than aggregated as same-leg numerical error.

### Normalization underflow: `confirmation-0197`

`S=0x1.c05c0a7166b4ap+432`, `K=T=1`, `r=q=0`,
`sigma=0x1.d7f9822a4c63fp+2`. The OTM put reference is
1.23873711822703942e-300, a normal binary64-scale value. Its normalized value
is approximately 8.88806e-366, which rounds to zero before Jäckel's wrapper
multiplies by `sqrt(F)*sqrt(K)`. The wrapper returns zero; the prototype and
direct code return about 7.46086e-300, also inaccurate. No method wins a
high-accuracy certificate here.

### Subnormal final price: `confirmation-0314`

`S=0x1.753025a61bfe5p+55`, `K=T=sigma=1`, `r=q=0`. The put reference is
7.38871066525728405e-318. The normalized price is about 3.22404e-326 and
rounds to zero; the historical Jäckel wrapper returns zero. The prototype
returns approximately 2.88542835e-316, with relative error about 38.05.
Its separate log price is accurate to about 3.95e-14 absolute here. This
illustrates why an accurate log output does not certify the ordinary-price
path, and why a normalized-price API can lose range during reconstruction.

Neither of the last two observations has been established as new. They concern
an old mirrored source and extremely small synthetic prices. They are not
presented as an undisclosed vulnerability, a current production incident,
a bounty-eligible finding, or a demonstrated monetary loss.

## 6. Changes from the preliminary v0 evidence

The old oracle used the decimal spelling of a float, some errors were measured
outside the high-precision context, and its log comparison rounded the reference
first. The old corpus had already influenced implementation choices, and
pre-clipping values were invisible. These weaknesses are documented, not erased:
[`evidence/archive-v0/manifest.json`](evidence/archive-v0/manifest.json) identifies
the archived original code and output. The corrected development price
win/loss/tie count against the direct method happens to remain 21/5/21; this
coincidence does not validate the old measurement method or its log-error claims.

An internal AI review of the first v1 run identified 12 wrong-side log outputs
that should not enter a same-leg error aggregate. That run is preserved in
`evidence/archive-v1-initial`; the final runner marks them separately and the
corrected run retains exactly the same ordinary prices and price comparisons.
This is a measurement correction, not a new confirmation experiment or candidate.

## 7. Reproduction and next research question

Use the commands in [`README.md`](README.md), Python with mpmath 1.3.0, and a
C++ compiler. Build the native core serially, run the 12 tests, and write a new
report path; existing evidence files are protected against overwrite. Detailed
rows, per-family summaries, reference precision attempts, mapping errors and
code hashes are in [`evidence/benchmark-v1.json`](evidence/benchmark-v1.json).
The observed environment was macOS 15.5 arm64, Python 3.9.6. Numerical thread
limits were one; no GPU, market data, trading or parallel numerical jobs were used.

The next bounded task is to design candidate v2 with robust near-ATM input
preparation and a consistent ordinary/log-price contract, then freeze a new
confirmation corpus before evaluating it. The consumed v1 confirmation data
can be used for debugging only. Further claims require latest-upstream checks,
independent reproduction, boundary/monotonicity tests, Greeks, and a separately
designed timing study. Model accuracy and implied-volatility recovery remain
separate research questions.

This work used AI assistance for code, exposition, and internal review.
Internal role names inspired by mathematicians do not represent participation,
authorship, approval, or external peer review by those people or an institution.
