# Experiment v2 protocol

2026-09-26, Asia/Tashkent. Frozen v0 and historical Jaeckel remain baselines.
The v1 corpus (all 316 points) is now development data. No v1 point is treated
as independent confirmation for a changed implementation.

## Candidate and scope

Develop a separate pure-Python binary64 candidate with robust near-ATM
log-moneyness, positive-payoff quadrature, and range-aware reconstruction of
the monetary price. No multiprecision arithmetic inside the candidate and no
oracle fallback. Keep ordinary price and log price tied to the same OTM leg.
Quadrature convergence is an estimate, not a rigorous total error certificate.
Before freeze, internal review added explicit rejection when positive sigma,T
produce rounded zero total volatility. That unsupported input must not be
silently treated as a deterministic option: a large cash scale may retain time
value. True sigma=0 or T=0 remains supported. Nonfinite log/exponent arithmetic
and exhausted integration bounds/budgets fail explicitly as well.

The implementation shares integral factors between ordinary/log outputs;
ordinary reconstruction retains the original monetary mantissa and uses
frexp/ldexp, rather than exponentiating a rounded complete log price. Fixed
tail windows have analytic integrand tail bounds checked against the estimated
integral. GL16/GL32 differences do not include transcendental/input errors.

This is a numerical engineering experiment, not a new BSM model or a claim of
novelty. Slower evaluation is acceptable in this first accuracy experiment;
never infer speed superiority from results against C++.

## Development and freeze

Use the 316 v1 cases, analytical identities, polynomial quadrature identities,
monotonicity, symmetry, parity and targeted regression checks for development.
Once those checks are satisfactory, freeze candidate source hashes and generate
the prescribed new corpus before computing any candidate outputs on it.
Do not change the candidate after viewing confirmation outputs; preserve any
failure, and give a further candidate a new version and new confirmation data.

## New confirmation corpus

Seed 2026092602, exact binary64 hex inputs; 288 points:

- 96 near-ATM points at scales 1e-250 through 1e250, displaced by one to four
  nextafter steps, with total volatility in [1e-12,1e-3].
- 96 normalized points, K=T=1, r=q=0, total volatility in [1e-12,20], with
  moneyness sampled by distance in normal-standard-deviation units and capped
  at absolute 600 before encoding S=exp(m).
- 72 nonzero-carry points with K in [1e-150,1e150], S/K in [exp(-2),exp(2)],
  T in [1e-6,10], r in [-0.12,0.12], q in [-0.06,0.06], sigma in [1e-4,2].
- 24 tail/range points: nominal m around signed 36 to 43, v in [0.7,1.3],
  K cycling through 1e-200, 1, 1e200.

Primary oracle: unchanged v1 exact-binary reference with 100/180/260 digits.
Independent integral checks at fixed indices 0,31,63,95,127,159,191,223,255,287
using 100 decimal digits and absolute log agreement <=1e-50. Record failures
explicitly. Boundary-specific nearest-even tests remain separate unit tests.

## Predetermined practical gates

- No nonfinite ordinary prices, negative prices, unexplained zero prices, or
  wrong-side OTM metadata within this corpus; no silent quadrature exhaustion.
- For nonzero reference prices: absolute OTM price error <=
  max(1e-10 * reference, one ULP at rounded reference). This combines a normal
  relative budget with a subnormal rounding budget; it is not a global proof.
- Same-leg absolute log-price error <= max(2e-11,16 ULP at rounded log).
- Record own v0/Jaeckel win/loss/tie counts separately; passing these practical
  gates does not require or imply outperforming Jaeckel in every case.
- Keep nonzero carry comparisons end-to-end; record mapping error separately.

Run sequentially on one numerical CPU thread, no GPU. No publication, external
communication or financial transaction is part of this computational experiment.
