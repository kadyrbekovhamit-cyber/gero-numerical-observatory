# Interval implied volatility: protocol fixed before confirmation

Date: 2026-09-26. Authors: Khamit Kadyrbekov and Daniyal Kadirbekov.
Substantial OpenAI Codex assistance; no independent human review claimed.

## Contract and mathematical invariant

For exact S,K>0, T>0 and real r,q, European BSM call and put prices are
continuous on sigma>=0, strictly increasing for sigma>0 (Vega>0).
The attainable price range is [intrinsic, discounted S) for a call, or
[intrinsic, discounted K) for a put. The upper limit is not attained by any
finite sigma. T=0 is treated separately. Thus the preimage of an exact quote
interval is empty, {0}, an interval with inherited endpoint closure, or all
volatilities at expiry. Interior quote endpoints have unique roots.

`lab/iv_interval.py` constructs root brackets using only definite signs of
Arb enclosures of price-target. A definite negative endpoint and positive
endpoint plus monotonicity enclose a root. Unknown sign is not equality.
While bracketing, an uncertain finite upper trial is discarded as an upper
bound and +infinity is reported. After bracketing, uncertainty retains the
last proved bracket. Precision, iteration and sigma exhaustion are explicit.
`resolved` only concerns root-location tolerance, not unique identification
from the quote. The outer interval may over-enclose by root tolerances.

The library contract is a dependency, not independently proved here. We do
not claim proof-assistant verification, parameter/model uncertainty handling,
market validity or a new inversion theorem. Numerical effort is bounded by
precision<=4096 bits, <=2048 bisections, bounded rational input size and a
sigma search cap. These are work bounds, not a wall-time/RAM SLA.

## Development, kept separate

24 unit-test methods cover exact decimal versus binary64 input, adjacent
rounding cells (including zero, odd/even ties, subnormal, maximum finite),
intrinsic/ceiling/open/expiry states, monetary and time scaling, parity,
and deliberately exhausted budgets. The v4 loss example is development.
First test run had one reference-convergence gate failure: subtraction from
an ITM price magnified *relative residual* error. The high-precision residual
was inside the Arb witness. Before freeze the control gate was corrected to
require convergence at 1e-160 times the price/target scale AND at 1e-8 times
the witness width; a zero-width witness requires identical references.
Both logs are retained. No candidate correction was required by that failure.

## New confirmation input construction

160 distinct contract/quote/settings scenarios, constructed without calling
the candidate solver. The generator uses a separately written mpmath formula
at 260 decimal digits to construct synthetic quote intervals from selected
latent volatilities. This construction is disclosed; it is not real market
data or a blind external test. Families are explicit Cartesian products:

- 64 decimal quote intervals, both option types, four moneyness values,
  four maturities, two sigma values and varying carry.
- 24 nearest-even binary64 cells, both option types, four monetary scales
  including subnormal prices, three latent sigma values.
- 16 explicit intrinsic/ceiling/empty/open boundary scenarios.
- 8 zero-maturity inside/outside scenarios.
- 16 rounded intrinsic/ceiling observations, including zero price.
- 8 exact point quotes (root uncertainty is not economic quote uncertainty).
- 8 intentionally constrained sigma/iteration/precision/boundary budgets.
- 16 symmetry scenarios: four base/parity/monetary/time quartets.

An initial generator cardinality assertion caught only two rather than four
maturities; a subsequent uniqueness assertion caught one repeated negative
put quote. Both were corrected before any corpus was written or solver output
viewed and before freeze. No confirmation result was discarded.

Candidate, all runner/generator/reference/test/CLI sources, this protocol,
requirements, exact inputs, package source/native file hashes and runtime
versions are frozen before candidate outputs on this corpus are viewed.
No passing result may be claimed after silently modifying this freeze.

## Predeclared gates

1. Exact expected status and expected `resolved` for every scenario.
   The eight deliberate budget cases must remain unresolved with the
   predeclared endpoint state. They are successful *budget handling*, not
   successful accurate root recovery.
2. For every retained definite sign, verify the saved exact dyadic witness
   sign and containment of a separately computed mpmath 260-digit difference.
   180/260 references must meet both convergence checks described above.
   This is an independent implementation control, not an independent person
   or a rigorous replacement for Arb. Uncertain comparisons are counted.
3. Every converged endpoint bracket has width<=1e-12 absolute sigma.
4. For the 88 constructed latent-sigma quote cases, confirm the latent sigma
   lies in the returned outer enclosure and its independently computed price
   lies strictly inside the supplied exact quote interval.
5. For all four symmetry quartets, the rescaled endpoint brackets overlap
   for both endpoints: eight checks. No timing/specialist superiority gate.

All outcomes are stored, including failed gates. Only development tests are
allowed before freeze. Confirmation is one sequential process through the
shared one_worker wrapper, python-flint ctx.threads=1, numerical env threads=1,
GPU disabled. Native binaries remain local; distribution provides pinned
requirements. The portable runner checks exact source/input hashes and
dependency versions, allowing platform-specific python-flint file hashes.
Same-host archive replay is not an external reproduction.
