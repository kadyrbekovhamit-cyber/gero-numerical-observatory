# Second confirmation protocol, after v1 harness diagnosis

26 September 2026. Candidate unchanged: SHA256
2519288a7ee616274ad7f87f18ec2d6021e4407b4bc7272b1674bcb7b252985a.

The first 160 scenarios remain frozen, with **158/160** passing all original
gates. They must never be relabelled 160/160. Two failures:

1. iv-086: float(mpmath price) selected the adjacent lower subnormal float.
   The discrepancy persisted at 180/260/360 digits. Converting the high
   precision decimal through an exact Fraction selected the correct cell;
   a 1024-bit Arb check independently showed the seed price above the old
   cell. All returned root witnesses passed. This is a generator problem;
   no claim of a newly discovered mpmath defect or maintainer report is made.
2. iv-144: the assumed 64-bit boundary ambiguity did not occur. Arb proved
   a sign and the solver resolved the roots. Two mpmath reference convergence
   checks also failed because computing A-B cancelled a tiny intrinsic value.
   The 260-digit references were still contained in both Arb witnesses.
   The oracle now evaluates discounted forward differences with expm1 and
   computes log-forward from log(S/K)+(r-q)T. The candidate is unchanged.

The new protocol/generator/reference/runner and exact new inputs are frozen
before viewing their candidate outputs. Original files/results are retained
and transitively hashed. No reuse of the first inputs as new confirmation.

96 new distinct contract/quote/settings cases:

- 64 rounded quote recovery scenarios: both options, four monetary scales
  including 1e-309, two maturities, four volatilities. Exact Fraction-mediated
  rounding must agree at 180/260 digits during input construction.
- 16 rounded intrinsic/ceiling scenarios at different S,K from v1.
- 8 deliberate budget scenarios: two sigma caps, two iteration limits,
  three uncertain root comparisons, one uncertain boundary.
- 8 expiry inside/outside cases.

Same predeclared gates as v1: expected status/resolution; witness containment,
sign and 180/260 convergence; root width<=1e-12 when converged; all 64 latent
sigmas in the outer enclosure with reference price strictly inside the quote.
No symmetry quartet here; v1's eight checks remain separate.

The changed reference is a separately written high-precision numerical
control, not a new formal proof. No performance advantage, model accuracy or
market claims. All failures retained; no post-output retuning of this freeze.
One sequential shared worker, python-flint ctx.threads=1, no GPU.
