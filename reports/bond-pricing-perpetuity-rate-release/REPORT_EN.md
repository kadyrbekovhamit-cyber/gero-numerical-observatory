# bond_pricing: perpetuity-rate inversion fixed in release 1.0.1

Xamit Kadirbekov · GERO · 22 October 2026. Investigation and writing were AI-assisted.

## Result

The maintainer fixed the reported perpetuity branch in **bond_pricing 1.0.1** and publicly credited Xamit Kadirbekov / GERO Research in [issue 11](https://github.com/jrvarma/bond_pricing/issues/11). This is a released-fix retrospective, not a new unresolved finding. The implementation credit belongs to the maintainer.

For a payment of 100 each year forever, with the first payment one year away and present value 1,000, the periodic rate is 100/1,000 = **0.10**. Release 1.0.0 returned **10.0**. Feeding that rate back into the pricing function returned 10 instead of 1,000. The infinite-period branch inverted price and payment and omitted timing/frequency handling.

For payments in arrears, q = payment / price. For immediate first payment, q = payment / (price − payment). If cash flows occur f times per year and nominal compounding is m times per year, r = m[(1+q)^(f/m)−1]; continuous compounding uses f log(1+q). These formulas follow the geometric series. The positive, convergent inputs used here avoid the boundary price = payment.

## Fresh verification

On 22 October, the official PyPI 1.0.1 wheel was downloaded and checked against its published SHA-256. The unchanged public API was run against an independent 80-digit Decimal reference.

| Check | 1.0.0 | 1.0.1 |
|---|---:|---:|
| Perpetuity rate mismatches, 288 cases | 288 | 0 |
| Price round-trip mismatches, same 288 cases | 288 | 0 |
| Finite-term controls, 18 cases | 18 pass | 18 pass |
| Two-element timing/frequency broadcast | fails | passes |

The matrix varies price, payment ratio, cash-flow frequency, nominal/continuous compounding and first-payment timing. Before and after are separate actual package copies. Missing optional isda_daycounters produces a warning; these tests do not depend on specialist day-count conventions. The wheel also emits an internal vectorization warning; the finite outputs are checked explicitly, so this is not a warning-free or full-suite certification.

## Timeline and limits

The report was sent on 1 October. The maintainer opened and closed the credited public issue on 3 October; version 1.0.1 is available on [PyPI](https://pypi.org/project/bond-pricing/1.0.1/). The current repository revision checked on 22 October is `a8b753c09b971624104ea9bdccd7b1ed936b8545`.

These are synthetic regression checks, not evidence of actual customer loss. Zero, negative, non-convergent and near-singular inputs are outside this corpus. No private correspondence is included. The reproduction archive contains the independent harness, before/after numeric results and provenance; upstream source remains under GPL-3.0. Report CC BY 4.0; original harness MIT.
