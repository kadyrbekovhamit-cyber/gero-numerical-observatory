# Numerical experiment v0

Date: 2026-09-25 (Asia/Tashkent). CPU only, one numerical thread, no GPU.

**Superseded 2026-09-26:** corrected measurements and the pinned Jaeckel
comparison are in `../TECHNICAL_NOTE_V1.md` and `benchmark-v1.json`. The text
below is historical; in particular, below-minimum and round-to-zero must not
be conflated. Original code and evidence are preserved in `archive-v0`.

**Status update, 2026-09-25:** preliminary development results, pending a
methodology audit before external publication. The oracle's decimal-string
inputs differ from the candidate's binary inputs; relative errors are evaluated
outside the high-precision context; the grid has already informed candidate
changes; nonnegative outputs are checked after internal clipping. Historical
counts below are preserved, not independently certified. See
`../PUBLICATION_STRATEGY_2026-09-25.md` for the corrective sequence.

## Question

Can an unfitted regime-aware representation reduce float64 cancellation in the
same Black--Scholes price? This experiment does not change the model.

## Frozen contract

- normalized inputs: `K=1`, `T=1`, `r=q=0`, `S=exp(m)`;
- 13 log-forward-moneyness values from -100 to 100;
- 9 total-volatility values from `1e-10` to `10`;
- OTM call for `m<=0`, OTM put for `m>0`;
- oracle: independent `mpmath`, 100 decimal digits;
- candidates: direct `erfc` formula and a condition-selected combination of
  direct, `erf`, parity/`expm1`, and scaled Mills-ratio forms.

## Result

There are 117 cases. Seventy exact prices are below the least positive float64,
so an ordinary float64 price cannot represent them. Among the 47 representable
cases, the regime-aware prototype has smaller absolute error in 21 cases, the
direct formula in 5, and they tie in 21. Neither method produced a negative,
infinite, or NaN price. The new log-domain OTM result remains finite in all 117
cases. For the 70 ordinary underflows, against the 100-digit oracle its worst
discrepancy after conversion is 4 ULP of the float64 logarithm. This 4-ULP claim
is limited to that underflow subset; representable central/tail values have a
different error profile reported in the JSON. All seven
unit/property tests pass; the property sample
contains 100 deterministic cases and checks bounds and volatility monotonicity.

The prototype therefore fixes concrete cancellation regimes, especially very
small ATM total volatility, but it does **not** dominate the direct formula on
the frozen grid. The five losses are evidence against a premature superiority
claim, not points to delete.

## Decision

Continue as a research hypothesis. The prototype now exposes call/put
intrinsic, the common time value and its log-domain representation separately.
Before any novelty or production claim, compare against Jäckel's
*Let's Be Rational*/Cody-quality kernel, add monetary scaling and Greeks, and
measure runtime at equal accuracy.

Machine-readable evidence: `benchmark-v0.json`.
