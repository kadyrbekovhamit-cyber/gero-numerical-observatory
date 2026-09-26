# Experiment v1 preregistration

2026-09-26, Asia/Tashkent. Audit of frozen candidate v0; no algorithm tuning.

The v0 kernel is preserved in `evidence/archive-v0/lab/black_scholes.py` with
SHA-256 `72051118289b8f5442376cdfe964df01dc2a9afe4eb585e00d4fb926520359fc`.
Instrumentation may expose pre-clipping results but must not change numerical
outputs. The archive manifest preserves the old oracle and old result table too.

## Protocol fixed before evaluation

- Exact binary64 inputs are reconstructed from `float.hex()`. No decimal-string
  conversion is used in the primary oracle.
- The 117 legacy points are development data, not a held-out test.
- Confirmation includes 96 seeded normalized cases, 75 scale cases, 18 carry
  cases and 10 underflow-boundary cases. They are stored before the first run.
- The primary oracle is evaluated independently at 100 and 180 decimal digits.
  Its OTM relative change must be <= 1e-60 and log-price absolute change <= 1e-55.
  Failure is reported; it is not silently replaced by a candidate result.
  Before the full corpus run, the legacy stress point m=-100, v=1e-10 showed
  relative drift ~7.12e-56. An explicit bounded retry at 180/260 digits is
  therefore allowed; both attempts and the escalation flag are retained. If
  this retry fails, the case cannot support any accuracy claim.
- A deterministic subset is cross-checked using a positive-payoff integral at
  90 decimal digits. It is a different analytic representation, with mpmath
  arithmetic shared with the primary oracle; not an interval certificate.
  Fixed zero-based indices: 0, 17, 58, 89, 116, 117, 145, 174, 212, 230, 256,
  280, 300, 315. Require absolute log-price agreement <= 1e-50.
- Report monetary absolute error, relative error at high precision, real error
  in units of `ulp(round(reference))`, distance to rounded reference, and
  log-price error against the unrounded high-precision logarithm.
- Use correct round-to-nearest-even binary64 classification, including the
  half-minimum-subnormal boundary. A positive exact value may legitimately
  round to zero. Count unexpected zero/negative/NaN/Inf and clipping separately.
- Jaeckel is included only from an attributed, unmodified source baseline.
  Direct Black inputs are compared without carry for the primary price
  comparison (not an isolated normalized-kernel comparison). For nonzero carry,
  report the error from mapping BSM inputs into
  rounded F and D separately from the native Black evaluation error.
  Use vollib/lets_be_rational commit 83ae882df8e19323798c7ebfb8898f94d2d92ade;
  clang++ -std=c++11 -O2 -DNDEBUG -ffp-contract=off, sequential translation units.
  This is an attributed historical mirror; latest official archive unavailable.
- No speed superiority claim: a Python prototype and compiled C++ baseline
  would require a carefully matched timing experiment.
- Resource limits: one process at a time, one numerical thread, no GPU; use
  small fixed fixtures. No market data and no live transactions.

## Success criteria

The deliverable is a correct comparison and an honest technical report, whether
or not the candidate improves on a strong baseline. Confirmation results are
not reused for tuning. A changed algorithm requires a new evaluation version.
