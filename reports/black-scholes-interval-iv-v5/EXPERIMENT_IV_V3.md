# Third pre-output confirmation protocol: exact reference boundaries

26 September 2026. **Same unchanged candidate** SHA256
2519288a7ee616274ad7f87f18ec2d6021e4407b4bc7272b1674bcb7b252985a.

Retain the previous results as originally measured: 158/160 (v1) and 82/96
(v2) passing *all predeclared gates*. The v2 stable expm1 reference inadvertently
replaced exactly representable zero-carry intrinsic differences by log/exp
calculations. Tiny oracle rounding error then failed zero-width exact witness
checks in 14 scenarios (30 witness checks). This is a reference defect.
V3 preserves exact S-K for zero carry / zero maturity. Three reference-specific
unit methods check exact boundary identities and tiny-carry convergence.

Post-output diagnosis with this reference checks all 1108+673 retained
definite witnesses from v1/v2 successfully. That is **post hoc**, not new
confirmation, and does not erase the old failed gates or iv-086's invalid
latent input. No candidate change, gate relaxation, exception suppression or
evidence deletion. The correction is in the independent reference formula.

Freeze a third corpus before its candidate outputs: **64 distinct new
contract/quote/settings scenarios**, with an assertion excluding both earlier
corpora. 32 nearest-even cells (both options, two scales including 1e-309,
two maturities, four latent sigmas), 16 rounded boundaries, eight deliberately
insufficient budgets, eight expiry cases. No new symmetry cases.

The gates are unchanged: expected status/resolution, 180/260 reference
convergence and containment in every retained definite-sign witness, valid
dyadic signs, width<=1e-12 for converged roots, all 32 latent sigmas within
the returned outer enclosure and their reference prices strictly in the
supplied cells. The eight budget cases must be visibly unresolved.

Candidate, generator, reference, runner, tests, protocol, CLI, requirements,
exact inputs and prior frozen receipts/results are hashed before new output.
Original errors are retained. One shared sequential CPU worker, ctx.threads=1,
no GPU. No speed, economic forecast, novel formula or external-review claim.
