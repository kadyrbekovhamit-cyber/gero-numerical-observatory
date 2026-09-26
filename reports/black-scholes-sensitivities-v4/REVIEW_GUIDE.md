# Independent review guide, v4

Start with TECHNICAL_NOTE_V4.md, EXPERIMENT_GREEKS_V1.md and the release
README. This is an engineering/educational working paper using standard BSM
mathematics. The preceding price paper is public at
https://doi.org/10.5281/zenodo.22973900. Publication does not imply peer review.

## Questions worth trying to falsify

1. Are Delta, Gamma and Vega derivatives of the unspecialized model with the
   stated inputs fixed? Does the ATM example distinguish a surface derivative
   from a partial derivative and avoid alleging an actual AD bug?
2. Do exact-binary64 inputs, reference convergence, range labels, mixed value/log
   tolerances and recorded denominators agree?
3. Can the exponent decomposition fail outside the selected corpus? Seek
   counterexamples near underflow, overflow, discount and tail switches.
4. Does the rounded-reference finite-difference control separate truncation,
   rounding and kernel error? Tiny steps are not market failure probabilities.
5. Do the six IV examples demonstrate information loss without overclaiming
   a certified threshold or a completed interval-IV solver?
6. Are literature overlap and the limited coverage of branch neighborhoods
   honestly stated? This is not an exhaustive seam audit.

## Reproduction and provenance

reproduce_greeks_v1.py checks the original frozen numeric sources and inputs.
reproduce_sensitivity_audit.py checks the frozen price sources and original
exploratory runner. Both portable harnesses were added after the results; they
change local verification only, not numerical code or metrics. Use fresh
output paths. Original evidence must not be overwritten.

The original freeze also records the local worker and runtime hashes. The
portable replay does not require the original machine or a native comparator.
Differences on another libm/CPU/compiler should be retained and investigated,
not hidden by updating the original evidence. mpmath source-tree hashes are
strict; a changed dependency requires an explicit separate audit.

Same-host local replays already match. No independent human review or
other-machine reproduction is claimed. The v4 release is focused on Greeks
and information loss; full historical price/comparator packages remain in
preceding published versions. No trading, novel model or production claim.
