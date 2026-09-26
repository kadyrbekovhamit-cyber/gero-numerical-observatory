# Review priorities for v3

Check finite Taylor remainder derivations separately from floating-point error.
Inspect the direct-path heuristic, exact-input ATM condition, and frozen fallback.
Review all failures/losses, subnormal rounding and near-zero log ULP behavior.
Replay uses exact binary64 inputs and a historical native comparator, not current
upstream. The local freeze is not external preregistration; replay is post-result.
The unpublished book and private handoff state are excluded from this release.
