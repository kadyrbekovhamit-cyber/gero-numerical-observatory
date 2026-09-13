# Integer overflow reverses MLX isclose and allclose results

Independent numerical audit by Xamit Kadirbekov / GERO Research, 13 September 2026. AI-assisted.

Native MLX source 0.32.3 returns True for uint8(0) versus uint8(255) with rtol=0, atol=1, and for int32(-2147483648) versus zero with zero tolerances. Integer subtraction and absolute value overflow before comparison.

A bounded CPU prototype for integral/bool inputs no wider than 32 bits passes 351 scenarios and 826,670 element comparisons; the original fails 233 scenarios. All 118 previously passing records remain unchanged. Full source, runner, logs and patch are being added to this folder.

GPU and int64/uint64 are outside the repair scope. Blanket floating-point casts can lose adjacent integers. Priority, maintainer acceptance, performance and model impact are unestablished.
