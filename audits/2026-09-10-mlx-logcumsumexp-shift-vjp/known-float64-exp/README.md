# Known CPU float64 exp limitation
This is a dependency investigation, not a new finding. MLX issue #3047 explicitly includes exp among operations using float32 approximations for double inputs:
https://github.com/ml-explore/mlx/issues/3047

The isolated CPU exp regression has 47 checks: 34 mismatches before and none after a research double specialization. It is used with the separate scan recurrence to distinguish their effects. Only unary.cpp is recompiled for this comparison; other consumers of the header are not rebuilt. See the main report for the four variants and scope. Original MLX MIT licensing remains. No performance or maintainer-acceptance claim.

Set MLX_SOURCE_ROOT and MLX_CPU_BUILD as described in ../BUILD.md. Runs are sequential with numerical thread limits of one.
