# Sources and evidence ledger

Verified 9 September 2026. Distinguish local measurements from history and inference.

1. MLX Contributors, Indexing Arrays / Boolean Mask Assignment, version 0.32.2 documentation:
   https://ml-explore.github.io/mlx/build/html/usage/indexing.html#boolean-mask-assignment
   Contract fact: a non-scalar source may have more elements than the mask consumes; mask shape must match indexed axes.
2. Apple MLX pinned implementation:
   https://github.com/ml-explore/mlx/blob/24c699ecee2f7c8b2040de8da1c8382c8bcf31c7/mlx/primitives.cpp#L4748
   Source fact: global flattened mask prefix sum in VJP. Six related functions were compared with the compatible native base.
3. Actual local measurements: probe-results.json (official MLX 0.32.2 wheel), run-before.log and run-after.log (real C++). Fresh reruns for publication.
   Measured facts: unused-element gradient 20 instead of 0; actual-forward finite differences and explicit-loop control agree; synthetic loss step 1070 to 1168.75 versus 818.75.
4. Independent algebra: fixed-mask source selection is linear. The correct flat source address is b*K+p[b,j].
   Derivation, corroborated by execution; no full-model impact inferred.
5. Source history: https://github.com/ml-explore/mlx/pull/2663 (initial boolean assignment), https://github.com/ml-explore/mlx/pull/2832 (shapes), https://github.com/ml-explore/mlx/pull/3421 (vmap test), https://github.com/ml-explore/mlx/pull/3633 (JVP fixes).
   The original audit preserved four limited GitHub searches and these PR changes. No exact batch-VJP report found in that selection; novelty unestablished. Presence of code in an old PR is not proof of prior reporting.
6. Proposed fix: masked-scatter-batch-vjp.patch. Passes 255 targeted checks in 19 scenarios on the documented partial native build. Not a clean full build of main; no GPU/compiled/full-suite/performance validation.

AI-assisted preparation. No claim of Apple endorsement or maintainer acceptance. MIT license preserved.
