## What this could mean in practice

Explanation added 19 September 2026. This explains the existing experiment; it is not a new defect or a new numerical run.

**Measured in our synthetic example:** a parameter should move from 1 to 0.95848924. Intermediate overflow instead makes the proposed update negative infinity. Our GERO finite-value guard skips it, so the saved parameter stays at 1. The guard and decision document are demonstration code; the loss, gradient and proposed update are computed by the native MLX core.

**Possible in a training system:** if non-finite values are allowed into model parameters or optimizer state, later calculations could become unusable and the run could require recovery or a restart. If protective checks repeatedly skip valid updates, learning could make less progress. These are conditional consequences, not effects measured in a real model by this audit.

**Practical concern:** avoidable computation, debugging time and unpredictable progress. Their frequency, effect on model quality and monetary cost have not been measured. This case uses an extreme but finite float32 gradient seed; it does not establish a problem in typical workloads or a defect in a deployed Apple device.

The local candidate corrects the primary first-derivative cases but still has tail and higher-derivative limitations. No upstream submission or acceptance is claimed.
