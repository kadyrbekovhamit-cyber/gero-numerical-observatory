# YouTube release metadata

Title: Forward Correct, Backward Wrong: nntrainer Average Pooling

Description:
Can an average-pooling layer produce the correct forward values and still lose gradients in backward? This native C++ reproduction on Samsung nntrainer commit a7ea056 shows one asymmetric `padding=same` case.

Input: `[[1,2],[3,4]]`, 2×2 average window, stride 1, unit output gradients.

- Forward: `[2.5, 3, 3.5, 4]` — correct.
- Original backward: `[0.25, 0.25, 0.25, 0.25]`.
- Forward Jacobian and central finite differences: `[0.25, 0.75, 0.75, 2.25]`.
- Focused suite: 8 failures out of 58 before the two-line repair; 58/58 pass after it.

Evidence and exact reproduction:
https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/audits/2026-09-09-nntrainer-average-pool-padding

Full GERO article:
https://www.gero.uz/research/articles/nntrainer-average-pool-asymmetric-padding.html

Tested upstream source:
https://github.com/nntrainer/nntrainer/blob/a7ea056e79ab8e14447ea305c1b634e233343258/nntrainer/layers/pooling2d_layer.cpp

History boundary: PR #1360 from 2021 already contained the correct bottom/right backward boundaries, so this publication does not claim first discovery or established novelty:
https://github.com/nntrainer/nntrainer/pull/1360

Scope: macOS arm64, native CPU, FP32, NCHW, one runtime thread. FP16, NHWC, released-product impact, security impact and upstream acceptance were not established.

Narration uses a synthetic voice for the fictional digital narrator Alex Vector. No real-person voice cloning, stock footage or music. Original diagrams. Research and editorial review by Xamit Kadirbekov / GERO, published on Technology Product.

#MachineLearning #NumericalComputing #Cpp #Samsung #Shorts

Audience: not made for children. Language: English.

Suggested pinned comment: The full evidence package includes the native test patch, before/after XML, finite-difference check, SHA-256 manifest and public-history review: https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/main/audits/2026-09-09-nntrainer-average-pool-padding
