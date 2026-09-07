# Losses and autodiff audit — publication status

Published and verified on 7 September 2026, following the author's explicit authorization.

- [GERO technical note and evidence download](https://www.gero.uz/research/articles/when-small-losses-and-gradients-disappear.html)
- [GitHub evidence at the published commit](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/01cf73b86c8099efc258039d94e3b6f79d82a910/audits/2026-09-07-losses-autodiff)
- [LinkedIn public post](https://www.linkedin.com/feed/update/urn:li:share:7502621453343911936/)
- [YouTube Short: Zero Loss, Missing Gradient](https://www.youtube.com/shorts/fcOZBV1P6mA)

Three reproduced implementation cases in two cause families. Local repairs pass 31 BCE tests per device (CPU and Metal), 438 native MLX gradient checks on CPU and 77 nntrainer tests on CPU. The native MLX gradient patch is CPU-tested only. Maintainer acceptance, absolute novelty, model-level impact and a version-to-version regression are not established.

The Short is 50 seconds, 1080×1920 at 30 fps, with synthetic English narration, burned-in captions and a manually uploaded English (United States) subtitle track. YouTube reported no copyright-check issues. Its automatic thumbnail is used because a custom thumbnail requires channel phone verification.

The evidence commit is `01cf73b86c8099efc258039d94e3b6f79d82a910`. The [existing ONNX CI](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/actions/runs/34093003750) passed; the MLX and nntrainer audit executions are recorded separately in the evidence package.

Evidence ZIP SHA-256: `95e949524548b3599922354cf997b38e9fed9a841cbda5fbbbc98f0d0fc5211e`.
Video SHA-256: `21dd2bf818925ced18afb6e4381d40d01b018d64e8e48edfd91e3493591b91b6`.

The released evidence archive is preserved unchanged. Its preparation-time publication status is a historical snapshot; this document records the completed publication.
