# MLX scan-shift and expm1-tail audits — publication record

10 September 2026 · Xamit Kadirbekov

These are two separate implementation reports. The known CPU float64-exp limitation in [MLX #3047](https://github.com/ml-explore/mlx/issues/3047) is a repair dependency, not a third new finding. The expm1 report credits the analogous [JAX #39794](https://github.com/jax-ml/jax/issues/39794). Neither report claims absolute priority, production qualification, full-model impact or upstream acceptance.

| Report | Evidence | Article | LinkedIn | YouTube |
|---|---|---|---|---|
| logcumsumexp: large common shift | [GitHub](../audits/2026-09-10-mlx-logcumsumexp-shift-vjp/) | [GERO](https://www.gero.uz/research/articles/mlx-logcumsumexp-shift-gradient.html) | [Post](https://www.linkedin.com/feed/update/urn:li:share:7503699307825369088/) | [Short](https://youtube.com/shorts/Y0XoB3Lllxs) |
| expm1: disappearing tail derivative | [GitHub](../audits/2026-09-10-mlx-expm1-tail-vjp/) | [GERO](https://www.gero.uz/research/articles/mlx-expm1-tail-vjp.html) | [Post](https://www.linkedin.com/feed/update/urn:li:share:7503699618493517824/) | [Short](https://youtube.com/shorts/hsGTSW1Qhjo) |

The existing logcumsumexp publication was retained when the expm1 package was added. Evidence commits are `83c2ef82dcf9210cbef0a8f137f27519edb55a75` and `46f1d22b329ef94eecab22b6fac8f0a7393b10b8`, respectively. English captions and synthetic-narration disclosures accompany both videos. Both LinkedIn posts returned publication-success confirmations; both YouTube videos returned public-publication confirmations.

The expm1 site update preserved the preceding 150-file production snapshot. Ten live byte comparisons verified the article, ZIP, JSON results, video, VTT, index, feeds and existing logcumsumexp article against the staged files. Numerical reruns were sequential and CPU-only; one numerical thread was configured. This is not a claim of physical CPU affinity. The recurrence + known-exp control passes 1,122 gradient and 477 higher-derivative comparisons; the expm1 + known-exp control passes 216 comparisons.

**Zenodo is not published.** Repeated attempts to open its homepage and upload UI returned HTTP 504 or navigation timeouts. No new record, upload or DOI was created. The versioned evidence archives remain downloadable from the GERO articles, with source pins, patches and checksums.

No upstream MLX issue or PR was submitted as part of this publication. MLX's [contribution policy](https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/CONTRIBUTING.md) prohibits AI-written posts; these reports are published in the independent author's own repository and channels, with AI assistance disclosed.
