# Primary sources and review boundaries

- [Gaussian NLL API](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.gaussian_nll_loss.html): variance, epsilon, constant and reduction contract.
- [Pinned loss implementation](https://github.com/ml-explore/mlx/blob/ce916dbbcaa88e433b6fd1e60a17f766d49c27fe/python/mlx/nn/losses.py#L330): residual square before division.
- [Pinned Divide VJP](https://github.com/ml-explore/mlx/blob/ce916dbbcaa88e433b6fd1e60a17f766d49c27fe/mlx/primitives.cpp#L1806): denominator square in backward.
- [Retained measurements](evidence/mlx-review-results.json): observed outputs, derivatives, counterexample and runtime provenance.
- [Bounded duplicate searches](evidence/duplicate-searches.json), 9 September 2026: Gaussian query returned two documentation/typo PRs; gradient-overflow query returned four results concerning a loop feature, attention training, recursion and memory overflow. No exact Gaussian gradient-sign report was identified in this scope. Search history, comments, private reports and every spelling variant were not comprehensively reviewed. Initial network errors were retried successfully; no novelty claim follows from this search.
- [nntrainer Upsample source](https://github.com/nntrainer/nntrainer/blob/a7ea056e79ab8e14447ea305c1b634e233343258/nntrainer/layers/upsample2d_layer.cpp): related negative result at a different project.

The user's GitHub tree contained no Gaussian audit at publication preparation.
LinkedIn's visible recent posts were checked back through the previous day;
the latest post was the Dropout case study. The current GERO production file
inventory was checked before adding the new article.
