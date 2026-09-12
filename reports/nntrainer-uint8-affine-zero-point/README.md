# A scale-dependent zero point silently corrupts UINT8 affine quantization in nntrainer

Author: Xamit Kadirbekov. Original audit: 3 September 2026. Archival republication: 10 September 2026.

A three-value UINT8 input collapses a negative endpoint and real zero into the same code. The pinned evidence package records a candidate ratio-formula repair and 15 passing local tests.

- [Pinned original report and evidence](https://github.com/kadyrbekovhamit-cyber/gero-nntrainer-quantization-audit/tree/89d07cf2bb051803b8537741bd08dc6c5e83e3e7)
- [GERO article](https://www.gero.uz/research/articles/nntrainer-uint8-affine-zero-point.html)
- [Readable article](article.txt)
- [Existing LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:7501930972641124352/)

No numerical tests were rerun for this archival deposit. The report preserves the rejected UINT4 hypothesis and bounded novelty search. Original GERO article and metadata: CC BY 4.0. Code and copied material retain their original notices.

## Publication records

- [GERO](https://www.gero.uz/research/articles/nntrainer-uint8-affine-zero-point.html)
- [Individual Zenodo record](https://doi.org/10.5281/zenodo.22696596)
- [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7501930972641124352/)
- [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7501307096056111104/)
- [Hugging Face document](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/nntrainer-uint8-affine-zero-point.md)
