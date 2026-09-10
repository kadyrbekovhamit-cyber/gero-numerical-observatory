# Sorted Indices, Wrong Gradients: MLX Gather

Authors as published: Xamit Kadirbekov. First publication: 2026-09-09.

Valid sorted-index calls produce incorrect gather_mm and affine gather_qmm gradients. A bounded local FP32 CPU repair passes 69 scenarios and 626 checks; quantized FP16 fallback remains a limitation.

This article was already archived in the 45-item GERO collection (https://doi.org/10.5281/zenodo.22683900). This individual record provides a separate citation; it is not a new finding or a new experiment.

- [GERO article](https://www.gero.uz/research/articles/mlx-gather-vjp-preconditions.html)
- [Readable article](article.txt)
- [Original HTML](article.html)
- [Source and publication metadata](metadata.json)
- [LinkedIn publication](https://www.linkedin.com/feed/update/urn:li:activity:7503473794536144896/)

This archive preserves the claims and limitations of the original report. It makes no new claim about current upstream status, maintainer acceptance, security impact, monetary rewards or priority. Original GERO article and archival metadata: CC BY 4.0. Source code, third-party material and linked artifacts retain their own licenses; no blanket relicensing is intended.
