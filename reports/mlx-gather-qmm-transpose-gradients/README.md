# A Missing Transpose in MLX Quantized Gradients

Authors as published: Xamit Kadirbekov. First publication: 2026-09-09.

With transpose=False, affine gather_qmm miscomputes scale and bias gradients. A local C++ orientation repair passes 205 scenarios and 379 checks; the measured quadratic update then lowers the loss.

This article was already archived in the 45-item GERO collection (https://doi.org/10.5281/zenodo.22683900). This individual record provides a separate citation; it is not a new finding or a new experiment.

- [GERO article](https://www.gero.uz/research/articles/mlx-gather-qmm-transpose-gradients.html)
- [Readable article](article.txt)
- [Original HTML](article.html)
- [Source and publication metadata](metadata.json)
- [LinkedIn publication](https://www.linkedin.com/feed/update/urn:li:activity:7503503941146681345/)

This archive preserves the claims and limitations of the original report. It makes no new claim about current upstream status, maintainer acceptance, security impact, monetary rewards or priority. Original GERO article and archival metadata: CC BY 4.0. Source code, third-party material and linked artifacts retain their own licenses; no blanket relicensing is intended.
