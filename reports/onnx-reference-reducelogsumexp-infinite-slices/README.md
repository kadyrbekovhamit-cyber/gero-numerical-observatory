# ONNX ReferenceEvaluator turns infinite ReduceLogSumExp slices into NaN

Authors as published: Xamit Kadirbekov. First publication: 2026-09-09.

Slices [inf, -inf] and [-inf, -inf] both become NaN, although the exact results and ONNX Runtime give +inf and -inf respectively.

This article was already archived in the 45-item GERO collection (https://doi.org/10.5281/zenodo.22683900). This individual record provides a separate citation; it is not a new finding or a new experiment.

- [GERO article](https://www.gero.uz/research/articles/onnx-reference-reducelogsumexp-infinite-slices.html)
- [Readable article](article.txt)
- [Original HTML](article.html)
- [Source and publication metadata](metadata.json)

This archive preserves the claims and limitations of the original report. It makes no new claim about current upstream status, maintainer acceptance, security impact, monetary rewards or priority. Original GERO article and archival metadata: CC BY 4.0. Source code, third-party material and linked artifacts retain their own licenses; no blanket relicensing is intended.

## Publication records

- [GERO](https://www.gero.uz/research/articles/onnx-reference-reducelogsumexp-infinite-slices.html)
- [Individual Zenodo record](https://doi.org/10.5281/zenodo.22694989)
- [Earlier collection archive](https://doi.org/10.5281/zenodo.22683900)
- [LinkedIn](https://www.linkedin.com/feed/update/urn:li:share:7503826526207963136/)
