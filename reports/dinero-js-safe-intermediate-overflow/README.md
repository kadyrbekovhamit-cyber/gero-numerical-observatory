# Dinero.js misallocates safe integers after an unsafe intermediate product

Authors as published: Xamit Kadirbekov. First publication: 2026-09-06.

All inputs remain JavaScript safe integers, yet binary64 rounding changes a 3-unit allocation from the exact [2, 1, 0] to [3, 0, 0].

This article was already archived in the 45-item GERO collection (https://doi.org/10.5281/zenodo.22683900). This individual record provides a separate citation; it is not a new finding or a new experiment.

- [GERO article](https://www.gero.uz/research/articles/dinero-js-safe-intermediate-overflow.html)
- [Readable article](article.txt)
- [Original HTML](article.html)
- [Source and publication metadata](metadata.json)
- [LinkedIn publication](https://www.linkedin.com/feed/update/urn:li:activity:7502311825464606720/)

This archive preserves the claims and limitations of the original report. It makes no new claim about current upstream status, maintainer acceptance, security impact, monetary rewards or priority. Original GERO article and archival metadata: CC BY 4.0. Source code, third-party material and linked artifacts retain their own licenses; no blanket relicensing is intended.
