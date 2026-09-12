# Dinero.js toDecimal accepts non-decimal bases and returns plausible wrong strings

Authors as published: Xamit Kadirbekov. First publication: 2026-09-06.

Scalar base 240 is accepted and formatted as 1.10 for the exact value 250/240, while the equivalent multi-base currency [20, 12] is correctly rejected.

This article was already archived in the 45-item GERO collection (https://doi.org/10.5281/zenodo.22683900). This individual record provides a separate citation; it is not a new finding or a new experiment.

- [GERO article](https://www.gero.uz/research/articles/dinerojs-todecimal-non-decimal-base-guard.html)
- [Readable article](article.txt)
- [Original HTML](article.html)
- [Source and publication metadata](metadata.json)
- [LinkedIn publication](https://www.linkedin.com/feed/update/urn:li:activity:7502311825464606720/)

This archive preserves the claims and limitations of the original report. It makes no new claim about current upstream status, maintainer acceptance, security impact, monetary rewards or priority. Original GERO article and archival metadata: CC BY 4.0. Source code, third-party material and linked artifacts retain their own licenses; no blanket relicensing is intended.

## Archival correction — 13 September 2026

Read [the explicit erratum](ERRATUM-2026-09-13.md) with the historical article: 250/240 is not exactly binary-representable. The counterexample still follows from exact rational arithmetic. Original article files and test results are preserved. The individual Zenodo archive includes the correction.

## Publication records

- [GERO](https://www.gero.uz/research/articles/dinerojs-todecimal-non-decimal-base-guard.html)
- [Individual Zenodo record](https://doi.org/10.5281/zenodo.22728993)
- [Earlier collection archive](https://doi.org/10.5281/zenodo.22683900)
- [LinkedIn](https://www.linkedin.com/feed/update/urn:li:share:7504616775582539776/)
- [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7502311825464606720/)
- [Hugging Face document](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/dinerojs-todecimal-non-decimal-base-guard.md)
