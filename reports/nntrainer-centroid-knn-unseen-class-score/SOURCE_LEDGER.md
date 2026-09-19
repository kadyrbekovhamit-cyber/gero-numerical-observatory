# Source and evidence ledger

- nntrainer current main: https://github.com/nntrainer/nntrainer/commit/2d1e4974ae84e1e6d37784416bdfb52b647919a6 . Executed native CPU source. All 2,005 archived files have recorded Git blob IDs and SHA-256 digests in source-manifest.json.
- Exact layer: https://github.com/nntrainer/nntrainer/blob/2d1e4974ae84e1e6d37784416bdfb52b647919a6/nntrainer/layers/centroid_knn.cpp . SHA-256 e0d06d603bf2c31b97e08d5fcb2d18c93b7a2c451974acf703a3d4683ddeb7ed.
- Reviewed release source: https://github.com/nntrainer/nntrainer/blob/v0.5.0/nntrainer/layers/centroid_knn.cpp . Source inspected, released binary not run. SHA-256 0dd24ce936f75d2c7bcf40f7297146a0829d781807138d2d0d97a666b3b69593.
- Official public API: pinned api/ccapi/include/{model,layer,dataset}.h and five unchanged api/ccapi/src files. Public-model score extraction and max_element selection are GERO demonstration code.
- Public historical context: nntrainer PRs859,1394,1499,1580,1585,1702 and issue1521. See evidence/DUPLICATE_REVIEW.md for boundaries and unresolved historical ambiguity.
- Official developer report: https://github.com/nntrainer/nntrainer/issues/4341 . Submitted with full reproducer and AI-assistance disclosure; not an accepted fix.
- Independent expected results: Python standard-library Decimal, 80-digit Euclidean distance and exact dyadic arithmetic-centroid construction. No nntrainer score is used as the expected result.
- Current source and dependency archives: files and hashes in source-manifest.json. nntrainer retains Apache-2.0; bundled dependencies retain their own licenses. Original GERO C++/Python reproducer code uses MIT; original prose is CC BY 4.0.
