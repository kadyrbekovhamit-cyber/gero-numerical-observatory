# Source ledger

Checked 17 September 2026. All numerical executions use CPU float32.

## mlx

- Repository: https://github.com/ml-explore/mlx
- Commit: 59d600b5e64c238427d0f8d897ab7c682ef4d3d2
- Archive: sources/mlx-59d600b5e64c238427d0f8d897ab7c682ef4d3d2.tar.gz
- SHA-256: 425905d1c2b7c21c35cb86f0eeb5b6acfe5ae55ff7aa4413d002f39508f433a6
- Git blobs verified: 958

## fmt

- Repository: https://github.com/fmtlib/fmt
- Commit: 407c905e45ad75fc29bf0f9bb7c5c2fd3475976f
- Archive: sources/fmt-407c905e45ad75fc29bf0f9bb7c5c2fd3475976f.tar.gz
- SHA-256: 2bc1fe4a5b6d5d6a614239b4ca1d520e66e152a02d3d262684d26dfd6ab3438a
- Git blobs verified: 128

## json

- Repository: https://github.com/nlohmann/json
- Commit: 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03
- Archive: sources/json-9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03.tar.gz
- SHA-256: 0dbc5e40a01ff142e7e68c03e85247a4dcede2f592d12d3677dee3664d17975a
- Git blobs verified: 1090

## Reference and environment

- mpmath 1.3.0; official universal wheel and its PyPI SHA-256 are retained in sources/ and evidence/mpmath-wheel.json.
- Python 3.12.14; AppleClang 17.0.0.17000013; CMake/Ninja commands are recorded by the replay.
- MLX v0.32.2 Sqrt::jvp/vjp method source comparison: identical. Release runtime not executed.
- Original/candidate/restored are built from the same pinned complete source. Only the documented primitives.cpp branch changes in the candidate.
- Metal/CUDA disabled; one compiler worker; one numerical flow; no model, device or financial-loss measurements.
