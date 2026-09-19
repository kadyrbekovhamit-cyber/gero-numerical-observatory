# GERO CentroidKNN evidence package

Source pin: 2d1e4974ae84e1e6d37784416bdfb52b647919a6. Report submitted to https://github.com/nntrainer/nntrainer/issues/4341 before distribution. No upstream acceptance claimed.

Read REPORT_EN.md for results and limitations. The component grid gives 1,548/0/1,548 mismatches in 1,728 scenarios; public-model execution gives 2/0/2 wrong decisions across five checks with three fully populated controls unchanged. Both sequences are original/candidate/restored.

The included clean replay was executed successfully on macOS 15.5 ARM64 using Apple clang 17, Python, Meson and Ninja. From an empty working directory with those tools on PATH:

    python3 /path/to/package/replay.py

No network download is needed. The wrapper extracts bundled pinned sources, builds one CPU core library using ninja -j1, runs each variant, checks dynamic library paths, compares native results byte for byte and recomputes the independent Decimal oracle. It writes centroid-replay/evidence/REPLAY_RECEIPT.json. It intentionally refuses to reuse an existing replay directory.

No GPU or audio/video playback. One configured compiler/numerical worker; this is not an OS-wide core or power cap. Other platforms need the official nntrainer build setup and were not executed here.

Sources: source-manifest.json and sources/. Current Git blob IDs and SHA-256 digests cover 2,005 compact source files. Large unrelated repository assets are omitted. Official dependency code is unmodified. Darwin header includes and a build-directory malloc.h shim are platform compatibility only; neither changes the audited arithmetic.

Original raw grid files and dated historical verification receipts are in evidence/. An older receipt says the prior-art review was incomplete at that point. The later DUPLICATE_REVIEW.md and verified vendor receipt supersede that historical status; numerical files are unchanged. Historical issue #1521 has an unspecified accuracy-drop discussion and remains ambiguous.

Tests use synthetic feature vectors; no production model, real image, deployment, customer loss, cross-platform or full test-suite claim. An all-unobserved model and nonfinite/extreme distances need separate policy and validation. Original code/prose licenses are in LICENSE-GERO.txt; upstream archives retain their own notices.
