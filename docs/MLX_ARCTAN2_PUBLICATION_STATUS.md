# MLX arctan2 audit — publication record

10 September 2026 · Xamit Kadirbekov

For positive C, atan2(C*t,C) equals atan(t), so its derivative at t=1 is 0.5. MLX 0.32.2 CPU returns zero or infinity at extreme representable scales. A fresh native rerun reproduced all original rows: 1368 comparisons, 772 mismatches before and zero after. Four real dtypes are covered; higher derivatives and broadcasting are checked for float32/64. Both failed intermediate patch stages are preserved.

| Channel | Verified publication |
|---|---|
| GitHub | [Report, reproduction, patch and failed variants](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/6416016f0d508a5482ed60144d2170162da20e6f/audits/2026-09-10-mlx-arctan2-scale-autodiff/) |
| GERO | [Article and canonical evidence ZIP](https://www.gero.uz/research/articles/mlx-arctan2-scale-autodiff.html) |
| LinkedIn | [Public post](https://www.linkedin.com/feed/update/urn:li:share:7503731652662026240/) |
| YouTube | [Public Short](https://youtube.com/shorts/AUo3ys0DQ3E) |
| Zenodo | [10.5281/zenodo.22685198](https://doi.org/10.5281/zenodo.22685198) |

Evidence commit: `6416016f0d508a5482ed60144d2170162da20e6f`. Canonical archive SHA-256: `253617dd0454251647f08c77d3c8fc2aa1291d5121963bb7011ad5644fcaf50a`; 342987 bytes, 64 files. The archive includes the source pins, portable runner, actual result logs, patch, intermediate failures, English/Russian reports, licenses and checksums.

Native validation sequentially rebuilds the test and both baseline/patched primitives.cpp, linking against an existing CPU archive. It is not a clean current-main build. Every before/after JSON row matches the original completed audit. Child CPU time for this build/link/run preparation was about 4.27 seconds; this is not a benchmark. One numerical thread, no GPU arithmetic. Original source-audit files were not modified.

The 58.1-second Short uses original evidence diagrams and disclosed synthetic Samantha narration by fictional Alex Vector. Six cards were visually checked, audio/video fully decoded, and continuous YouTube Studio playback reached the final frame with six manually uploaded English (United States) caption cues. YouTube confirmed public publication; LinkedIn returned its success notification and a canonical post link. CPU rendering used one encoder thread.

The article deployment `dpl_AKnA21yHBZutmZ3nVfvLKFejrsY4` preserved 254 prior files byte for byte; the other three received only the new research card, RSS item and sitemap URL. Twelve live byte comparisons passed. A later DOI update `dpl_DUPajRZHL6XjeP5kXmAQkrpatz3D` preserved 323 of 327 production files byte for byte and inserted only a DOI paragraph into each of four audit articles. All four changed public pages matched the staged files exactly. The research index now contains 50 articles.

The previously blocked Zenodo queue is complete: this report and the three earlier common-offset logcumsumexp, expm1-tail and inverse-hyperbolic reports were deposited separately. Public unauthenticated downloads of all four archives match their canonical ZIPs byte for byte. See the [four-record DOI and checksum register](MLX_ZENODO_PUBLICATION_RECORDS.md). No duplicate record or pending DOI is required.

GPU, speed, compile/vmap, complete subnormal/nonfinite behavior, the undefined origin, full-model effects and absolute priority remain outside the established results. A bounded search found no direct MLX duplicate. No upstream MLX issue or PR was submitted, and upstream acceptance is not claimed. This is AI-assisted independent research and publication.
