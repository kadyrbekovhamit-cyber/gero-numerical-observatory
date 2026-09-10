# MLX inverse hyperbolic audit — publication record

10 September 2026 · Xamit Kadirbekov

The report covers one shared real-derivative overflow defect in arcsinh and arccosh. At float16 x=1000, finite forward values coexist with zero JVP/VJP instead of a representable slope near 0.001. A fresh isolated CPU rerun reproduces 188 mismatches across 518 comparisons before the patch and zero after. All original completed-audit JSON rows match. Higher derivatives are tested only for float32/float64.

| Channel | Verified publication |
|---|---|
| GitHub | [Evidence, reproduction, patch and source pins](https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/tree/e23471189c5111d6080f6c7ab20b3ef9f6d66b9f/audits/2026-09-10-mlx-inverse-hyperbolic-overflow/) |
| GERO | [Article and downloadable archive](https://www.gero.uz/research/articles/mlx-inverse-hyperbolic-overflow.html) |
| LinkedIn | [Public post](https://www.linkedin.com/feed/update/urn:li:share:7503723054225776640/) |
| YouTube | [Public Short](https://youtube.com/shorts/TxgtkhNph2A) |

GitHub evidence commit: `e23471189c5111d6080f6c7ab20b3ef9f6d66b9f`. GERO deployment: `dpl_Ah48BMHQtZUaLNgi36bzA2Lktfnn`. The preceding 204-file production snapshot was preserved, with only the authorized research index, RSS and sitemap edits. Eleven live byte comparisons passed, including both earlier September 10 articles, the new article, ZIP, JSON results, MP4 and VTT.

LinkedIn returned “Post successful.” YouTube returned its publication-success dialog. English (United States) VTT captions were uploaded manually and saved. Synthetic Samantha narration is disclosed in the description and on the original graphics. The preview played through the final frame; all six cards, cue timing and complete audio decoding were checked. Native calculations and rendering were sequential with one numerical/encoder thread and no GPU calculation.

Archive: `mlx-inverse-hyperbolic-overflow-2026-09-10.zip` (181978 bytes). SHA-256: `057503a3d10a4e95cd979e786a4b69d5817e051fad6cedc876e5067180b297a0`.

**Zenodo is published:** [10.5281/zenodo.22685280](https://doi.org/10.5281/zenodo.22685280), version 1.0.0. The earlier HTTP 504/navigation failures were resolved later on 10 September 2026. The public record identifies Xamit Kadirbekov, preserves the canonical archive and its license notices, and the unauthenticated ZIP download matches the local archive byte for byte. See the [four-record Zenodo register](MLX_ZENODO_PUBLICATION_RECORDS.md).

No full current-main build, GPU behavior, performance, compile/vmap, complete nonfinite behavior, full-model effect, absolute discovery priority or upstream acceptance is claimed. No upstream MLX issue or PR was submitted; this is the independent author's own evidence repository. AI assistance is disclosed.
