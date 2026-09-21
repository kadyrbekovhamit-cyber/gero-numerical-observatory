# Sources and claim ledger — MLX log10

Prepared 21 September 2026. Source links are evidence, not an assertion of endorsement or acceptance.

| Source | Supported proposition and limit |
| --- | --- |
| https://github.com/ml-explore/mlx/tree/59d600b5e64c238427d0f8d897ab7c682ef4d3d2 | Exact public MLX source executed through its complete CPU library; pin is not a claim of release-binary execution. |
| https://github.com/ml-explore/mlx/blob/59d600b5e64c238427d0f8d897ab7c682ef4d3d2/mlx/primitives.cpp | `Log::jvp`/`Log::vjp` evaluation order, seed divided by input before base-dependent scaling. |
| https://github.com/ml-explore/mlx/releases/tag/v0.32.2 | Release identity; archived affected source bodies match main. Installed release import failed before calculation. |
| https://github.com/ml-explore/mlx/pull/3605 | Reviewed prior complex VJP conjugation work; distinct from positive real finite-seed overflow. |
| https://github.com/ml-explore/mlx/pull/4266 | Reviewed prior logarithm compilation-equivalence work; distinct root cause. |
| https://github.com/ml-explore/mlx/blob/main/.github/ISSUE_TEMPLATE/bug_report.md | Live raw GitHub API retrieved 21 September includes prohibition on AI-written issues. Web-indexed page was older and omitted that line; live source takes precedence. No author attestation submitted. |
| https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits | Original dataset provenance and CC BY4.0 attribution. |
| https://doi.org/10.24432/C50P49 | Alpaydin and Kaynak1998 dataset citation. |
| https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html | Provenance of the 1797-image8×8 copy. The actual installed exporter version is recorded in DATASET.json. |
| https://creativecommons.org/licenses/by/4.0/ | License retained on transformed public digit data; separate code licenses remain with the software. |
| `evidence/primitive/oracle.json`, `check_rounding.py`, `CANDIDATE_RECEIPT.json` | Independent derivative references, actual native outputs,66finite-to-nonfinite failures,4candidate boundary residuals, restoration. Local measured evidence. |
| `evidence/model/EXPERIMENT_PLAN.json`, `DATASET.json`, `SELECTION.json` | Predeclared design, exact input split, train-only selections; all3selected points have label8. |
| `evidence/model/IMPACT_RECEIPT.json`, `CONTROL_COMPARISON.json`, raw CSV files | Actual CPU classifier/update/inference measurements; this is a GERO stress experiment, not a deployed incident. |

## Claim classification

- **Measured:** actual native JVP/VJP nonfinite outputs where independently rounded references are finite;294unique inputs.
- **Measured:** one frozen classifier,3independent stress steps,65/65nonfinite parameters and108/108nonfinite logits/probabilities without guard; original/restored match.
- **Measured:** experimental candidate equals ordinary control weights and held-out predictions bit for bit in all3steps; scales1and1e20unchanged.
- **Measured:** GERO guard skips the step and keeps checkpoint byte-identical with103/108correct.
- **Measured limitation:** four candidate boundary mismatches remain; no complete correction claim.
- **Inference:** applications that propagate such nonfinite updates without checking could lose a usable checkpoint. Frequency and real-application prevalence are not established.
- **Not established:** customer loss, Apple-device exposure, release-wheel execution, maintainer acceptance, full-suite correctness, performance cost, general-model robustness.

Original GERO experiment code and narrative are AI-assisted. Preserve MLX MIT, fmt MIT, json MIT, and datasetCCBY4.0 licenses. No third-party media is used; a future cartoon is explanatory original artwork, not device-test footage.
