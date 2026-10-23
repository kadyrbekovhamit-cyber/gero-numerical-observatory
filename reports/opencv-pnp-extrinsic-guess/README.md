# OpenCV RANSAC: preserve the camera-pose initial guess

Float32 and row-vector initial poses are lost when output arrays are allocated before reading the supplied guess.

- Upstream PR: https://github.com/opencv/opencv/pull/30481

- GERO: https://www.gero.uz/research/articles/opencv-pnp-extrinsic-guess.html

See REPORT_EN.md and the reproduction archive. See the report for the exact upstream status; all six new reports have open proposed fixes; no merge or corrected release is claimed. AI assistance is disclosed. Report CC BY 4.0; original reproduction MIT; upstream licenses are retained.

Publication links verified 23 October 2026:
- github_catalog: https://github.com/kadyrbekovhamit-cyber/gero-numerical-observatory/blob/main/catalog/reports/opencv-pnp-extrinsic-guess.md
- huggingface: https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/opencv-pnp-extrinsic-guess.md
- linkedin: https://www.linkedin.com/feed/update/urn:li:activity:7519420557763854336/
- zenodo: https://doi.org/10.5281/zenodo.23599315
