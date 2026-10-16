# A shared constant flips two convolution biases in coremltools

A reproduced converter defect changes two subtractions into additions. A focused patch and six regression cases are submitted to Apple in PR #2901; review is pending.

- Upstream PR: https://github.com/apple/coremltools/pull/2901

- GERO: https://www.gero.uz/research/articles/coremltools-shared-convolution-bias.html

See REPORT_EN.md and the reproduction archive. Review is pending; maintainer acceptance is not claimed. AI assistance is disclosed. Report CC BY 4.0; original reproduction MIT; upstream licenses are retained.
