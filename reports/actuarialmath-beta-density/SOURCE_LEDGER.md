# Source ledger — actuarialmath Beta density

- Official implementation: https://github.com/terence-lim/actuarialmath/blob/7d18f11ad304898f177b7922b3c53f70e4c2b4f4/src/actuarialmath/mortalitylaws.py . All 91 Git blobs were checked against the Git tree; this remained the default-branch head on September 17, 2026.
- Official release: https://pypi.org/project/actuarialmath/1.1.0/ . Wheel SHA-256 matches PyPI metadata and wheel RECORD entries were checked. Source packaging metadata is 1.0.1; do not describe the Git tree itself as release 1.1.0.
- Author guide: https://actuarialmath-guide.readthedocs.io/en/latest/mortalitylaws.html . The survival law supplies the mathematical contract; expectations were derived independently by differentiation/integration, not copied from the implementation. Its density typography is not treated as an executable oracle.
- Existing original report: https://github.com/terence-lim/actuarialmath/issues/5 , filed September 15. Open with zero comments at September 17 review. No maintainer acceptance claim.
- Prior UDD fractional-age density report/PR: https://github.com/terence-lim/actuarialmath/pull/2 . Different class and formula. All five issue/PR records reviewed, including already published variance and ConstantForce cases.
- Actual local executions: `recorded-evidence/`, `CURRENT_MASTER_REPLAY_RECEIPT.json`, `PORTABLE_REPLAY_RECEIPT.json`. Synthetic parametric examples only; no production insurer or policyholder measurements.
- Narration client: https://github.com/rany2/edge-tts . Preset Microsoft en-US-JennyNeural, rate -3%. Uses Microsoft's online service; only the reviewed public script is submitted. No audible playback during QA.
