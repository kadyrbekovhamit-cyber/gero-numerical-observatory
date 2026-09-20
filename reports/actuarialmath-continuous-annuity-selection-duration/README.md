# Complete reproduction packet

Download and extract [gero-actuarialmath-annuity-selection-evidence-2026-09-20.zip](gero-actuarialmath-annuity-selection-evidence-2026-09-20.zip) first. The ZIP contains the runnable sources and expected outputs; loose files are browsing excerpts.

# GERO continuous-annuity selection-duration evidence

This packet contains the pinned public actuarialmath source and official 1.1.0 wheel, an independent Decimal reference, a one-line candidate, restoration control, raw numerical rows and synthetic JSON memo/read-back consumer. REPORT_EN.md states the result, versions and limits. No real client data is included.

Use Python 3.12 and Git, from a fresh extracted copy:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-repro.txt
.venv/bin/python -B reproduce.py
```

The exact direct dependency versions are pinned. Installation needs network; replay itself is offline and does not post, send, play media or access accounts. It creates candidate/restored copies from current source, applies and reverses the patch with Git, checks the complete source manifest and all 22 release modules against the official wheel, then executes four variants sequentially. Standard numerical-library thread variables are set to 1; this is not a machine-wide CPU cap.

Expected: 2,160 calls per variant; 648/648/0/648 failures current/release/candidate/restored. 1,512 passing controls stay unchanged. 59 Decimal controls per variant repeat at 120 digits. Explicit Uniform-class checks accompany the main Annuity grid. The synthetic written memo changes 9166.67 → 8750.00 and its read-back ceiling decision OVER_LIMIT→WITHIN_LIMIT. These are continuous payment rates, not annual discrete payments.

Immutable expected outputs are in expected/. Generated outputs appear in evidence/, downstream/ and REPLAY_RECEIPT.json. The wrapper checks exact serialized numerical values in the documented macOS arm64 environment; other platforms may differ in final bits and require review. Do not silently loosen tolerances. Historical receipts under expected/ retain the status at their original time; the outer publication receipt records later distribution.

The package does not validate every actuarial method, invoke the full upstream suite or prove insurer/customer impact. It does not correct the separate Uniform.temporary_annuity shortcut. Original source license notices are retained. GERO code MIT; original report CC BY 4.0. No GPU/audio playback.
