# PyLoan explicit-payment regression evidence

Independent GERO research by Xamit Kadirbekov, 18 September 2026. Synthetic inputs only; AI-assisted preparation. Read REPORT_EN.md for the result, scope and retained limitations. Vendor issue 70 precedes publication.

## Reproduce

Use Python 3.12, install `python -m pip install -r requirements.txt`, then run `python -B reproduce.py`. It uses the included hash-checked official 0.7.0 and 0.7.2 wheels and pinned current modules. Jobs run sequentially with library thread settings of one. No GPU, audio, paid services or network is needed during the replay after installing python-dateutil.

New outputs go to `evidence/`. Recorded outputs in `expected/` remain separate. The replay verifies their byte-for-byte agreement for all five 864-scenario grid files.

Expected primary result: release/current/restored code violates the specified-payment cap in 408 scenarios; the candidate and 0.7.0 do not. All 456 previously passing schedules and 60 additional controls are unchanged. Five focused regression tests produce 2 / 0 / 2 failures for original / candidate / restored code.

The upstream discovery runs 16 tests and retains three baseline failure reports in every patch state. Twenty-four candidate scenarios still differ from the exact Decimal oracle by up to 0.02 currency units. Do not interpret this patch as fixing all numerical behavior or as an accepted upstream contribution.

The candidate targets explicit annuities only. `reproduce.py` generates it from the pinned source and runs an unmodified restoration. The main grid uses ordinary regular payments, complete 30/360 periods, and no special payments, fees or grace periods. A valid early final payment can be smaller than the specified amount. No bank deployment, borrower loss, security or legal conclusion was measured.

## License

Upstream code is MIT-licensed; original notices are preserved in `LICENSE` and `source/LICENSE`. Original GERO probes, report and patch are also provided under MIT, copyright 2026 Xamit Kadirbekov. Downloaded wheels retain their own embedded notices.
