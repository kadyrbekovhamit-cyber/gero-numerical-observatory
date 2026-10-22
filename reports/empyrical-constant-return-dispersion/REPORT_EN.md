# Empyrical: constant returns and floating-point Sharpe ratios

22 October 2026 · GERO Numerical Assurance · Xamit Kadirbekov

**REPRODUCED IN THE STATED SCOPE; FIX PROPOSED.** The upstream PR is open. Acceptance and a corrected release are not claimed.

Exactly constant returns can produce huge finite ratios instead of the documented signed infinity. A focused fix passes all 860 package tests.

Upstream submission: [https://github.com/stefan-jansen/empyrical-reloaded/pull/124](https://github.com/stefan-jansen/empyrical-reloaded/pull/124)

Closes #123. Related contract: #80/#81.

For identical nonzero returns, floating-point centering can produce a tiny nonzero sample standard deviation. For example, 252 copies of `0.001` currently produce an excess Sharpe around `1.53e15` and Sharpe around `2.44e16`, rather than the intended positive infinity.

Normalize the computed deviation to zero only when the observed values are exactly equal and the deviation is finite. This preserves undefined NaN deviations, does not threshold near-constant data, and keeps the existing annualization and signed-infinity convention. Apply the helper to both Sharpe functions.

The 13 public-API regression cases cover several lengths and positive/negative constants, plus constant and varying DataFrame columns together. The existing insufficient-observation tests are unchanged.

Validation (Python 3.14, NumPy 2.5.4, pandas 3.0.6):

- Base `a54620df2077f56e8ef126d9b230b2ec2e1fadf6` with new tests: `PYTHONPATH=src python -m pytest tests/test_zero_dispersion_ratios.py -q` → **8 failed, 5 passed**.
- Fix `6d255c8`: `PYTHONPATH=src python -m pytest tests/ -q` → **860 passed**.
- Black 24.4.2 and flake8 on the changed source and new test file: **pass**.

All runs used one local CPU worker. AI assistance: Codex reviewed the candidate fix, repeated the actual public-API regressions and complete suite, and prepared this patch description. Independent human review is not claimed; maintainer review is requested.


## Scope and attribution

This is an AI-assisted numerical correctness investigation using synthetic inputs and, where stated, the upstream test fixtures. There is no evidence here of executed trades, client losses, a security exploit, or effects in a commercial deployment. Independent human review is pending. This report is CC BY 4.0; upstream code retains its original license.
