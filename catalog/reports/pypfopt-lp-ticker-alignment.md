# PyPortfolioOpt: a ticker-order mismatch breaks the allocation budget

22 October 2026 · GERO Numerical Assurance · Xamit Kadirbekov

**REPRODUCED IN THE STATED SCOPE; FIX PROPOSED.** The upstream PR is open. Acceptance and a corrected release are not claimed.

A synthetic $10,000 allocation costs $500,050 when priced by ticker. A one-line fix and regression tests have been submitted.

Upstream submission: [https://github.com/PyPortfolio/PyPortfolioOpt/pull/767](https://github.com/PyPortfolio/PyPortfolioOpt/pull/767)

`lp_portfolio` pairs the weights in dictionary order with `latest_prices.values` in Series order. When those orders differ, the returned share counts are attached to the wrong tickers, and the reported leftover can disagree with the cost of the returned allocation.

For example, with a $10,000 budget:

```python
weights = {"BBB": 0.5, "AAA": 0.5}
prices = pd.Series({"AAA": 10.0, "BBB": 1000.0})
DiscreteAllocation(weights, prices).lp_portfolio(solver="SCIPY")
```

Before this fix the allocation is `{'BBB': 500, 'AAA': 5}` with zero leftover. Pricing that allocation by ticker costs $500,050. After the fix it is `{'BBB': 5, 'AAA': 500}`, which costs $10,000. This is a synthetic reproduction, not a report of executed trades or losses.

Select prices by the weight tickers before constructing the optimization problem. No objective, solver policy or API change is needed. Tests cover both Series orders and a reversed price Series from the repository's stock-price fixture, checking the independent identity `sum(price[ticker] * shares) + leftover == budget`.

Validation on main `a6638d2e06dae6f444fd022cfd4b3c528902a85b`:

- Before the one-line fix: 2 new regression cases fail; the already aligned case passes.
- After: `python -m pytest tests/test_discrete_allocation.py -q` → **20 passed**.
- `git diff --check` passes.

Prepared with AI assistance as part of GERO's numerical correctness work. The examples and tests were executed locally.


## Scope and attribution

This is an AI-assisted numerical correctness investigation using synthetic inputs and, where stated, the upstream test fixtures. There is no evidence here of executed trades, client losses, a security exploit, or effects in a commercial deployment. Independent human review is pending. This report is CC BY 4.0; upstream code retains its original license.
