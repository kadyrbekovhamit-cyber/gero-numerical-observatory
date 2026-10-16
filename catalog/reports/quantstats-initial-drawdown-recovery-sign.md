## Two reporting defects, two separate proposed fixes

A performance report should preserve a strategy's losses. In quantstats 0.0.86 we reproduced two independent defects: a drawdown starting on the first observation can disappear from the period table, and Recovery Factor loses the sign of the return numerator.

The fixes are submitted as [PR #577](https://github.com/ranaroussi/quantstats/pull/577) and [PR #578](https://github.com/ranaroussi/quantstats/pull/578). Both await upstream review. No accepted fix or official corrected release is claimed.

## One three-day example, an exact reference

Use daily returns -5%, -5%, and +2%, on 1–3 January 2024, and an initial equity of 1. Rational arithmetic gives equity 0.95, 0.9025, 0.92055. The running peak remains 1, so the worst drawdown is -0.0975 = -39/400. It starts on January 1, reaches its valley on January 2 and remains unrecovered on January 3.

```python
import pandas as pd
import quantstats as qs
r = pd.Series([-0.05, -0.05, 0.02], index=pd.date_range("2024-01-01", periods=3))
print(qs.stats.drawdown_details(qs.stats.to_drawdown_series(r)))
print(qs.stats.recovery_factor(r))
```

| Quantity | Original | With the corresponding isolated patch |
|---|---|---|
| Drawdown period table | Empty | January 1–3, valley January 2, max drawdown -9.75% |
| Recovery Factor | +0.8205128205 | -0.8205128205 |

These are two **separate** patch executions. Testing one patch does not imply that the other was applied. The recorded original, drawdown-only and recovery-only results are supplied in the archive.

## Why the first drawdown vanishes

In [the pinned stats module](https://github.com/ranaroussi/quantstats/blob/9ef4c6d0a4ffe7831f6ebba6a5824fd33a90c05a/quantstats/stats.py), `drawdown_details` detects the beginning of a period by looking for a transition from a zero drawdown to a negative one. The shifted first value is missing. The first row therefore has no start marker even when it is already underwater.

The focused patch gives the shifted boolean mask a `True` initial fill, representing the no-drawdown state immediately before the first observation. This also covers an initial drawdown that subsequently recovers. A six-day full-report regression separately verifies Max Drawdown -10%, Longest DD Days 6, and the benchmark control. It does not claim that every quantstats drawdown statistic was wrong: `max_drawdown` already returns the correct -9.75% in the small example.

## Why a loss has a positive Recovery Factor

The function computes an excess return numerator, then takes its absolute value. That turns a negative numerator into a positive score. For the library's current sum-of-returns convention with `rf=0`, the numerator is -0.08, so the independent ratio is (-0.08)/0.0975 = -32/39.

PR #578 removes the absolute value from the numerator and retains a positive drawdown magnitude in the denominator. It preserves the library's existing summation and risk-free-return convention; it does not replace it with compounded return or a different annualization. The [-0.05, -0.05, +0.02] compounded return is -7.945%, which is different from the -8% numerator used by this function.

The [official MQL5 statistics reference](https://www.mql5.com/en/docs/constants/environment_state/Statistics) defines Recovery Factor using net profit and maximum balance drawdown. This supports retaining the numerator's sign, but does not establish that every platform has identical conventions. If quantstats intentionally wants a nonnegative magnitude score, maintainers may choose to document that interpretation instead.

## Validation and prior art

Pinned main: `9ef4c6d0a4ffe7831f6ebba6a5824fd33a90c05a`. Python 3.12, NumPy 1.26.4, pandas 2.2.3, SciPy 1.13.1, pytest 8.3.3. Offline numerical executions were sequential on CPU.

- Drawdown patch: seven new cases; four fail and three pass before the source fix. The complete local suite with this patch has **232 passed**.
- Recovery patch: seven new cases; five fail and two pass before the source fix. The complete local suite with this patch has **232 passed**.
- The Recovery Factor checks include Series, DataFrame, mixed gain/loss columns, nonzero risk-free input, and zero-drawdown behavior.

On 17 October 2026, public issue/PR titles, bodies and comments, recent records and relevant changes were checked. #541 and #576 concern baseline inference or phantom indexing; #31/#233/#426 discuss dates/indexing; #463 concerns DataFrame handling. No exact matching fix for these two symptoms was found in that bounded review. This is not a worldwide-priority guarantee.

## Scope, attribution and reuse

Synthetic inputs only. No brokerage, investment recommendation, production portfolio loss, security vulnerability or regulatory conclusion is claimed. Maintainer review may change the proposed behavior or patch.

Investigation and report: Xamit Kadirbekov and Daniyal Kadirbekov / GERO, with AI assistance in investigation, code and writing. The reported checks were executed locally. Report CC BY 4.0; original reproduction MIT; upstream source and submitted changes retain Apache-2.0 terms.
