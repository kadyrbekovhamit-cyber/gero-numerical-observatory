# A zero guarantee with a minus-200 outcome: penaltyblog partial hedging

Xamit Kadirbekov and Daniyal Kadirbekov · GERO · 4 October 2026

AI-assisted numerical investigation and report. Synthetic software example; no real bets or measured customer losses.

## Finding

In penaltyblog's `arbitrage_hedge`, the `hedge_all=False` branch can report a worst-case profit inconsistent with the additional stakes it returns. The checked upstream source is commit `72de6519e0c9a357b8b1aa2a6441a454b18ff54e`, whose project metadata declares version 1.13.0.

The witness starts with 100 units on outcome A at decimal odds 3.0 and zero on B. Current hedge odds are 2.0 on A and 1.9 on B. These are two mutually exclusive, exhaustive hypothetical outcomes, not an ordinary three-way football market with an omitted draw.

| Quantity | Result |
|---|---:|
| Existing total stake | 100 |
| Returned additional stake on A | 100 |
| Returned additional stake on B | 0 |
| Reported `guaranteed_profit` | 0 |
| Actual net if A wins | +300 |
| Actual net if B wins | **-200** |

The cash ledger is simple. The total amount paid is 200. If A wins, the old position pays 300 and the new position pays 200: `300 + 200 - 200 = 300`. If B wins, neither position pays: `0 - 200 = -200`. The original worst case was -100.

## The invariant that fails

For back stakes, the independent reference for each winning outcome is:

```text
net[i] = existing_stake[i] × existing_odds[i]
       + additional_stake[i] × current_odds[i]
       - sum(existing_stakes) - sum(additional_stakes)
guaranteed_profit = min(net)
```

This contract is documented by the upstream result class and is also implemented by its `_calculate_final_profit` helper. The partial branch does not apply that helper when there is no negative stake to redistribute.

## Cause and possible correction

`_calculate_partial_hedges` derives its stake from a payoff whose sign is opposite to the returned positive back stake. Its comments describe a loss when the backed outcome wins and a gain otherwise. Moreover, its intermediate liability uses `h × odds`, which is not the standard `h × (odds - 1)` liability of a decimal-odds lay stake. Calling the output a lay bet would therefore not resolve the issue.

The minimum defensive change is to recompute the reported minimum from the actual returned positions. That would expose the -200 result; it would not make these stakes a good hedge. A complete strategy change requires the maintainer to define whether partial mode restricts additional back stakes to already held outcomes, selects exposures but permits opposite positions, or supports explicitly typed lay positions. No complete tested patch or maintainer acceptance is claimed here.

## An unrestricted comparison, not a replacement definition

If stakes on B are permitted, the synthetic equalizing amount is `3000/19`, approximately 157.894737. The net is `800/19`, approximately 42.105263, on either outcome. This is a comparison with unrestricted hedging, not a claim that partial mode must allow a previously unstaked outcome.

Using exactly 157.89 instead gives 42.11 if A wins and 42.101 if B wins, before any settlement rounding. The displayed two-decimal shorthand must not be mistaken for exact equality. Commissions, minimum stakes, market limits and settlement rules are absent from this model.

## Evidence and reproducibility

The archive contains `verify.py`, the unchanged upstream module with its MIT licence, pinned metadata and a machine-readable replay. The function is loaded directly with `importlib` to avoid unrelated package initialization. This is not a whole-package installation test. The full-hedge control sets only HiGHS' `threads=1` resource option; the source file and optimization objective are unchanged.

The replay checked six partial-mode scenarios: five nonzero-exposure scenarios violate the payoff invariant; the zero-exposure control agrees. One additional full-hedge control succeeds and reproduces the approximately 42.105263 unrestricted result. These deliberately selected cases are not an estimate of failure prevalence. The run used one shared CPU worker, no GPU, and about 0.154 seconds of process CPU time.

`SOURCE_REVIEW.json` records the pinned file hashes, the public issue/PR search and source history. On 4 October, the 49 public issue/PR titles and bodies had no matches for the bounded arbitrage/partial-hedge search. This does not establish worldwide priority or exclude private reports; comments were not exhaustively reviewed.

## Sources and status

- [Pinned implementation](https://github.com/martineastwood/penaltyblog/blob/72de6519e0c9a357b8b1aa2a6441a454b18ff54e/penaltyblog/betting/arbitrage.py)
- [Official hedging documentation](https://penaltyblog.readthedocs.io/en/latest/betting/arbitrage_hedging.html)
- [Maintainer report, issue #50](https://github.com/martineastwood/penaltyblog/issues/50) — submitted 4 October 2026; review pending.

Maintainer contact and publication URLs are recorded separately in the publication receipt. A submitted report is not an acknowledgment or an accepted correction. This audit identifies one defect in a specific calculation path, not a verdict on the whole library or betting industry.

Original report: CC BY 4.0. Original replay code: MIT. Vendored upstream code retains Martin Eastwood's licence. No betting recommendation, production loss, external peer review, or reward is claimed.
