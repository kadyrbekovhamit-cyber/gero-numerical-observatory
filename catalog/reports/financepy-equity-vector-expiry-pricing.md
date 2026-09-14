> Archival mirror. [Original report](https://huggingface.co/datasets/XamitK/gero-research-evidence-2026-09/blob/main/financepy-equity-vector-expiry-pricing.md). Claims, dates, authorship and licenses remain those of the original publication; this catalog update does not rerun or revalidate its numerical experiments.

# FinancePy vector pricing uses the last expiry for every option

Independent numerical audit by Xamit Kadirbekov / GERO Research, 13 September 2026.

`EquityVanillaOption` accepts a list of expiry dates, and the project ships a
notebook dedicated to this vectorized interface. In FinancePy **1.1.2**, the
list branch calculates every time to expiry but appends only once, after the
loop. The resulting one-element array contains only the final horizon and is
broadcast across all discount factors.

The installed 1.1.2 source and upstream commit
`2b9227fea9d832c4033421d6cd53a54316414fca` are byte-identical: SHA-256
`3a90ed5998e75713f1a8062a3afd1f24b971d8ff093dbc6e8d46ce3ee12fd9dc`.

For synthetic at-the-money European calls with valuation date 1 January 2015,
expiries 1 July 2015, 1 January 2016 and 1 January 2017, spot and strike 100,
continuous rates 5% and 1%, and volatility 30%, the released wheel returns:

```text
vector: [17.55845313, 18.32964795, 19.89308021]
scalar: [ 9.30205599, 13.61641464, 19.89308021]
delta:  [ 8.25639714,  4.71323331,  0.00000000]
```

The scalar values agree within `3e-6` with an independent implementation of
the closed-form Black--Scholes equation using only Python's `math.erf`. The
last vector element agrees because the accidentally retained horizon belongs
to that expiry.

## Reproduce the released result

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 .venv/bin/python reproduce.py --expect-mismatches 2
```

The input is synthetic. This report establishes a library-level pricing
discrepancy; it does not establish use by a financial institution, a customer
loss or a production deployment.

## Correction and validation

The minimal correction moves `t_exp.append(t)` inside both list loops in
`intrinsic()` and `value()`. The intrinsic cash value happens to cancel the
incorrect horizon algebraically under the tested flat curves, but the stored
`t_exp` state is still wrong; the option value depends on the volatility
horizon and does not cancel.

The correction is submitted as
[FinancePy PR #258](https://github.com/domokane/FinancePy/pull/258), commit
`5f614f113acab591bf8539e7ce7e1198a354bae7`.
At the latest check the pull request is open and mergeable; both official
upstream regression and unit-test workflows are successful.

Validation performed locally, sequentially with one numerical thread:

- focused file: 5 tests passed;
- complete unit suite: 959 tests passed, with four pre-existing `RankWarning`
  messages from LSMC tests;
- vector expiries combined with vector strikes and mixed call/put types match
  their scalar valuations exactly;
- restoring the original append position makes the new regression fail.

The earlier [vectorization PR #153](https://github.com/domokane/FinancePy/pull/153)
added list-date support, but its option comparison was an expression without
an `assert` and used three copies of the same expiry. Bounded GitHub searches
for vector expiry, expiry list, different expiries and the affected class found
no separate report of this defect. This is not a worldwide priority guarantee.

## Sources

- [FinancePy 1.1.2 on PyPI](https://pypi.org/project/financepy/1.1.2/)
- [Released implementation](https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_vanilla_option.py)
- [Vectorization PR #153](https://github.com/domokane/FinancePy/pull/153)
- [Correction PR #258](https://github.com/domokane/FinancePy/pull/258)

The reproducer and report are supplied under GPL-3.0-or-later; see `LICENSE`.

