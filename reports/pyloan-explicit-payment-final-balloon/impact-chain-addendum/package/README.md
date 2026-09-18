# PyLoan: measured downstream-chain addendum (v1.1.0)

This is an extension of the existing explicit-payment report, not a new independent defect. Original DOI: https://doi.org/10.5281/zenodo.22834912 . Original evidence SHA-256: 6be9658bd28e6bf72d9dac8add77319a703f4d6c7fedcacefee069604dfcc0a9. Those original files remain unchanged.

Run with Python 3 and python-dateutil 2.9.0.post0:

```sh
python3 -m pip install -r requirements.txt
python3 -B reproduce.py
```

The script checks the included official wheel and pinned source hashes, then runs release/current/candidate/restored plus an independent Decimal recurrence in separate processes. It creates replay-output/ with synthetic notices and decisions. No account, network, bank, payment or real customer is accessed by the replay. It never plays sound or uses a GPU.

The GERO adapter serializes a payment notice, reads its amount, and checks request <= budget150. Original code requests919.10 instead of100 and changes WITHIN_BUDGET to EXCEEDS_BUDGET. Extra cash requested is819.10; cash above this illustrative budget is769.10. Candidate restores both outputs, and the restored original reproduces the differences. Values are unspecified currency units, not measured losses, fees or actual collections.

The 864-case original audit is separate and remains available in its frozen archive. This addendum executes one stated downstream scenario in five states. Its illustrative document/budget adapter was authored by GERO and is not a PyLoan feature or a deployed banking workflow. The original rounding and upstream-test limitations remain. Research, adapter and report preparation were AI-assisted. Existing vendor issue: https://github.com/darius-lesch/pyloan/issues/70 .

MIT license for this package, with original upstream notices retained in source/LICENSE and the included wheel. Current source:73a052cb40f9c23963fbc1ec4370aa7ac9c3efc1. All package files except SHA256SUMS are listed with SHA-256 hashes. expected/ preserves observed outputs; repeat runs write replay-output/.
