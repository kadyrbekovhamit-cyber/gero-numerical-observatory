# Frappe Lending quarterly day-count evidence

This directory preserves a reproducible, synthetic check of the quarterly
repayment day-count defect present in Frappe Lending source commit
`605ae2a866aade812a4a147938b8a4f3b7aaa760` and release `v16.6.0`.

Run the extracted-method replay with a standard Python 3 interpreter:

```bash
python3 verify.py
```

The script executes the relevant upstream methods directly from the pinned
source copies under `evidence/source/`, using small framework doubles. It does
not start a Frappe site or database. Its output should show 90 days for the
first row and 3 days for every later quarterly row.

`evidence/full-site-quarterly-check.json` is the separately saved result of a
synthetic Frappe site run. `evidence/full-site-quarterly-check.py` records the
procedure, but the full bench and database are not included in this archive.

The upstream source copies retain Frappe Lending's GPL-3.0 licence in
`evidence/source/license.txt`. The GERO report is CC BY 4.0; the original GERO
replay is released under MIT.

Current correction status is documented in the report and in the saved public
pull-request metadata under `evidence/upstream-fix/`.

## Distribution and release update — 7 October 2026

Official v16.6.1 contains the calendar-day correction. See the dated report update and separate release-verification archive; the original evidence ZIP is unchanged.

- zenodo: https://zenodo.org/records/23218489
- youtube: https://youtube.com/shorts/7CYnCfdLQaM
