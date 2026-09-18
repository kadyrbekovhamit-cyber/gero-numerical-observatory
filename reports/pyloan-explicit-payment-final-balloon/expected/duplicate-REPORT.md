# PyLoan explicit-payment regression review

Updated 18 September 2026, 21:34 Asia/Tashkent (16:34 UTC).
Read-only source/history review. **No numerical code executed and no external message sent.**

## Assessment

This is a strong **explicit-payment contract regression** candidate, rather than a newly discovered general final-balloon policy. PR68 intentionally introduced that policy for residuals in automatically calculated annuity loans. Its implementation also applies to loans whose periodic payment is supplied explicitly; current documentation still permits these loans to retain a balance at the end of the requested term. No explicit documented decision withdrawing that capability was found.

[Current source](https://github.com/darius-lesch/pyloan/blob/73a052cb40f9c23963fbc1ec4370aa7ac9c3efc1/src/pyloan/pyloan.py#L515) at master `73a052cb40f9c23963fbc1ec4370aa7ac9c3efc1` unconditionally sets final principal to the opening balance for every non-interest-only loan. No `payment_amount is None` guard is present there.

Current develop `e94537c5679e6c22757a1ce2b1f73c35a8b06249` and tag `v0.7.2` share the same `pyloan.py` blob `d742b696b6cba5892229186fe164756cf469b8c0`. Develop does not already fix it.

## Contract

The [pinned quickstart](https://github.com/darius-lesch/pyloan/blob/73a052cb40f9c23963fbc1ec4370aa7ac9c3efc1/docs/docsrc/source/quickstart.rst#L70), [latest docs](https://pyloan.readthedocs.io/en/latest/quickstart.html#specify-payment-amount), and stable docs distinguish automatic full amortization from explicit periodic payments, which can leave a residual. The example supplies160000 principal,1.1% annual interest,10years and888.33 monthly payments. Docs blob `3292105a3ff5cb8256bd2474e6c84053f38ef858` matches master and develop.

[Issue15](https://github.com/darius-lesch/pyloan/issues/15) historically defines automatic full amortization specifically for `payment_amount=None`. It has no comments.

## Prior issue and fixes: bodies, diffs and comments read

- [Issue67](https://github.com/darius-lesch/pyloan/issues/67) reports unwanted residual for an **automatically sized**40-year annuity under Actual/365. Its single [maintainer comment](https://github.com/darius-lesch/pyloan/issues/67#issuecomment-3560543121) recommends0.7.2 and a specific alternative fixed-payment configuration. It does not explicitly discuss intentionally under-amortizing inputs or announce their removal.
- [PR68](https://github.com/darius-lesch/pyloan/pull/68/files), merged20Nov2025, adds the unconditional final-principal override and changes version0.7.0→0.7.1. Body explicitly describes final adjustment as its chosen solution. Therefore the existence of a final adjustment is **known and intentional**, and must be credited.
- [PR69](https://github.com/darius-lesch/pyloan/pull/69/files), merged later that day, adds the automatic-payment solver and changes version0.7.1→0.7.2. It keeps the final override, changing only surrounding comments. Its compatibility statement that explicit payments are unaffected is relative to its0.7.1 base; **do not use that sentence alone as proof of pre0.7.1 compatibility**.
- All issue comments and inline review comments for PR68/69 returned empty. Both full returned diffs were inspected.
- v0.7.0 source blob `72708e8c203e1d554d706d08c0743171a0833ea5` lacks the final override. v0.7.1 source blob `fe309ccd3e8a8914e5d387bc7f4a65da0d69394e` contains it.

## Versions

GitHub latest release: [v0.7.2](https://github.com/darius-lesch/pyloan/releases/tag/v0.7.2),20Nov2025. Tag points to develop commit `e94537c5679e6c22757a1ce2b1f73c35a8b06249`; master is its merge commit. The official [PyPI project page](https://pypi.org/project/pyloan/) also identifies0.7.2:

- sdist SHA256 `53950ac525ea8e82909bebff98711e6e68c675d38f5540e53c386395406c83e8`.
- wheel SHA256 `a0a24cb9b545d05d49e61186d7dded865511f95be20703031de79b7b0f0d16cb`.

This agent did **not** download or execute the artifacts; their source equivalence remains for root's artifact verification. The web tool's PyPI page had a cached last-month crawl; direct JSON retrieval returned a cache miss.

## Duplicate boundary

Six fresh all-state issue/PR searches across titles, bodies and comments: payment_amount, balloon, residual, final payment, last payment, specified. Only15,67,68,69 matched; each `incomplete_results=false`. Fresh repository inventory contains69 closed issues/PRs, none open and none newer than69. No exact prior report or proposed explicit-payment guard was found. GERO catalog filenames contain no pyloan match; this is a filename check, not a complete full-content catalog guarantee.

## Evidence root should establish

Use an explicit **annuity** payment greater than periodic interest but intentionally too small to clear the loan. Without special payments, the independent recurrence is balance-next = balance + rounded-interest − supplied-payment; the documented term can end with positive balance. This isolates the contract from linear-loan principal/gross-payment semantics.

Compare official0.7.2/current with0.7.0 and0.7.1. Prefer a rate unaffected by PR69's rate-precision change and full monthly30E/360ISDA periods. Check that a candidate guard preserves automatic amortization, early repayment and interest-only behavior. A valid final payment may be **smaller** than the supplied amount when the loan is repaid; do not assert equality indiscriminately.

Official report route: [bug template](https://github.com/darius-lesch/pyloan/issues/new?template=bug_report.md). Template requires reproducer, expected result, OS/Python/PyLoan versions; no AI attestation was found in it or CONTRIBUTING. No report was sent.

Full raw API evidence, queries and hashes: `SOURCES.json`. Runtime confirmation and final publication decisions belong to root. No bank deployment or borrower-loss claim is supported.
