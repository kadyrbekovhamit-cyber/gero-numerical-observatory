# PyLoan 0.7.3: accepted fix and independent release verification

19 September 2026 · Xamit Kadirbekov · GERO · update to the same reported defect.

The maintainer [confirmed the report and implementation of the proposed guard](https://github.com/darius-lesch/pyloan/issues/70#issuecomment-5736689028), added a regression test, closed issue 70 and released PyLoan 0.7.3 on 18 September 2026 UTC. [Fix commit](https://github.com/darius-lesch/pyloan/commit/d5229ad8c0fedfd85339db76bf6af941d2ccda10) and [merged PR 71](https://github.com/darius-lesch/pyloan/pull/71). This is a verified upstream correction, not a new independent defect.

We executed the official PyPI 0.7.3 wheel and master `12772f720b432bc1e8d76ca0dda5bad68e2c3450` against the existing independent Decimal reference. All six package modules match between the wheel and pinned master.

| Actual execution | Payment-cap violations / 864 schedules |
|---|---:|
| Official 0.7.2 |408|
| Official 0.7.3 |0|
| Current pinned master |0|
| 0.7.3 with only the new guard removed |408|

All 456 formerly passing schedules and 60 separate automatic-annuity/linear/interest-only controls are unchanged. All 864 updated-release schedules match the earlier local candidate byte for byte on the tested platform. Five focused regression tests pass on 0.7.3/current and two fail on 0.7.2/restored.

The downstream chain was rerun through the same disclosed GERO adapter: 0.7.2 produces a synthetic notice requesting 919.10, which exceeds the example budget 150 by 769.10. Official 0.7.3 produces 100.00 and explicitly retains 819.10 of principal; the budget result becomes WITHIN_BUDGET. Removing the guard restores the original notice and decision. These are accelerated-principal and synthetic decision effects, not measured fees, extra interest, collected payments or borrower losses. The notice/budget logic is GERO demonstration code, not a PyLoan feature or a bank integration.

**Limits retained:** 24 scenarios still differ from the Decimal recurrence by up to 0.02 currency units. Current upstream suite executes 17 test methods with the same three baseline failure reports (two scenario snapshots and one February/March day-count case); removing the new guard adds the new regression failure, for four. The full suite is not green. No deployment frequency, customer-loss, security, performance or legal conclusion is measured.

Python 3.12.14, python-dateutil 2.9.0.post0, macOS arm64, Decimal 28-digit precision and ROUND_HALF_EVEN. One configured CPU worker; no GPU or audio playback. Preparation was AI-assisted. The original version 1.0 and synthetic-chain version 1.1 archives remain immutable. This update supplies official artifacts, pinned source, execution code, restored-guard mutation, complete schedules/documents and logs. Cross-platform byte identity is not promised.
