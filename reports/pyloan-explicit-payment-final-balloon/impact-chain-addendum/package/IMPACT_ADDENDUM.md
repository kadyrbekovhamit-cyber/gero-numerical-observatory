# PyLoan downstream-chain evidence addendum

## Measured chain: calculation → document → decision

**New evidence addendum, 18 September 2026.** This extends the same reported defect; it is not another independent bug.

We executed the official PyLoan 0.7.2 wheel, pinned current source, local candidate and restored original through a small published GERO adapter. The adapter writes a synthetic payment notice from the final schedule row, reads that serialized document, then checks whether its requested amount fits an example budget of **150.00 currency units**. The notice and budget rule are GERO demonstration code, not built-in PyLoan features or a tested bank integration.

| Measured output | Independent reference / candidate | Official release / current / restored |
|---|---:|---:|
| Requested amount in generated notice | 100.00 | **919.10** |
| Remaining principal shown by schedule | 819.10 | **0.00** |
| Budget check: request ≤ 150.00 | WITHIN_BUDGET | **EXCEEDS_BUDGET** |
| Requested cash above that budget | 0.00 | **769.10** |

The chain is: **the final-payment override → a generated notice requesting 819.10 more than specified → a reversed budget-check result**. The requested amount is **9.191 times** the configured payment. The candidate restores the document fields and decision to the independent reference; restoring the original branch reproduces both differences. The archive contains executable code, all five sets of generated JSON/HTML documents, and checksums.

These are measured consequences inside an explicitly synthetic workflow. The additional request represents accelerated principal repayment in this schedule, not a measured fee, extra interest charge or borrower loss. No payment was collected, no real person was assessed, and no production use or frequency was measured. The example budget is a stated demonstration input, not a credit-eligibility rule. The original 864-case results and remaining rounding/test limitations below remain unchanged.

