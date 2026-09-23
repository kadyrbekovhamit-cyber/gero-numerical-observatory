# bond_pricing immediate-start annuity audit

[Full report](REPORT_EN.md).

The ZIP is the complete reproducibility package; see its README for execution. The proposed fix is not yet accepted upstream.

Maintainer report: https://github.com/jrvarma/bond_pricing/issues/8

720 synthetic scenarios: 288 mismatches before, zero after the proposed correction. Six additional regression checks passed. One implementation defect, not 288 separate bugs. No real customer loss has been measured.
