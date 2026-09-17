# Source ledger

Reviewed17September2026. This archives the existing16September issue; current source is fixed.

- Original source: https://github.com/domokane/FinancePy/tree/2b9227fea9d832c4033421d6cd53a54316414fca
- Current source: https://github.com/domokane/FinancePy/tree/4c7cd50bdadd6efc9ac74fa81e93374397d18e3e
- Upstream units correction: https://github.com/domokane/FinancePy/commit/bb10c3936e078a5212694d017746453175151c09
- Existing report and verification: https://github.com/domokane/FinancePy/issues/267
- Official released version: https://pypi.org/project/financepy/1.1.2/

SOURCE.json records archive hashes; both complete archive inventories are checked against Git blob identities before execution. The wheel RECORD is verified independently. No external network access is required by reproduce.py.

The independent oracle prices one terminal cash flow with80/120-digit arithmetic and Python calendar days. Its finite-difference convention is kept separate from the exact derivative. Frozen input space and tolerances are in recorded-evidence/protocol.json.
