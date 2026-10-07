"""Real-site check: frappe/lending quarterly repayment schedule interest.

Run with the bench Python under the shared compute lock:
  one_worker.py -- bench/env/bin/python quarterly_check.py
Synthetic test company/customer only; local MariaDB; nothing leaves the machine.
"""
import json
import os
import sys
from datetime import date
from fractions import Fraction as F

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench")
SITE = "lending.lab"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "QUARTERLY_CHECK.json")

os.chdir(os.path.join(BENCH, "sites"))
import frappe  # noqa: E402

frappe.init(site=SITE, sites_path=".")
frappe.connect()
frappe.flags.in_test = True
frappe.set_user("Administrator")
# Lab site only: ERPNext test bootstrap creates users with fixture passwords.
frappe.db.set_single_value("System Settings", "enable_password_policy", 0)
frappe.db.commit()
frappe.clear_cache()

from lending.tests.test_utils import (  # noqa: E402
	before_tests, create_loan_product, create_loan, make_loan_disbursement_entry, make_customer,
)

before_tests()
create_loan_product("QTR-LAB", "Quarterly Lab Product", 5000000, 12)
make_customer("_Quarterly Lab Customer")
applicant = frappe.db.get_value("Customer", {"customer_name": "_Quarterly Lab Customer"})

P, RATE, N = 1000000, 12, 8
POSTING, START = "2026-01-01", "2026-04-01"
loan = create_loan(applicant, "QTR-LAB", P, "Repay Over Number of Periods", N,
                   repayment_start_date=START, posting_date=POSTING, rate_of_interest=RATE,
                   repayment_frequency="Quarterly")
loan.submit()
# Same entry point as the Loan form's "Disburse" button: frequency comes from the loan.
from lending.loan_management.doctype.loan.loan import make_loan_disbursement
disb = make_loan_disbursement(loan.name, P, submit=True, repayment_start_date=START,
                              posting_date=POSTING, disbursement_date=POSTING)
assert disb.repayment_frequency == "Quarterly", disb.repayment_frequency
frappe.db.commit()

sched = frappe.get_all("Loan Repayment Schedule", {"loan": loan.name, "loan_disbursement": disb.name, "docstatus": 1},
                       ["name", "status"], order_by="creation desc")[0]
rows = frappe.get_all("Repayment Schedule", {"parent": sched.name},
                      ["payment_date", "number_of_days", "principal_amount", "interest_amount",
                       "total_payment", "balance_loan_amount"], order_by="idx")

# Independent oracle: the code's own convention (interest = balance * rate * days / 36500,
# actual days between payment dates), applied to the balances the site produced.
checks, prev = [], date.fromisoformat(POSTING)
balance = F(P)
for r in rows:
	d = r.payment_date
	days = (d - prev).days
	expected = balance * F(RATE, 100) * days / 365
	checks.append({
		"payment_date": str(d), "site_number_of_days": r.number_of_days,
		"actual_days": days, "site_interest": float(r.interest_amount),
		"expected_interest_actual_days": round(float(expected), 2),
		"site_principal": float(r.principal_amount), "site_total": float(r.total_payment),
		"site_balance_after": float(r.balance_loan_amount),
	})
	balance = F(str(r.balance_loan_amount))
	prev = d

result = {
	"date": "2026-10-05", "timezone": "Asia/Tashkent", "site": SITE,
	"apps": {a: frappe.get_attr(a + ".__version__") for a in ("frappe", "erpnext", "lending")},
	"loan": loan.name, "disbursement": disb.name, "disbursement_frequency": disb.repayment_frequency, "schedule": sched.name, "schedule_status": sched.status,
	"inputs": {"principal": P, "annual_rate_pct": RATE, "periods": N, "frequency": "Quarterly",
	           "posting_date": POSTING, "repayment_start_date": START},
	"rows": checks,
	"site_total_interest": round(sum(c["site_interest"] for c in checks), 2),
	"expected_total_interest_on_site_balances": round(sum(c["expected_interest_actual_days"] for c in checks), 2),
}
with open(OUT, "w") as fh:
	json.dump(result, fh, indent=2, ensure_ascii=False)
print(json.dumps(result, indent=2, ensure_ascii=False))
frappe.destroy()
