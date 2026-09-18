"""Published GERO demonstration adapter, not part of PyLoan or a banking product."""
from pathlib import Path
from decimal import Decimal
import argparse
import hashlib
import html
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--module-root', type=Path)
parser.add_argument('--reference', action='store_true')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
budget = Decimal('150.00')
if args.reference:
    balance = Decimal('1000.00')
    for _ in range(2):
        interest = (balance * Decimal('0.01')).quantize(Decimal('0.01'))
        payment = min(Decimal('100.00'), balance + interest)
        balance = balance + interest - payment
    date = '2026-03-01'
    source = 'Independent exact Decimal recurrence for the stated full monthly periods'
else:
    sys.path.insert(0, str(args.module_root))
    from pyloan import Loan
    last = Loan(loan_amount=1000, interest_rate=12, loan_term=2,
                loan_term_period='M', start_date='2026-01-01',
                payment_end_of_month=False, payment_amount=100).get_payment_schedule()[-1]
    payment, interest, balance = last.payment_amount, last.interest_amount, last.loan_balance_amount
    date = last.date.date().isoformat()
    source = 'PyLoan schedule consumed by the separate GERO adapter'
money = lambda value: format(Decimal(value).quantize(Decimal('0.01')), 'f')
notice = {'document_type': 'synthetic_payment_notice', 'due_date': date,
          'amount_requested': money(payment), 'interest_component': money(interest),
          'remaining_principal_in_schedule': money(balance), 'currency': 'unspecified currency units',
          'scope': 'Synthetic demonstration only; no real borrower, invoice, collection or bank integration.'}
args.output.mkdir(parents=True, exist_ok=True)
(args.output/'payment-notice.json').write_text(json.dumps(notice, indent=2)+'\n')
notice = json.loads((args.output/'payment-notice.json').read_text())
decision = {'rule': 'amount_requested <= available_budget', 'available_budget': money(budget),
            'result': 'WITHIN_BUDGET' if Decimal(notice['amount_requested']) <= budget else 'EXCEEDS_BUDGET',
            'cash_above_budget': money(max(Decimal(notice['amount_requested'])-budget, Decimal(0))),
            'scope': 'GERO demonstration budget check; not a credit decision or a PyLoan feature.'}
args.output.mkdir(parents=True, exist_ok=True)
for name, obj in [('payment-notice.json', notice), ('budget-decision.json', decision)]:
    (args.output/name).write_text(json.dumps(obj, indent=2)+'\n')
# Deliberately read the serialized document before the final decision check.
saved = json.loads((args.output/'payment-notice.json').read_text())
assert (Decimal(saved['amount_requested']) <= budget) == (decision['result'] == 'WITHIN_BUDGET')
rows = ''.join('<tr><th>'+html.escape(k.replace('_',' '))+'</th><td>'+html.escape(str(v))+'</td></tr>'
               for k,v in {**notice, **{('budget_'+k):v for k,v in decision.items()}}.items())
(args.output/'payment-notice.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1"><title>Synthetic payment notice</title>'
    '<style>body{font:18px system-ui;margin:40px;max-width:900px}th,td{text-align:left;vertical-align:top;'
    'padding:12px;border-bottom:1px solid #ddd}th{width:36%}h1{font-size:30px}</style>'
    '<h1>Synthetic payment notice — demonstration only</h1><p>'+html.escape(source)+'</p><table>'+rows+'</table></html>')
print(json.dumps({'notice': notice, 'decision': decision}, sort_keys=True))
