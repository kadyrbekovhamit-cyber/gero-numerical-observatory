import sys, json
from pathlib import Path
from dataclasses import asdict
from decimal import Decimal
sys.path.insert(0, str(Path(__file__).parent/'source/src'))
from pyloan import Loan
loan=Loan(loan_amount=1000,interest_rate=12,loan_term=2,loan_term_period='M',start_date='2026-01-01',payment_end_of_month=False,payment_amount=100)
rows=[{k:str(v) for k,v in asdict(p).items()} for p in loan.get_payment_schedule()]
# Independent exact-cent recurrence for a 30/360 monthly annuity with specified payment.
balance=Decimal('1000.00')
reference=[]
for i in range(2):
    interest=(balance*Decimal('0.01')).quantize(Decimal('0.01'))
    payment=min(Decimal('100.00'),balance+interest)
    balance-=payment-interest
    reference.append({'payment':str(payment),'interest':str(interest),'balance':str(balance)})
result={'inputs':{'loan_amount':1000,'interest_rate':12,'months':2,'payment_amount':100,'start_date':'2026-01-01','method':'30E/360 ISDA'},'actual':rows,'reference':reference,'claim_scope':'current master source, two synthetic monthly payments'}
(Path(__file__).parent/'evidence/minimal-current.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
