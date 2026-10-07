"""Execute current upstream methods, unchanged AST, with small framework doubles.
This is NOT a full Frappe site, posting or production demand test.
"""
import ast, math, types, datetime, json
from pathlib import Path
P=Path(__file__).resolve().parent
base=P/'evidence/source/lending/loan_management/doctype/loan_repayment_schedule'
ns={'math':math,'frappe':types.SimpleNamespace(db=types.SimpleNamespace(get_default=lambda _:2)),
    'cint':lambda x:int(x or 0),'flt':lambda x,p=None:round(float(x or 0),p) if p is not None else float(x or 0),
    'date_diff':lambda a,b:(a-b).days}
def extract(file,names):
 tree=ast.parse(file.read_text())
 for node in ast.walk(tree):
  if isinstance(node,ast.FunctionDef) and node.name in names:
   exec(compile(ast.Module(body=[node],type_ignores=[]),str(file),'exec'),ns)
extract(base/'utils.py',{'get_amounts','get_monthly_repayment_amount','round_emi','get_frequency'})
extract(base/'loan_repayment_schedule.py',{'get_days_and_months','get_non_monthly_days'})
s=types.SimpleNamespace(repayment_frequency='Quarterly',repayment_start_date=datetime.date(2026,4,1),posting_date=datetime.date(2026,1,1))
s.get_non_monthly_days=types.MethodType(ns['get_non_monthly_days'],s)
rows=[]; balance=1_000_000.; emi=ns['get_monthly_repayment_amount'](balance,12,8,'Quarterly')
for y,m in [(2026,4),(2026,7),(2026,10),(2027,1),(2027,4),(2027,7),(2027,10),(2028,1)]:
 date=datetime.date(y,m,1)
 days,den=ns['get_days_and_months'](s,date,0,balance,12,'repayment_schedule',100,100)
 result=ns['get_amounts'](balance,12,days,den,emi)
 rows.append({'date':str(date),'balance_before':balance,'days':days,'denominator':den,'interest':result[0],'principal':result[1],'balance_after':result[3]})
 balance=result[3]
assert rows[0]['days']==90 and all(r['days']==3 for r in rows[1:])
assert ns['get_amounts'](1_000_000,12,3,365,142456)[0]==986.30
print(json.dumps({'scope':'Unchanged extracted methods with framework doubles; site/database not started','quarterly_emi':emi,'rows':rows,'interest_total':sum(r['interest'] for r in rows),'independent_single_period_comparison':{'balance':1000000,'rate_percent':12,'upstream_3_days':986.30,'91_actual_days_365':round(1000000*12*91/36500,2),'nominal_quarter':30000}},indent=2))
