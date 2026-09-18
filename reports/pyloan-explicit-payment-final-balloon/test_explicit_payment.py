import unittest
from decimal import Decimal as D
from pyloan import Loan

class ExplicitPaymentRegression(unittest.TestCase):
    def loan(self, **overrides):
        args=dict(loan_amount=1000,interest_rate=12,loan_term=2,loan_term_period='M',start_date='2026-01-01',payment_end_of_month=False,payment_amount=100)
        args.update(overrides)
        return Loan(**args)

    def test_partial_amortization_preserves_final_payment(self):
        schedule=self.loan().get_payment_schedule()
        self.assertEqual([p.payment_amount for p in schedule[1:]], [D('100.00'),D('100.00')])
        self.assertEqual(schedule[-1].loan_balance_amount,D('819.10'))

    def test_summary_reports_positive_residual(self):
        s=self.loan().get_loan_summary()
        self.assertEqual(s.total_payment_amount,D('200.00'))
        self.assertEqual(s.total_principal_amount,D('180.90'))
        self.assertEqual(s.residual_loan_balance,D('819.10'))

    def test_automatic_annuity_still_closes_balance(self):
        s=self.loan(payment_amount=None).get_payment_schedule()
        self.assertEqual(s[-1].loan_balance_amount,D('0.00'))

    def test_early_repayment_caps_payment_to_debt(self):
        s=self.loan(payment_amount=2000).get_payment_schedule()
        self.assertEqual(len(s),2)
        self.assertEqual(s[-1].payment_amount,D('1010.00'))
        self.assertEqual(s[-1].loan_balance_amount,D('0.00'))

    def test_interest_only_retains_principal(self):
        s=self.loan(loan_type='interest-only',payment_amount=None).get_payment_schedule()
        self.assertEqual(s[-1].loan_balance_amount,D('1000.00'))
        self.assertEqual(s[-1].payment_amount,D('10.00'))

if __name__=='__main__': unittest.main()
