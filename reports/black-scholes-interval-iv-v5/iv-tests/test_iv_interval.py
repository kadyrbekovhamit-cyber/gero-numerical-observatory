import math,sys,unittest
from fractions import Fraction as F
import mpmath as mp
from flint import ctx
from lab.iv_interval import Contract,QuoteInterval,invert_quote,rounding_cell,rational
from lab.iv_reference import value,check_witness,witnesses

ctx.threads=1


class Intervals(unittest.TestCase):
    def check(self,c,q,**kw):
        result=invert_quote(c,q,**kw)
        for w in witnesses(result):
            check=check_witness(c.as_dict(),w)
            if check['checked']:self.assertTrue(check['pass'],(w,check))
        return result

    def test_exact_inputs(self):
        self.assertEqual(rational('0.1'),F(1,10))
        self.assertNotEqual(rational(0.1),F(1,10))

    def test_even_cell(self):
        q=rounding_cell(1.0)
        self.assertEqual(q.lower,1-F(1,2**54));self.assertEqual(q.upper,1+F(1,2**53))
        self.assertTrue(q.lower_closed and q.upper_closed)

    def test_odd_cell(self):
        q=rounding_cell(math.nextafter(1.0,2.0))
        self.assertFalse(q.lower_closed or q.upper_closed)
        self.assertEqual(float(q.lower),1.0)

    def test_zero_cell(self):
        q=rounding_cell(0.0)
        self.assertEqual((q.lower,q.upper),(F(0),F(1,2**1075)))
        self.assertTrue(q.upper_closed)

    def test_smallest_subnormal_cell(self):
        q=rounding_cell(math.ulp(0.0))
        self.assertEqual((q.lower,q.upper),(F(1,2**1075),F(3,2**1075)))
        self.assertFalse(q.lower_closed or q.upper_closed)

    def test_maximum_cell(self):
        q=rounding_cell(sys.float_info.max)
        self.assertEqual(q.upper,F(2**1024-2**970))
        self.assertFalse(q.upper_closed)

    def test_below_intrinsic(self):
        self.assertEqual(self.check(Contract(120,100,1),QuoteInterval(0,19))['status'],'empty')

    def test_intrinsic_singleton(self):
        z=self.check(Contract(120,100,1),QuoteInterval(20,20))
        self.assertEqual(z['status'],'singleton_zero')

    def test_excluded_intrinsic(self):
        z=self.check(Contract(120,100,1),QuoteInterval(19,20,True,False))
        self.assertEqual(z['status'],'empty')

    def test_unattained_ceiling(self):
        z=self.check(Contract(120,100,1),QuoteInterval(120,121))
        self.assertEqual(z['status'],'empty')

    def test_unbounded_and_open_zero(self):
        z=self.check(Contract(120,100,1),QuoteInterval(20,120,False,False))
        self.assertEqual(z['status'],'interval');self.assertTrue(z['unbounded'])
        self.assertFalse(z['zero_included']);self.assertIsNone(z['outer_upper'])

    def test_expiry_all_and_empty(self):
        for option,payoff in [('call',20),('put',0)]:
            c=Contract(120,100,0,option=option)
            self.assertEqual(self.check(c,QuoteInterval(payoff,payoff))['status'],'all_volatilities')
            self.assertEqual(self.check(c,QuoteInterval(payoff+1,payoff+2))['status'],'empty')

    def test_open_point_empty(self):
        self.assertEqual(self.check(Contract(100,100,1),QuoteInterval(10,10,False,True))['status'],'empty')

    def test_lost_information(self):
        z=self.check(Contract(200,100,1),rounding_cell(100.0),tolerance=F(1,10**15))
        self.assertTrue(z['resolved']);self.assertEqual(z['outer_lower'],'0')
        e=z['upper_endpoint'];self.assertLess(F(e['upper'])-F(e['lower']),F(1,10**15))
        self.assertLess(F(e['lower']),F('0.0894076873552444'))
        self.assertGreater(F(e['upper']),F('0.0894076873552443'))

    def test_regular_interval(self):
        c=Contract(100,100,1)
        z=self.check(c,QuoteInterval(7,9))
        self.assertTrue(z['resolved']);self.assertLess(F(z['outer_lower']),F('0.2'))
        self.assertGreater(F(z['outer_upper']),F('0.2'))

    def test_put_call_parity(self):
        c=Contract(120,100,1)
        call=self.check(c,QuoteInterval(25,26))
        put=self.check(Contract(120,100,1,option='put'),QuoteInterval(5,6))
        self.assertEqual((call['outer_lower'],call['outer_upper']),(put['outer_lower'],put['outer_upper']))

    def test_monetary_scale(self):
        base=self.check(Contract(120,100,1),QuoteInterval(25,26))
        for scale in [F(1,10**200),F(10**200)]:
            z=self.check(Contract(120*scale,100*scale,1),QuoteInterval(25*scale,26*scale))
            self.assertEqual((base['outer_lower'],base['outer_upper']),(z['outer_lower'],z['outer_upper']))

    def test_time_scaling(self):
        a=self.check(Contract(100,100,1),QuoteInterval(7,9),tolerance=F(1,2**40))
        b=self.check(Contract(100,100,4),QuoteInterval(7,9),tolerance=F(1,2**41))
        self.assertEqual(F(a['outer_lower']),2*F(b['outer_lower']))
        self.assertEqual(F(a['outer_upper']),2*F(b['outer_upper']))

    def test_iteration_budget(self):
        z=self.check(Contract(100,100,1),QuoteInterval(7,9),max_steps=1)
        self.assertFalse(z['resolved']);self.assertLessEqual(F(z['outer_lower']),F('.2'))
        self.assertGreaterEqual(F(z['outer_upper']),F('.2'))

    def test_sigma_budget(self):
        z=self.check(Contract(100,100,1),QuoteInterval(7,9),max_sigma=F(1,100))
        self.assertFalse(z['resolved']);self.assertIsNone(z['outer_upper'])
        self.assertEqual(z['upper_endpoint']['state'],'sigma_budget_exhausted')

    def test_uncertain_first_bracket(self):
        c=Contract(100,100,1)
        with mp.workdps(180):p=F(mp.nstr(value(c.as_dict(),'1'),100))
        z=self.check(c,QuoteInterval(p,p),precisions=(64,))
        self.assertFalse(z['resolved']);self.assertIsNone(z['outer_upper'])
        self.assertEqual(z['upper_endpoint']['state'],'unresolved_bracketing')

    def test_uncertain_inside_bracket(self):
        c=Contract(100,100,1)
        with mp.workdps(180):p=F(mp.nstr(value(c.as_dict(),'1/2'),100))
        z=self.check(c,QuoteInterval(p,p),precisions=(64,))
        self.assertFalse(z['resolved']);self.assertIsNotNone(z['outer_upper'])
        self.assertEqual(z['upper_endpoint']['state'],'precision_budget_exhausted')

    def test_boundary_unknown(self):
        c=Contract(100,100,1,r='1e-30')
        z=self.check(c,QuoteInterval('1e-28',1),precisions=(64,))
        self.assertEqual(z['status'],'unresolved')

    def test_invalid_inputs(self):
        for f in [lambda:rounding_cell(-1.0),lambda:rounding_cell(float('inf')),
                  lambda:Contract(0,100,1),lambda:Contract(1,1,-1),lambda:rational(True),
                  lambda:rational('1e999999999'),lambda:QuoteInterval(2,1),
                  lambda:invert_quote(Contract(1,1,1),QuoteInterval(0,1),max_steps=True)]:
            with self.assertRaises((ValueError,TypeError)):f()


if __name__=='__main__':unittest.main()
