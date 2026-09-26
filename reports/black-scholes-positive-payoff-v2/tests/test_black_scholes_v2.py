import math
import unittest

import mpmath as mp

from lab.black_scholes_v2 import RULE16, RULE32, log_ratio, prices, scaled_product
from lab.metrics import round_binary64
from lab.reference import checked


class V2Tests(unittest.TestCase):
    def assert_pricing_close(self,args):
        result = prices(*args)
        ref,convergence = checked(*args)
        self.assertTrue(convergence['passed'])
        self.assertEqual(result.otm_option,ref.option)
        with mp.workdps(180):
            allowance = max(mp.mpf('1e-11')*ref.otm,mp.mpf(math.ulp(round_binary64(ref.otm))))
            self.assertLessEqual(abs(mp.mpf(result.otm_value)-ref.otm),allowance)
        return result

    def test_legendre_rules_integrate_polynomial_moments(self):
        for rule in (RULE16,RULE32):
            for degree in range(2*len(rule)):
                got = math.fsum(w*x**degree for x,w in rule)
                expected = 0.0 if degree%2 else 2.0/(degree+1)
                self.assertAlmostEqual(got,expected,delta=6e-15*max(expected,1e-2))

    def test_near_atm_moneyness_survives_exact_power_two_rescaling(self):
        ratio = math.nextafter(1.0,math.inf)
        expected = math.log1p(ratio-1.0)
        for exponent in (-900,-400,0,400,900):
            self.assertEqual(log_ratio(math.ldexp(ratio,exponent),math.ldexp(1.,exponent)),expected)
        self.assertLess(log_ratio(math.ulp(0.),2*math.ulp(0.)),0)

    def test_exponential_reconstruction_preserves_monetary_range(self):
        for base,exponent in ((1e200,-1000.),(1e-200,500.),(1.,-744.)):
            got = scaled_product(base,exponent)
            with mp.workdps(180):
                expected = mp.mpf(base)*mp.exp(mp.mpf(exponent))
                allowance = max(mp.mpf('2e-14')*expected,mp.mpf(math.ulp(round_binary64(expected))))
                self.assertLessEqual(abs(mp.mpf(got)-expected),allowance)

    def test_v1_failure_regressions(self):
        fixtures = (
            ('0x1.00000003237e2p+0','0x1.0000000000000p+0','0x1.54e9621a41fc9p-33'),
            ('0x1.c05c0a7166b4ap+432','0x1.0000000000000p+0','0x1.d7f9822a4c63fp+2'),
            ('0x1.87e92154ef7aep-665','0x1.87e92154ef7acp-665','0x1.5798ee2308c3ap-27'),
            ('0x1.753025a61bfe5p+55','0x1.0000000000000p+0','0x1.0000000000000p+0'))
        for s,k,v in fixtures:
            got=self.assert_pricing_close((float.fromhex(s),float.fromhex(k),1.,0.,0.,float.fromhex(v)))
            self.assertGreater(got.otm_value,0.)

    def test_atm_erf_identity_across_scales(self):
        for scale in (1e-200,1.,1e200):
            for vol in (1e-12,0.2,3.,20.):
                got = prices(scale,scale,1.,0.,0.,vol)
                with mp.workdps(180):
                    expected = mp.mpf(scale)*mp.erf(mp.mpf(vol)/(2*mp.sqrt(2)))
                    self.assertLess(abs(mp.mpf(got.call)-expected)/expected,mp.mpf('3e-14'))
                self.assertEqual(got.call,got.put)

    def test_volatility_monotonicity_and_bounds(self):
        for s in (50.,99.,100.,101.,200.):
            values = [prices(s,100.,1.,0.03,0.01,v) for v in (1e-4,.01,.2,1.,8.)]
            self.assertEqual([p.call for p in values],sorted(p.call for p in values))
            for value in values:
                stock,strike = s*math.exp(-.01),100.*math.exp(-.03)
                self.assertGreaterEqual(value.call,0.)
                self.assertLessEqual(value.call,stock+8*math.ulp(stock))
                self.assertLessEqual(value.put,strike+8*math.ulp(strike))
                self.assertAlmostEqual(value.call-value.put,stock-strike,delta=3e-13*max(stock,strike))

    def test_reciprocal_call_put_symmetry(self):
        for s,k,t,r,q,v in ((100.,105.,.5,.04,.01,.3),(1e-150,2e-150,2.,-.01,.03,.7),(1.01,1.,1.,0.,0.,.1)):
            a,b=prices(s,k,t,r,q,v),prices(k,s,t,q,r,v)
            self.assertAlmostEqual(a.call,b.put,delta=4e-12*max(a.call,b.put,math.ulp(0.)))

    def test_deterministic_limits(self):
        value = prices(101.,100.,0.,.1,.02,.3)
        self.assertEqual((value.call,value.put,value.parity),(1.,0.,1.))
        value = prices(99.,100.,1.,0.,0.,0.)
        self.assertEqual((value.call,value.put,value.parity),(0.,1.,-1.))
        self.assertEqual(value.log_otm_value,-math.inf)

    def test_invalid_input_is_explicit(self):
        for args in ((0.,1.,1.,0.,0.,.2),(1.,1.,-1.,0.,0.,.2),(1.,1.,1.,0.,0.,-.2),(math.nan,1.,1.,0.,0.,.2)):
            with self.assertRaises(ValueError):
                prices(*args)

    def test_positive_volatility_underflow_is_not_a_deterministic_contract(self):
        # True ATM time value is about 3.99e-151 after the large cash scale,
        # even though binary64 sigma*sqrt(T) rounds to zero. Fail explicitly.
        with self.assertRaisesRegex(ArithmeticError,'positive total volatility'):
            prices(1e200,1e200,1e-100,0.,0.,1e-300)


if __name__ == '__main__':
    unittest.main()
