import math
import unittest
import mpmath as mp
from lab.black_scholes_v3 import prices, _small_v_series, _tail_series
from lab.reference import checked
from lab.metrics import round_binary64


class V3Tests(unittest.TestCase):
    def check(self,args):
        got = prices(*args)
        ref, convergence = checked(*args)
        self.assertTrue(convergence['passed'])
        self.assertEqual(got.otm_option, ref.option)
        with mp.workdps(180):
            limit = max(mp.mpf('1e-10')*ref.otm,mp.mpf(math.ulp(round_binary64(ref.otm))))
            self.assertLessEqual(abs(mp.mpf(got.otm_value)-ref.otm),limit)
            self.assertLessEqual(abs(mp.mpf(got.log_otm_value)-ref.log_otm),
                                max(mp.mpf('2e-11'),16*mp.mpf(math.ulp(float(ref.log_otm)))))
        return got

    def test_positive_integral_series_components(self):
        # Independent high-precision integration, without the recurrences.
        with mp.workdps(70):
            for z in (-1.,-.4,0.,.0625):
                for v in (1e-14,.001,.125):
                    got, _, _ = _small_v_series(z,v)
                    vv,zz = mp.mpf(v),mp.mpf(z)
                    ref = mp.quad(lambda u:mp.exp(zz*u-u*u/2)*(-mp.expm1(-vv*u))/vv,[0,1,4,mp.inf])
                    self.assertLess(abs(mp.mpf(got)/ref-1),mp.mpf('5e-15'))
            for a in (12.,20.,50.):
                for v in (1e-14,.1,5.):
                    got,_,_ = _tail_series(a,v)
                    aa,vv = mp.mpf(a),mp.mpf(v)
                    ref = mp.quad(lambda w:mp.exp(-w-w*w/(2*aa*aa))*(-mp.expm1(-vv*w/aa))/(vv/aa),[0,1,4,mp.inf])
                    self.assertLess(abs(mp.mpf(got)/ref-1),mp.mpf('5e-15'))

    def test_analytic_regions_and_fallback(self):
        fixtures = (
            ((100.,100.,1.,0.,0.,.3),'exact_atm'),
            ((1.00001,1.,1.,0.,0.,.01),'small_v_moments'),
            ((math.exp(-15.1),1.,1.,0.,0.,1.),'tail_difference_series'),
            ((100.,110.,1.,0.,0.,1.),'conditioned_direct'),
            ((math.exp(-1.),1.,1.,0.,0.,.2),'v2_fallback'))
        for args,method in fixtures:
            self.assertEqual(self.check(args).method,method)

    def test_switching_neighborhoods(self):
        # Inputs are perturbations on both sides, not exact internal switches:
        # log(exp(m)) need not recover m exactly.
        for z,v in ((-1.,.1),(-12.,.2),(-.5,.125)):
            for direction in (0.,math.inf):
                for varying in ('z','v'):
                    zz = math.nextafter(z,-math.inf if direction==0 else math.inf) if varying=='z' else z
                    vv = math.nextafter(v,direction) if varying=='v' else v
                    m = -vv*(vv/2-zz)
                    self.check((math.exp(m),1.,1.,0.,0.,vv))

    def test_atm_subnormal_volatility_survives_large_cash_scale(self):
        v = math.ulp(0.)
        got = prices(1e300,1e300,1.,0.,0.,v)
        with mp.workdps(100):
            ref = mp.mpf(1e300)*mp.erf(mp.mpf(v)/(2*mp.sqrt(2)))
            self.assertLess(abs(mp.mpf(got.call)/ref-1),mp.mpf('1e-14'))
        self.assertGreater(got.call,0.)
        self.assertEqual(got.call,got.put)

    def test_scaling_parity_symmetry_and_monotonicity(self):
        for s in (50.,99.,101.,200.):
            calls=[]
            for vol in (.001,.1,.125,.3,1.,3.):
                args = (s,100.,1.,.03,.01,vol)
                a = self.check(args)
                b = prices(100.,s,1.,.01,.03,vol)
                self.assertAlmostEqual(a.call,b.put,delta=4e-12*max(a.call,b.put,math.ulp(0.)))
                self.assertAlmostEqual(a.call-a.put,s*math.exp(-.01)-100*math.exp(-.03),delta=1e-11)
                c = prices(s*2**300,100.*2**300,1.,.03,.01,vol)
                self.assertAlmostEqual(c.call/2**300,a.call,delta=2e-12*max(a.call,math.ulp(0.)))
                calls.append(a.call)
            self.assertEqual(calls,sorted(calls))

    def test_input_limits_are_explicit(self):
        self.assertEqual(prices(101.,100.,0.,.1,.02,.3).call,1.)
        with self.assertRaisesRegex(ArithmeticError,'positive total volatility'):
            prices(1e200,1e200,1e-100,0.,0.,1e-300)
        with self.assertRaises(ValueError):
            prices(0.,1.,1.,0.,0.,.2)


if __name__ == '__main__':
    unittest.main()
