import math
import unittest
import mpmath as mp
from lab.black_scholes_greeks import greeks, _scaled_ratio
from lab.greeks_reference import checked, NAMES, LOG_NAMES


class GreekTests(unittest.TestCase):
    def check_fixture(self,args):
        got=greeks(*args)
        ref,check=checked(args)
        self.assertTrue(check['passed'])
        with mp.workdps(180):
            for name in NAMES:
                value=getattr(got,name)
                rounded=float(ref[name])
                if math.isinf(rounded):
                    self.assertEqual(value,rounded)
                else:
                    tolerance=max(mp.mpf('2e-11')*abs(ref[name]),mp.mpf(math.ulp(rounded)))
                    self.assertLessEqual(abs(mp.mpf(value)-ref[name]),tolerance,(args,name))
            for name in LOG_NAMES:
                tol=max(mp.mpf('2e-11'),32*mp.mpf(math.ulp(float(ref[name]))))
                self.assertLessEqual(abs(mp.mpf(getattr(got,name))-ref[name]),tol,(args,name))
        return got

    def test_atm_model_derivatives_not_restricted_leaf(self):
        got=self.check_fixture((100.,100.,1.,0.,0.,.3))
        self.assertAlmostEqual(got.call_delta,.5596176923702425,places=14)
        self.assertAlmostEqual(got.gamma,.013149311030262963,places=16)
        self.assertGreater(abs(got.call_delta-math.erf(.3/(2*math.sqrt(2)))),.4)

    def test_product_range_without_early_underflow(self):
        for args in ((1e300,1e300,1.,0.,0.,math.ulp(0.)),
                     (1e-300,1e-300,1.,0.,0.,.2),
                     (1e300,1e300,1.,0.,0.,.2),
                     (math.exp(-40.5),1.,1.,0.,0.,1.),
                     (1.,math.exp(40.5),1.,-600.,-600.,1.),
                     (1.,1.,1.,-710.,-710.,.2)):
            self.check_fixture(args)
        self.assertGreater(_scaled_ratio(-800.,(1e300,)),0.)
        self.assertGreater(_scaled_ratio(0.,(1e-300,1e-300),(1e-300,)),0.)

    def test_scaling_and_call_put_delta_identity(self):
        args=(107.,100.,1.3,.035,.012,.27)
        a=self.check_fixture(args)
        for exponent in (-500,-100,100,500):
            scale=2.**exponent
            b=self.check_fixture((args[0]*scale,args[1]*scale,*args[2:]))
            self.assertAlmostEqual(a.call_delta,b.call_delta,delta=1e-14)
            self.assertAlmostEqual(a.gamma,b.gamma*scale,delta=1e-14)
            self.assertAlmostEqual(a.vega,b.vega/scale,delta=1e-11)
        self.assertAlmostEqual(a.call_delta-a.put_delta,math.exp(-args[4]*args[2]),delta=3e-16)
        self.assertAlmostEqual(a.vega/a.gamma,args[0]**2*args[-1]*args[2],delta=1e-11)

    def test_model_differentiation_crosscheck(self):
        # mpmath differentiation of the unspecialized price expression checks
        # the closed Greeks, rather than differentiating the numerical branches.
        for args in ((100.,100.,1.,0.,0.,.3),(70.,100.,2.,.04,.01,.2),
                     (150.,100.,.3,-.01,.02,.4)):
            got=self.check_fixture(args)
            with mp.workdps(90):
                s,k,t,r,q,sig=map(mp.mpf,args)
                def price(spot,vol):
                    v=vol*mp.sqrt(t)
                    d1=(mp.log(spot/k)+(r-q)*t)/v+v/2
                    return spot*mp.exp(-q*t)*mp.erfc(-d1/mp.sqrt(2))/2-k*mp.exp(-r*t)*mp.erfc(-(d1-v)/mp.sqrt(2))/2
                dc=mp.diff(lambda x:price(x,sig),s)
                ga=mp.diff(lambda x:price(x,sig),s,2)
                ve=mp.diff(lambda x:price(s,x),sig)
                for value,ref in ((got.call_delta,dc),(got.gamma,ga),(got.vega,ve)):
                    self.assertLess(abs(mp.mpf(value)/ref-1),mp.mpf('2e-13'))

    def test_tail_switch_and_inverse_vega(self):
        for d1 in (-40.,-12.,math.nextafter(-12.,0.),-2.,0.,2.,12.,40.):
            args=(math.exp(d1-.5),1.,1.,0.,0.,1.)
            a=self.check_fixture(args)
            if math.isfinite(a.inverse_vega) and a.vega>0:
                self.assertAlmostEqual(a.inverse_vega*a.vega,1.,delta=2e-14)

    def test_boundary_contract(self):
        for args in ((100.,100.,0.,0.,0.,.3),(100.,100.,1.,0.,0.,0.)):
            with self.assertRaises(ValueError):greeks(*args)
        with self.assertRaises(ArithmeticError):greeks(1.,1.,1e-100,0.,0.,1e-300)


if __name__=='__main__':unittest.main()
