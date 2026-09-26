import unittest
from fractions import Fraction as F
import mpmath as mp
from lab.iv_reference_v3 import value,number


class ReferenceIdentities(unittest.TestCase):
    def test_exact_intrinsic_zero_carry(self):
        for s,k in [(237,119),(101,227),(83,207),(149,117),(87,103),(111,103)]:
            for option in ('call','put'):
                for t in ('0','1','7/4'):
                    c=dict(s=str(s),k=str(k),t=t,r='0',q='0',option=option)
                    for digits in (180,260):
                        with mp.workdps(digits):
                            self.assertEqual(value(c,None,'intrinsic'),max(s-k if option=='call' else k-s,0))

    def test_tiny_intrinsic(self):
        c=dict(s='91',k='91',t='1',r='1e-32',q='0',option='call')
        with mp.workdps(180):a=value(c,None,'intrinsic')
        with mp.workdps(260):
            b=value(c,None,'intrinsic')
            self.assertLess(abs(a-b)/b,mp.mpf('1e-175'))
            self.assertLess(abs(b/mp.mpf('9.1e-31')-1),mp.mpf('1e-31'))

    def test_zero_expiry_unequal_rates(self):
        c=dict(s='111',k='103',t='0',r='1/10',q='3/100',option='call')
        with mp.workdps(180):self.assertEqual(value(c,None,'intrinsic'),8)


if __name__=='__main__':unittest.main()
