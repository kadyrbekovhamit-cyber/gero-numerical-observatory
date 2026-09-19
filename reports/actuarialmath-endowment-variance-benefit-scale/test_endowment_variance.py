"""Independent regression tests for discrete finite-support payouts."""
from fractions import Fraction as F
import unittest
from actuarialmath import LifeTable, Insurance

class EndowmentVariance(unittest.TestCase):
    def life(self,rate=0):
        return LifeTable().set_table(q={40:.25,41:1}).set_interest(i=rate)

    def test_nonunit_benefit(self):
        life=self.life()
        self.assertAlmostEqual(life.endowment_insurance(40,t=1,b=2,endowment=1,moment=life.VARIANCE),3/16)

    def test_zero_death_benefit(self):
        life=self.life()
        self.assertAlmostEqual(life.endowment_insurance(40,t=1,b=0,endowment=1,moment=life.VARIANCE),3/16)

    def test_deterministic_equal_benefits(self):
        life=self.life()
        self.assertAlmostEqual(life.endowment_insurance(40,t=1,b=100,endowment=100,moment=life.VARIANCE),0)

    def test_discounted_two_period_payouts(self):
        life=self.life(.05);v=1/(1+F(1,20));expected=F(1,4)*F(3,4)*(2*v-2*v*v)**2
        self.assertAlmostEqual(life.endowment_insurance(40,t=2,b=2,moment=life.VARIANCE),float(expected))

    def test_quadratic_unit_conversion(self):
        life=self.life();base=life.endowment_insurance(40,t=1,b=2,endowment=1,moment=life.VARIANCE)
        for c in [2,10,100]:
            with self.subTest(scale=c):
                scaled=life.endowment_insurance(40,t=1,b=2*c,endowment=c,moment=life.VARIANCE)
                self.assertAlmostEqual(scaled/(c*c),base)

    def test_shared_unit_moment_helper_unchanged(self):
        self.assertAlmostEqual(Insurance.insurance_variance(A2=7/4,A1=5/4,b=2),3/4)

if __name__=='__main__':unittest.main(verbosity=2)
