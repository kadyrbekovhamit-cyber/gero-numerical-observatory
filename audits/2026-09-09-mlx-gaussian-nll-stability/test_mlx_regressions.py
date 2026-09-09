"""Run actual-function regression assertions on original or candidate code."""
import argparse
from decimal import Decimal, localcontext
import json
import math
import os
from pathlib import Path
import types
import unittest

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import mlx.core as mx
from gaussian_candidate import gaussian_candidate
mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=['original', 'candidate'], required=True)
args = parser.parse_args()
module = types.ModuleType('pinned_losses')
exec(compile((ROOT/'pinned-losses.py').read_bytes(), str(ROOT/'pinned-losses.py'), 'exec'), module.__dict__)
fn = module.gaussian_nll_loss if args.variant == 'original' else gaussian_candidate


class GaussianReview(unittest.TestCase):
    def compare(self, mean, target, variance, dtype, fields=('loss', 'dx', 'dy', 'dv')):
        x, y, v = (mx.array([z], dtype=dtype) for z in (mean, target, variance))
        xf, yf, vf = x.item(), y.item(), v.item()
        with localcontext() as ctx:
            ctx.prec = 80
            a,b,c = map(Decimal.from_float,(xf,yf,vf));r=a-b
            ref = {'loss':float((c.ln()+r*r/c)/2),'dx':float(r/c),
                   'dy':float(-r/c),'dv':float((c-r*r)/(2*c*c))}
        out = fn(x,y,v);gx,gy,gv=mx.grad(fn,argnums=(0,1,2))(x,y,v)
        actual={'loss':out.item(),'dx':gx.item(),'dy':gy.item(),'dv':gv.item()}
        for key in fields:
            self.assertTrue(math.isfinite(actual[key]),(key,actual,ref))
            self.assertTrue(math.isclose(actual[key],ref[key],rel_tol=0.002,abs_tol=1e-30),
                            (key,actual[key],ref[key]))

    def test_01_fp16_large_residual_finite_loss(self):
        self.compare(300,0,300,mx.float16,('loss',))
    def test_02_fp16_large_residual_variance_gradient(self):
        self.compare(300,0,300,mx.float16,('dv',))
    def test_03_fp16_finite_loss_gradient_sign(self):
        self.compare(20,0,300,mx.float16)
    def test_04_fp16_small_variance_nonzero_residual(self):
        self.compare(.02,0,.0001,mx.float16)
    def test_05_fp16_small_variance_zero_residual(self):
        self.compare(0,0,.0001,mx.float16)
    def test_06_fp16_below_epsilon_constant_branch(self):
        x=mx.array([0.],dtype=mx.float16);v=mx.array([1e-7],dtype=mx.float16)
        for g in mx.grad(fn,argnums=(0,1,2))(x,x,v):
            self.assertEqual(g.item(),0.)
    def test_07_fp32_large_residual(self):
        self.compare(1e20,0,1e20,mx.float32)
    def test_08_fp32_finite_loss_gradient_sign(self):
        self.compare(2e10,0,1e20,mx.float32)
    def test_09_benign_controls(self):
        for dtype in (mx.float16,mx.float32):
            self.compare(2,0,4,dtype)
    def test_10_mean_and_target_gradient_symmetry(self):
        self.compare(300,0,300,mx.float16,('dx','dy'))
    def test_11_full_constant_and_reductions(self):
        x=mx.array([1.,2.,-3.]);y=mx.zeros_like(x);v=mx.array([2.,5.,10.])
        element=fn(x,y,v,reduction='none')
        self.assertAlmostEqual(fn(x,y,v,reduction='sum').item(),sum(element.tolist()),places=5)
        self.assertAlmostEqual(fn(x,y,v).item(),sum(element.tolist())/3,places=5)
        full=fn(x,y,v,full=True,reduction='none')
        for a,b in zip(element.tolist(),full.tolist()):
            self.assertAlmostEqual(b-a,.5*math.log(2*math.pi),places=5)
    def test_12_shape_validation(self):
        with self.assertRaises(ValueError):
            fn(mx.ones((2,)),mx.ones((1,)),mx.ones((2,)))


result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GaussianReview))
summary={'variant':args.variant,'tests':result.testsRun,'failures':len(result.failures),
         'errors':len(result.errors),'passed':result.testsRun-len(result.failures)-len(result.errors),
         'failed_tests':[test.id() for test,_ in result.failures],
         'scope':'12 bounded tests; excludes extreme FP32 counterexample at mean=variance=3e38. Candidate widens the FP16 loss to FP32 and is not universally stable.'}
(ROOT/('mlx-regressions-'+args.variant+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
raise SystemExit(0 if result.wasSuccessful() else 1)
