"""Focused CPU tests of current Muon and a local patch, with tiny tensors."""

import ast
import io
import json
import math
import os
from pathlib import Path
import resource
import time
import types
import unittest

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))

import mlx.core as mx
from mlx.utils import tree_map, tree_flatten

mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parent


def load(name, path):
    module = types.ModuleType(name)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def close(test, actual, expected, label):
    difference = mx.max(mx.abs(actual-expected)).item()
    test.assertTrue(mx.allclose(actual, expected, atol=2e-6, rtol=2e-5).item(), f"{label}: max error {difference}")


def build_suite(module):
    class Checks(unittest.TestCase):
        pass

    shapes = [(8, 1, 1, 2), (1, 1, 8, 2), (2, 2, 2, 2), (8, 1, 2), (2, 8, 2), (2, 1, 1, 8, 2), (8, 2), (2, 8)]
    configurations = [
        dict(momentum=0.0, nesterov=False, weight_decay=0.0),
        dict(momentum=0.95, nesterov=True, weight_decay=0.0),
        dict(momentum=0.7, nesterov=False, weight_decay=0.03),
    ]

    def make_reshape_test(shape, config):
        def test(self):
            flat = (shape[0], math.prod(shape[1:]))
            n = math.prod(shape)
            initial = ((mx.arange(n, dtype=mx.float32) % 9)-4)/16
            shaped, matrix = initial.reshape(shape), initial.reshape(flat)
            opt_s = module.Muon(learning_rate=0.01, **config)
            opt_m = module.Muon(learning_rate=0.01, **config)
            for step in range(3):
                grad = (((mx.arange(n, dtype=mx.float32)+3*step) % 11)-5)/8
                gs, gm = grad.reshape(shape), grad.reshape(flat)
                original_parameter, original_gradient = shaped.tolist(), gs.tolist()
                next_s = opt_s.apply_gradients({"w": gs}, {"w": shaped})["w"]
                next_m = opt_m.apply_gradients({"w": gm}, {"w": matrix})["w"]
                self.assertEqual(next_s.shape, shape)
                self.assertEqual(next_s.dtype, mx.float32)
                close(self, next_s.reshape(flat), next_m, f"shape={shape}, step={step}")
                close(self, opt_s.state['w']['v'].reshape(flat), opt_m.state['w']['v'], 'momentum state')
                self.assertEqual(shaped.tolist(), original_parameter)
                self.assertEqual(gs.tolist(), original_gradient)
                shaped, matrix = next_s, next_m
        return test

    for i, shape in enumerate(shapes):
        for j, config in enumerate(configurations):
            setattr(Checks, f"test_reshape_{i}_config_{j}", make_reshape_test(shape, config))

    def make_analytic_test(shape):
        def test(self):
            n = math.prod(shape)
            values = [(i+1)/16 for i in range(n)]
            grad = mx.array(values).reshape(shape)
            matrix_shape = (shape[0], math.prod(shape[1:]))
            norm = math.hypot(*values)
            factor = math.sqrt(max(1, matrix_shape[0]/matrix_shape[1]))
            expected = mx.array([-0.01*factor*v/(norm+1e-7) for v in values]).reshape(shape)
            opt = module.Muon(0.01, momentum=0, nesterov=False, weight_decay=0, ns_steps=0)
            actual = opt.apply_gradients({'w':grad},{'w':mx.zeros(shape)})['w']
            close(self, actual, expected, 'independent analytic zero-iteration normalization')
        return test

    for i, shape in enumerate([(8,1,1,2),(1,1,8,2),(8,2),(2,8)]):
        setattr(Checks, f'test_analytic_normalization_{i}', make_analytic_test(shape))

    def make_conv_test(momentum):
        def test(self):
            x = mx.array([[[[1.0,-0.5],[0.25,0.75]],[[-1.0,0.5],[0.5,1.0]]]])
            w = ((mx.arange(16,dtype=mx.float32)+1)/32).reshape(8,2)
            w4 = w.reshape(8,1,1,2)
            f2 = lambda z: mx.mean(mx.square(x @ z.T))
            f4 = lambda z: mx.mean(mx.square(mx.conv2d(x,z)))
            close(self, x@w.T, mx.conv2d(x,w4), 'forward')
            close(self, f2(w), f4(w4), 'loss')
            g2, g4 = mx.grad(f2)(w), mx.grad(f4)(w4)
            close(self, g2, g4.reshape(8,2), 'gradients')
            o2 = module.Muon(.01,momentum=momentum,nesterov=True,weight_decay=0)
            o4 = module.Muon(.01,momentum=momentum,nesterov=True,weight_decay=0)
            u2 = o2.apply_gradients({'w':g2},{'w':w})['w']
            u4 = o4.apply_gradients({'w':g4},{'w':w4})['w']
            close(self,u2,u4.reshape(8,2),'equivalent function, same gradient, same update')
        return test

    for i, momentum in enumerate([0.0,0.95]):
        setattr(Checks, f'test_real_conv_linear_{i}',make_conv_test(momentum))

    # Execute the unchanged upstream test body with its unchanged helper.
    # The full test module would also import optional PyTorch; that is unnecessary here.
    tree = ast.parse((ROOT/'upstream-test_optimizers.py').read_text())
    helper = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='tree_equal')
    test_class = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='TestOptimizers')
    body = next(node for node in test_class.body if isinstance(node,ast.FunctionDef) and node.name=='test_muon')
    context = {'mx':mx,'opt':module,'tree_map':tree_map,'tree_flatten':tree_flatten}
    exec(compile(ast.Module(body=[helper,body],type_ignores=[]),'upstream-test_optimizers.py','exec'),context)
    setattr(Checks,'test_upstream_muon_unchanged',context['test_muon'])
    return unittest.defaultTestLoader.loadTestsFromTestCase(Checks)


summary = {}
for label, filename in [('before','upstream-optimizers.py'),('after','patched-optimizers.py')]:
    module = load('optimizers_'+label, ROOT/filename)
    stream = io.StringIO()
    started = time.process_time()
    result = unittest.TextTestRunner(stream=stream,verbosity=2).run(build_suite(module))
    (ROOT/f'{label}.log').write_text(stream.getvalue())
    summary[label] = {'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
                      'passed':result.testsRun-len(result.failures)-len(result.errors),
                      'failure_names':[str(case) for case,_ in result.failures],
                      'cpu_seconds':time.process_time()-started}
    print(label,json.dumps(summary[label]),flush=True)
(ROOT/'regression-results.json').write_text(json.dumps(summary,indent=2)+'\n')
if summary['before']['failures']==0 or summary['before']['errors'] or summary['after']['failures'] or summary['after']['errors']:
    raise SystemExit(1)
