"""Small source, API, nonfinite, and autodiff compatibility checks."""
import ast
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time
from types import SimpleNamespace
import unittest
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_reduce
from candidate import clip_grad_norm as after
mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parent
namespace = dict(mx=mx, tree_flatten=tree_flatten, tree_map=tree_map, tree_reduce=tree_reduce)
exec((ROOT / "frozen-clip-source.py").read_text(), namespace)
before = namespace["clip_grad_norm"]
rows = []
started = time.process_time()

def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, complex): return dict(real=clean(x.real), imag=clean(x.imag))
    if isinstance(x, float) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else "Infinity" if x > 0 else "-Infinity"
    return x

def check(name, mode, actual, expected, tol=0):
    if not isinstance(actual, list): actual, expected = [actual], [expected]
    ok = len(actual) == len(expected) and all(
        (math.isnan(a) and math.isnan(b)) if math.isnan(b) else
        a == b if not math.isfinite(b) else
        math.isfinite(a) and abs(a-b) <= tol * max(abs(b), 1e-12)
        for a,b in zip(actual, expected))
    rows.append(dict(name=name, mode=mode, passed=ok, actual=actual, expected=expected, relative_tolerance=tol))

# Execute the exact public test method with unittest assertions.
test_source = (ROOT / "upstream-python_tests_test_optimizers.py").read_text()
nodes = ast.parse(test_source).body
helper = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == "tree_equal")
method = next(n for klass in nodes if isinstance(klass, ast.ClassDef)
              for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "test_clip_grad_norm")
for mode, fn in (("before", before), ("after", after)):
    env = dict(mx=mx, opt=SimpleNamespace(clip_grad_norm=fn), tree_flatten=tree_flatten, tree_map=tree_map)
    exec(compile(ast.Module(body=[helper, method], type_ignores=[]), "upstream-test-method", "exec"), env)
    test = type("FrozenUpstreamClipTest", (unittest.TestCase,), {"test_clip": env[method.name]})("test_clip")
    result = unittest.TestResult()
    test.run(result)
    check("upstream_test_clip_grad_norm", mode, int(result.wasSuccessful()), 1)

for dtype, powers in ((mx.float32, [0, 40, 80, 120]), (mx.float64, [0, 100, 600, 900])):
    tol = 1e-5 if dtype == mx.float32 else 5e-12
    for power in powers:
        c = 2.**power
        d = 5 + 1e-6/c
        first = [3/d - 27/(5*d*d), -36/(5*d*d)]
        second = [-6*1.8/(d*d)-3*1.152/(d*d)+6*1.8**2/(d**3),
                  -4*1.152/(d*d)+8*1.8**2/(d**3)]
        for mode, fn in (("before", before), ("after", after)):
            t = mx.array(1., dtype=dtype)
            def f(t):
                g = mx.stack([3*c*t, mx.array(4*c, dtype=dtype)])
                return fn({"w": g}, 1.)[0]["w"]
            _, jvp = mx.jvp(f, [t], [mx.ones_like(t)])
            _, vjp = mx.vjp(f, [t], [mx.array([2., -3.], dtype=dtype)])
            scalar = lambda t: mx.sum(f(t) * mx.array([2., -3.], dtype=dtype))
            hess = mx.grad(mx.grad(scalar))(t)
            check(f"AD/{dtype}/p{power}/JVP", mode, jvp[0].tolist(), first, tol)
            check(f"AD/{dtype}/p{power}/VJP", mode, vjp[0].item(), 2*first[0]-3*first[1], tol)
            check(f"AD/{dtype}/p{power}/second", mode, hess.item(), 2*second[0]-3*second[1], tol)
    for point, cap, expected in [([0.,0.],1.,[1.,2.]), ([0.,0.],0.,[0.,0.]), ([.1,.2],1.,[1.,2.])]:
        for mode, fn in (("before", before), ("after", after)):
            g = mx.array(point, dtype=dtype)
            _, found = mx.jvp(lambda z: fn({"w": z}, cap)[0]["w"], [g], [mx.array([1.,2.], dtype=dtype)])
            check(f"AD/{dtype}/{point}/cap{cap}", mode, found[0].tolist(), expected, tol)

# Audit rather than assume behavior outside the finite-real claim.
for values, cap in [([float('inf'),1.],1.), ([float('nan'),1.],1.),
                    ([3.,4.],float('nan')), ([3.,4.],float('inf')),
                    ([float('inf'),1.],float('inf')), ([float('nan'),1.],float('inf'))]:
    g = {"w": mx.array(values)}
    base, _ = before(g, cap)
    out, _ = after(g, cap)
    check(f"nonfinite/{values}/cap{cap}", "after", out["w"].tolist(), base["w"].tolist())

for values in [[1+2j, 3-4j], [0j, 0j]]:
    outcomes = []
    for fn in (before, after):
        try:
            out, norm = fn({"w": mx.array(values, dtype=mx.complex64)}, 1.)
            outcomes.append(clean(dict(out=out["w"].tolist(), norm=norm.item())))
        except Exception as e:
            outcomes.append(dict(exception=type(e).__name__, message=str(e)))
    check(f"complex_fallback/{values}", "after", int(outcomes[0] == outcomes[1]), 1)

summary = {mode: dict(checks=sum(r["mode"] == mode for r in rows),
                      failed=sum(r["mode"] == mode and not r["passed"] for r in rows))
           for mode in ("before", "after")}
result = clean(dict(version=importlib.metadata.version("mlx"), device=str(mx.default_device()),
                    cpu_seconds=time.process_time()-started, summary=summary, checks=rows))
(ROOT / "compatibility-results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
print(json.dumps(summary, indent=2))
print("CPU seconds", result["cpu_seconds"])
for row in result["checks"]:
    if row["mode"] == "after" and not row["passed"]: print(json.dumps(row))
raise SystemExit(bool(summary["after"]["failed"]))
