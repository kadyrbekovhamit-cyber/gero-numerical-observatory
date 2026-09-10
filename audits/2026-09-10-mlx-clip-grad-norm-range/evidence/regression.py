"""Run frozen public Python code on the real MLX wheel, CPU, one thread."""
import ast
from collections import Counter
from decimal import Decimal, localcontext
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_reduce
from candidate import clip_grad_norm as after

mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parent
source = (ROOT / "upstream-python_mlx_optimizers_optimizers.py").read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "clip_grad_norm")
original = "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno]) + "\n"
(ROOT / "frozen-clip-source.py").write_text(original)
namespace = dict(mx=mx, tree_flatten=tree_flatten, tree_map=tree_map, tree_reduce=tree_reduce)
exec(compile(original, "frozen-clip-source.py", "exec"), namespace)
before = namespace["clip_grad_norm"]
rows = []
started = time.process_time()
TOL = {mx.float16: 0.004, mx.bfloat16: 0.02, mx.float32: 3e-6, mx.float64: 3e-13}

def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else "Infinity" if x > 0 else "-Infinity"
    return x

def close(x, expected, tol):
    if not math.isfinite(expected):
        return x == expected if not math.isnan(expected) else math.isnan(x)
    if not math.isfinite(x):
        return False
    return x == 0 if expected == 0 else abs(x - expected) <= tol * abs(expected)

def check(name, mode, actual, expected, tol):
    if not isinstance(actual, list):
        actual, expected = [actual], [expected]
    ok = len(actual) == len(expected) and all(close(a, b, tol) for a, b in zip(actual, expected))
    rows.append(dict(name=name, mode=mode, passed=ok, actual=actual, expected=expected, relative_tolerance=tol))

def signature(t):
    if isinstance(t, dict):
        return ("dict", [(k, signature(v)) for k, v in t.items()])
    if isinstance(t, (list, tuple)):
        return (type(t).__name__, [signature(x) for x in t])
    return ("array", t.shape)

def case(name, tree, cap):
    leaves = tree_flatten(tree)
    values = [v for _, g in leaves for v in g.reshape(-1).tolist()]
    with localcontext() as ctx:
        ctx.prec = 100
        norm = sum((Decimal.from_float(float(v)) ** 2 for v in values), Decimal(0)).sqrt()
        factor = min(Decimal(1), Decimal.from_float(float(cap)) / (norm + Decimal.from_float(1e-6)))
        exact = [float(Decimal.from_float(float(v)) * factor) for v in values]
    orig_out, orig_norm = before(tree, cap)
    norm_dtype = orig_norm.dtype
    expected_norm = mx.array(float(norm), dtype=norm_dtype).item()
    for mode, fn in (("before", before), ("after", after)):
        out, found = fn(tree, cap)
        check(name + "/norm", mode, found.item(), expected_norm, TOL[norm_dtype])
        check(name + "/structure", mode, int(signature(out) == signature(tree)), 1, 0)
        check(name + "/norm_dtype", mode, int(found.dtype == norm_dtype), 1, 0)
        offset = 0
        for (path, g), (_, expected_dtype) in zip(tree_flatten(out), tree_flatten(orig_out)):
            count = g.size
            ref = mx.array(exact[offset:offset + count], dtype=g.dtype).tolist()
            check(name + "/" + path + "/values", mode, g.reshape(-1).tolist(), ref, TOL[g.dtype])
            check(name + "/" + path + "/dtype", mode, int(g.dtype == expected_dtype.dtype), 1, 0)
            offset += count
        check(name + "/input_unchanged", mode,
              int(values == [v for _, g in tree_flatten(tree) for v in g.reshape(-1).tolist()]), 1, 0)

for dtype, powers in (
    (mx.float16, [-14, -8, 0, 6, 8, 10, 13]),
    (mx.bfloat16, [-100, -80, -10, 0, 8, 40, 60, 80, 100, 120]),
    (mx.float32, [-120, -100, -80, -10, 0, 8, 40, 60, 80, 100, 120]),
    (mx.float64, [-900, -600, -400, -10, 0, 100, 400, 600, 900]),
):
    label = str(dtype).split(".")[-1]
    for power in powers:
        scale = math.ldexp(1.0, power)
        for cap in [0.0, 1e-4, 1.0, 2.0 * scale, 8.0 * scale]:
            case(f"{label}/p{power}/cap{cap}",
                 {"a": [mx.array([3 * scale], dtype=dtype)],
                  "b": (mx.array([[4 * scale]], dtype=dtype),)}, cap)
    case(f"{label}/zeros", {"w": mx.zeros((3,), dtype=dtype)}, 1.0)
    case(f"{label}/empty_leaf", {"w": mx.zeros((0, 2), dtype=dtype)}, 1.0)
    case(f"{label}/mixed_empty", {"w": mx.zeros((0,), dtype=dtype),
                               "z": mx.array([-3., 4., 0.], dtype=dtype)}, 1.0)
    # The true norm itself can exceed the dtype limit. Clipping can still be finite.
    largest = float(mx.finfo(dtype).max)
    case(f"{label}/unrepresentable_norm", {"w": mx.array([largest, -largest], dtype=dtype)}, 1.0)
    case(f"{label}/unrepresentable_norm_large_cap", {"w": mx.array([largest, largest], dtype=dtype)}, largest)
    # Accumulation overflow, even though each square is representable.
    magnitude = math.sqrt(largest / 4.0)
    case(f"{label}/sum_overflow", {"w": mx.array([magnitude] * 32, dtype=dtype)}, 1.0)

case("empty_dict", {}, 1.0)
case("empty_nested", {"a": [], "b": ({},)}, 0.0)
case("integer_example", {"w1": mx.array([2, 3]), "w2": mx.array([1])}, 2.0)
for d1, d2 in [(mx.float16, mx.bfloat16), (mx.float16, mx.float32), (mx.float32, mx.float64)]:
    case(f"mixed/{d1}/{d2}", {"a": mx.array([-3000., 4000.], dtype=d1),
                             "b": mx.array([1000.], dtype=d2)}, 1.0)

# These coordinates remain normal in their output dtype. They catch two
# distinct failures of an otherwise tempting fixed order of operations.
for dtype, small, large, cap in [
    (mx.float32, 2.**-120, 2.**30, 2.**29),
    (mx.float32, 2.**90, 2.**100, 2.**-100),
    (mx.float64, 2.**-1000, 2.**100, 2.**99),
    (mx.float64, 2.**890, 2.**900, 2.**-900),
]:
    case(f"unbalanced/{dtype}/{small}/{cap}", {"w": mx.array([small, -large], dtype=dtype)}, cap)

# An actual optimizer step, using an exactly representable learning rate.
for dtype, power in [(mx.float16, 10), (mx.bfloat16, 80), (mx.float32, 80), (mx.float64, 600)]:
    scale = 2.**power
    tree = {"w": mx.array([3 * scale, 4 * scale], dtype=dtype)}
    for mode, fn in (("before", before), ("after", after)):
        out, _ = fn(tree, 1.0)
        updated = optim.SGD(learning_rate=0.125).apply_gradients(out, {"w": mx.zeros_like(tree["w"])})
        expected = mx.array([-0.075, -0.1], dtype=dtype).tolist()
        check(f"SGD/{dtype}", mode, updated["w"].tolist(), expected, TOL[dtype])

for mode, fn in (("before", before), ("after", after)):
    try:
        fn({"w": mx.array([1.])}, -1.0)
        rejected = False
    except ValueError:
        rejected = True
    check("negative_cap", mode, int(rejected), 1, 0)

summary = {mode: dict(checks=sum(r["mode"] == mode for r in rows),
                      failed=sum(r["mode"] == mode and not r["passed"] for r in rows))
           for mode in ("before", "after")}
result = clean(dict(version=importlib.metadata.version("mlx"), device=str(mx.default_device()),
                    cpu_seconds=time.process_time()-started, summary=summary, checks=rows))
(ROOT / "regression-results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result["summary"], indent=2))
print("CPU seconds", result["cpu_seconds"])
for row in result["checks"]:
    if row["mode"] == "after" and not row["passed"]:
        print(json.dumps(row))
raise SystemExit(bool(summary["after"]["failed"]))
