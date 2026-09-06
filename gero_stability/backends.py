import time
import numpy as np
import onnxruntime as ort
from onnx.reference import ReferenceEvaluator
from .cases import assert_coverage


def evaluator(model, backend, optimization="disabled"):
    assert_coverage(model)
    if backend == "reference":
        reference = ReferenceEvaluator(model, optimized=False)
        return lambda feeds: reference.run(None, feeds)[0]
    if backend != "ort":
        raise ValueError(f"Unknown backend: {backend}")
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CPUExecutionProvider unavailable")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    options.graph_optimization_level = {"disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                                        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[optimization]
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    session.disable_fallback()
    return lambda feeds: session.run(None, feeds)[0]


def benchmark(run, feeds, warmup=2, repeats=7):
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup >= 0 and repeats >= 1 required")
    for _ in range(warmup):
        run(feeds)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = run(feeds)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return result, {"median_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
                    "samples_ms": samples, "warmup": warmup, "repeats": repeats,
                    "scope": "synchronous run including Python overhead; session construction excluded"}
