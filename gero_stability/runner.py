from datetime import datetime, timezone
import json
import platform
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path
from importlib.metadata import version
import numpy as np
import onnx
import onnxruntime as ort
from . import __version__
from .backends import benchmark, evaluator
from .compare import tolerance_for
from .compare import compare
from .invariants import output_checks, metamorphic_checks
from .storage import case_id, digest, npz_bytes, store_artifact, store_observation_bundle, write_json, export_csv


def environment():
    cpu = platform.processor()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                          text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
        except (OSError, subprocess.SubprocessError):
            cpu = f"{cpu} (CPU model unavailable)"
    elif Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    source_digest = digest({p.name: digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))})
    return {"python": platform.python_version(), "numpy": np.__version__, "onnx": onnx.__version__,
            "onnxruntime": ort.__version__, "platform": platform.platform(), "machine": platform.machine(),
            "processor": cpu, "provider": "CPUExecutionProvider", "threads": 1, "source_digest": source_digest,
            "reference_optimized": False, "deterministic_compute": True, "gero_version": __version__,
            "ort_build": ort.get_build_info(), "dependencies": {name: version(name) for name in
                ("onnx", "onnxruntime", "numpy", "protobuf", "ml_dtypes", "flatbuffers", "sympy", "packaging")}}


def baseline_state(row, baseline, current_environment):
    if baseline is None:
        return "not_assessed"
    # A runtime version may change. Changes to oracle, hardware or test semantics are incomparable.
    keys = ("numpy", "onnx", "python", "platform", "machine", "processor", "provider", "threads", "reference_optimized", "gero_version", "source_digest")
    if any(current_environment.get(k) != baseline["environment"].get(k) for k in keys):
        return "incomparable_environment"
    old_deps = {k: v for k, v in baseline["environment"].get("dependencies", {}).items() if k != "onnxruntime"}
    new_deps = {k: v for k, v in current_environment.get("dependencies", {}).items() if k != "onnxruntime"}
    if old_deps != new_deps: return "incomparable_environment"
    old = next((r for r in baseline["results"] if r["comparison_key"] == row["comparison_key"]), None)
    if old is None: return "new_coverage"
    if row["status"] in ("execution_error", "unsupported") or old["status"] in ("execution_error", "unsupported"):
        return "incomparable_execution"
    if old["status"] == "pass" and row["status"] != "pass": return "regression"
    if old["status"] != "pass" and row["status"] == "pass": return "resolved"
    return "unchanged"


def run_case(case, optimization, env, root, *, warmup=2, repeats=7, metamorphic=True, tolerance=None):
    row = {**case.metadata(), "case_id": case_id(case), "optimization": optimization, "environment": env,
           "checks": [], "timings": {}, "errors": [], "warnings": [], "outputs": {}, "regression": "not_assessed"}
    outputs = {}
    tol = tolerance or tolerance_for("float32" if case.operation == "DequantizeLinear" else case.feeds["x"].dtype)
    row["tolerance"] = tol.dict()
    row["comparison_key"] = digest({"case": row["case_id"], "optimization": optimization, "tolerance": tol.dict(),
                                     "metamorphic": metamorphic, "contract_version": __version__})
    for backend in ("reference", "ort"):
        try:
            with warnings.catch_warnings(record=True) as caught, np.errstate(all="warn"):
                warnings.simplefilter("always")
                run = evaluator(case.model, backend, optimization)
                output, timing = benchmark(run, case.feeds, warmup, repeats)
                outputs[backend] = output
                row["timings"][backend] = timing
                checks = output_checks(case, output, tol)
                row["checks"].extend({"backend": backend, **c} for c in checks)
                if metamorphic:
                    row["checks"].extend({"backend": backend, **c} for c in metamorphic_checks(case, output, backend, optimization, tol))
                row["warnings"].extend({"backend": backend, "message": str(w.message)} for w in caught)
        except Exception as exc:
            unsupported = isinstance(exc, NotImplementedError) or any(s in str(exc) for s in
                           ("NOT_IMPLEMENTED", "No implementation for operator", "No registered implementation for operator"))
            row["errors"].append({"backend": backend, "kind": "unsupported" if unsupported else "execution_error",
                                  "type": type(exc).__name__, "message": str(exc)})
    row["warnings"] = sorted({json.dumps(w, sort_keys=True) for w in row["warnings"]})
    row["warnings"] = [json.loads(w) for w in row["warnings"]]
    if len(outputs) == 2:
        row["comparison"] = compare(outputs["ort"], outputs["reference"], tol)
    if row["errors"]:
        row["status"] = "unsupported" if all(e["kind"] == "unsupported" for e in row["errors"]) else "execution_error"
    elif row["comparison"]["numerical_divergence"]:
        row["status"] = "divergence"
    elif any(not c["passed"] for c in row["checks"]):
        row["status"] = "invariant_violation"
    else:
        row["status"] = "pass"
    row["candidate_group"] = digest({"operation": case.operation, "dtype": str(case.feeds["x"].dtype),
                                      "variant": case.variant, "reason": row.get("comparison", {}).get("reason"),
                                      "failed_checks": sorted({f"{c['backend']}:{c['name']}" for c in row["checks"] if not c["passed"]})})
    row["output_hash"] = digest(npz_bytes(outputs))
    row["observation_id"] = digest({"comparison_key": row["comparison_key"], "environment": env,
                                     "output_hash": row["output_hash"], "checks": row["checks"], "errors": row["errors"],
                                     "status": row["status"]})
    row["artifacts"] = store_artifact(root, case, outputs)
    row["artifacts"].update(store_observation_bundle(root, case, row, outputs, metamorphic))
    obs_dir = Path(root) / "observations" / row["observation_id"]
    obs_dir.mkdir(parents=True, exist_ok=True)
    (obs_dir / "outputs.npz").write_bytes(npz_bytes(outputs))
    row["artifacts"]["outputs"] = f"observations/{row['observation_id']}/outputs.npz"
    write_json(obs_dir / "observation.json", row)
    return row


def run_suite(cases, root, *, optimizations=("disabled", "all"), seed=20260906, warmup=2, repeats=7,
              metamorphic=True, baseline=None, tolerance=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    env = environment()
    baseline_data = json.loads(Path(baseline).read_text()) if baseline else None
    results = []
    for i, case in enumerate(cases):
        for optimization in optimizations:
            row = run_case(case, optimization, env, root, warmup=warmup, repeats=repeats,
                           metamorphic=metamorphic, tolerance=tolerance)
            row["regression"] = baseline_state(row, baseline_data, env)
            results.append(row)
        if (i + 1) % 20 == 0: print(f"Evaluated {i+1}/{len(cases)} cases", file=sys.stderr)
    # Full observation identity, not rounded error values, defines exact duplicates.
    unique = {r["observation_id"]: r for r in results}
    results = sorted(unique.values(), key=lambda r: (r["name"], r["optimization"]))
    config = {"seed": seed, "warmup": warmup, "repeats": repeats, "optimizations": list(optimizations),
              "metamorphic": metamorphic, "excluded_operators": ["QuantizeLinear", "DynamicQuantizeLinear"]}
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "environment": env,
              "config": config, "results": results, "summary": dict(Counter(r["status"] for r in results)),
              "duplicates_removed": len(cases) * len(optimizations) - len(unique),
              "finding_graphs": len({r["case_id"] for r in results if r["status"] not in ("pass", "unsupported")}),
              "run_id": digest({"config": config, "environment": env, "observations": sorted(unique)}),
              "evidence_scope": "Local synthetic-graph CPU benchmark. A divergence is an investigation candidate, not an upstream bug or novelty claim.",
              "baseline": {"run_id": baseline_data["run_id"]} if baseline_data else None}
    write_json(root / "latest.json", report)
    # Preserve latency history even when numerical observation identity repeats.
    stamp = report["created_at"].replace(":", "").replace("+", "_")
    write_json(root / "runs" / f"{report['run_id']}-{stamp}.json", report)
    export_csv(root / "results.csv", results)
    lines = ["# GERO numerical stability report", "", f"Run: `{report['run_id']}`", "", report["evidence_scope"], "",
             f"ONNX {env['onnx']} / ONNX Runtime {env['onnxruntime']} / NumPy {env['numpy']} / {env['machine']}.", "",
             f"Summary: {report['summary']}. Excluded: QuantizeLinear and DynamicQuantizeLinear.", "",
             "Regressions are assessed only against a compatible explicit baseline. Both-NaN is never a pass.", "",
             "| Case | Optimization | Status | Max absolute error | Failed checks |", "|---|---|---|---:|---|"]
    for row in results:
        if row["status"] == "pass": continue
        failed = ", ".join(sorted({f"{c['backend']}:{c['name']}" for c in row["checks"] if not c["passed"]}))
        lines.append(f"| {row['name']} | {row['optimization']} | {row['status']} | {row.get('comparison', {}).get('max_abs', '—')} | {failed} |")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return report
