import argparse
import json
from pathlib import Path
import sys
import numpy as np
import onnx
from .cases import generate_cases, make_case, assert_coverage
from .compare import Tolerance
from .runner import run_suite
from .storage import aggregate, digest, export_site, write_json


def nonnegative(value):
    result = int(value)
    if result < 0: raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def positive(value):
    result = int(value)
    if result < 1: raise argparse.ArgumentTypeError("must be positive")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="GERO reproducible numerical verification")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run curated and seeded generated ONNX graphs")
    run.add_argument("--output", type=Path, default=Path("reports/local"))
    run.add_argument("--seed", type=nonnegative, default=20260906)
    run.add_argument("--random-cases", type=nonnegative, default=12)
    run.add_argument("--operation", help="Exact operation filter")
    run.add_argument("--dtype", choices=["float16", "float32", "float64", "int8", "uint8"])
    run.add_argument("--optimization", choices=["disabled", "all", "both"], default="both")
    run.add_argument("--warmup", type=nonnegative, default=2)
    run.add_argument("--repeats", type=positive, default=7)
    run.add_argument("--baseline", type=Path)
    run.add_argument("--atol", type=float)
    run.add_argument("--rtol", type=float)
    run.add_argument("--no-metamorphic", action="store_true")
    run.add_argument("--fail-on", choices=["none", "finding", "regression"], default="none")
    replay = sub.add_parser("replay", help="Verify hashes and replay an artifact directory")
    replay.add_argument("artifact", type=Path)
    replay.add_argument("--output", type=Path, default=Path("reports/replay"))
    replay.add_argument("--optimization", choices=["disabled", "all", "both"])
    replay.add_argument("--rtol", type=float)
    replay.add_argument("--atol", type=float)
    export = sub.add_parser("export", help="Prepare static gero.uz dashboard data and downloads")
    export.add_argument("report", type=Path)
    export.add_argument("--site", type=Path, default=Path("site"))
    agg = sub.add_parser("aggregate", help="Deduplicate exact observations across reports")
    agg.add_argument("reports", nargs="+", type=Path)
    agg.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run" and args.fail_on == "regression" and args.baseline is None:
        parser.error("--fail-on regression requires --baseline")
    if args.command == "export":
        export_site(args.report, args.site)
        print(f"Static dashboard exported to {args.site}"); return 0
    if args.command == "aggregate":
        rows = aggregate(args.reports)
        write_json(args.output, {"schema_version": 1, "results": rows, "unique_observations": len(rows)})
        print(f"{len(rows)} unique observations"); return 0
    if args.command == "replay":
        manifest = json.loads((args.artifact / "manifest.json").read_text())
        required = {"model.onnx", "inputs.npz", "case.json"}
        if not required.issubset(manifest): parser.error("Incomplete artifact manifest")
        for name, expected in manifest.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or not (args.artifact / relative).resolve().is_relative_to(args.artifact.resolve()):
                parser.error("Invalid artifact path")
            if digest((args.artifact / name).read_bytes()) != expected: parser.error(f"SHA-256 mismatch: {name}")
        execution = json.loads((args.artifact / "execution.json").read_text()) if "execution.json" in manifest else {}
        metadata = json.loads((args.artifact / "case.json").read_text())
        with np.load(args.artifact / "inputs.npz", allow_pickle=False) as archive:
            feeds = {k: archive[k] for k in archive.files}
        case = make_case(metadata["operation"], feeds, metadata["variant"], metadata["seed"], metadata["parameters"])
        case.model = onnx.load(args.artifact / "model.onnx", load_external_data=False)
        assert_coverage(case.model)
        onnx.checker.check_model(case.model, full_check=True)
        case.equivalent = onnx.load(args.artifact / "equivalent.onnx", load_external_data=False) if "equivalent.onnx" in manifest else None
        optimization = args.optimization or execution.get("optimization", "both")
        if (args.rtol is None) != (args.atol is None): parser.error("Set both --rtol and --atol")
        try:
            tol = Tolerance(args.rtol, args.atol) if args.rtol is not None else (Tolerance(**execution["tolerance"]) if "tolerance" in execution else None)
        except ValueError as exc: parser.error(str(exc))
        report = run_suite([case], args.output, seed=case.seed,
                           optimizations=("disabled", "all") if optimization == "both" else (optimization,),
                           tolerance=tol, metamorphic=execution.get("metamorphic", True))
    else:
        cases = generate_cases(args.seed, args.random_cases)
        if args.operation: cases = [c for c in cases if c.operation == args.operation]
        if args.dtype: cases = [c for c in cases if str(c.feeds["x"].dtype) == args.dtype]
        if not cases: parser.error("No cases selected; QuantizeLinear is explicitly excluded")
        if (args.rtol is None) != (args.atol is None): parser.error("Set both --rtol and --atol")
        try: tol = Tolerance(args.rtol, args.atol) if args.rtol is not None else None
        except ValueError as exc: parser.error(str(exc))
        report = run_suite(cases, args.output, seed=args.seed, warmup=args.warmup, repeats=args.repeats,
                           optimizations=("disabled", "all") if args.optimization == "both" else (args.optimization,),
                           baseline=args.baseline, metamorphic=not args.no_metamorphic, tolerance=tol)
        if any(r["status"] == "execution_error" for r in report["results"]):
            print(json.dumps(report["summary"])); return 3
        if args.fail_on == "finding" and any(r["status"] != "pass" for r in report["results"]):
            print(json.dumps(report["summary"])); return 2
        if args.fail_on == "regression" and any(r["regression"] == "regression" for r in report["results"]):
            print(json.dumps(report["summary"])); return 2
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {args.output / 'latest.json'}")
    if any(r["status"] == "execution_error" for r in report["results"]): return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
