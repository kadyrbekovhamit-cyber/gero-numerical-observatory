"""Content-addressed evidence and repeat-run deduplication."""
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
from importlib.metadata import distributions
import numpy as np


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def array_bytes(arr):
    buffer = io.BytesIO(); np.save(buffer, arr, allow_pickle=False); return buffer.getvalue()


def npz_bytes(arrays):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, arr in sorted(arrays.items()):
            info = zipfile.ZipInfo(name + ".npy", (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, array_bytes(arr))
    return buffer.getvalue()


def case_id(case):
    return digest({"model": digest(case.model.SerializeToString(deterministic=True)),
                   "inputs": {k: digest(array_bytes(v)) for k, v in sorted(case.feeds.items())}})


def store_artifact(root, case, outputs):
    ident = case_id(case)
    target = Path(root) / "artifacts" / ident
    target.mkdir(parents=True, exist_ok=True)
    files = {"model.onnx": case.model.SerializeToString(deterministic=True), "inputs.npz": npz_bytes(case.feeds),
             "case.json": canonical(case.metadata())}
    if case.equivalent:
        files["equivalent.onnx"] = case.equivalent.SerializeToString(deterministic=True)
    # Outputs depend on backend version/config. They belong to an observation, not a graph ID.
    for name, data in files.items(): (target / name).write_bytes(data)
    manifest = {name: digest(data) for name, data in files.items()}
    write_json(target / "manifest.json", manifest)
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted({**files, "manifest.json": canonical(manifest)}.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    (target / "reproducer.zip").write_bytes(bundle.getvalue())
    return {"model": f"artifacts/{ident}/model.onnx", "inputs": f"artifacts/{ident}/inputs.npz",
            "case": f"artifacts/{ident}/case.json", "manifest": f"artifacts/{ident}/manifest.json",
            "bundle": f"artifacts/{ident}/reproducer.zip", "bundle_sha256": digest(bundle.getvalue())}


def store_observation_bundle(root, case, row, outputs, metamorphic):
    """Self-contained source, exact installed requirements, graph and run policy."""
    directory = Path(root) / "observations" / row["observation_id"]
    directory.mkdir(parents=True, exist_ok=True)
    graph_dir = Path(root) / "artifacts" / row["case_id"]
    files = {name: (graph_dir / name).read_bytes() for name in ("model.onnx", "inputs.npz", "case.json")}
    if case.equivalent: files["equivalent.onnx"] = (graph_dir / "equivalent.onnx").read_bytes()
    files["outputs.npz"] = npz_bytes(outputs)
    files["execution.json"] = canonical({"optimization": row["optimization"], "tolerance": row["tolerance"], "metamorphic": metamorphic})
    files["environment.json"] = canonical(row["environment"])
    files["requirements.txt"] = ("\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()
                                                 if d.metadata['Name'].lower() not in ("pip", "setuptools", "gero-stability")})) + "\n").encode()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        files[f"gero_stability/{path.name}"] = path.read_bytes()
    files["REPLAY.txt"] = ("Unzip this archive into an empty directory. Use Python " + row["environment"]["python"] +
        ".\nCreate a virtual environment, then run:\npython -m pip install -r requirements.txt\n"
        "python -m gero_stability.cli replay . --output replay-results\n\n"
        "execution.json restores the exact tolerance, optimization and metamorphic policy.\n"
        "environment.json records the original machine. Timings are not expected to be identical.\n").encode()
    files["manifest.json"] = canonical({name: digest(data) for name, data in files.items()})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = buffer.getvalue()
    (directory / "reproducer.zip").write_bytes(payload)
    return {"bundle": f"observations/{row['observation_id']}/reproducer.zip", "bundle_sha256": digest(payload)}


def aggregate(paths):
    """Exact observations collapse; evidence and environment differences survive."""
    observations = {}
    for path in sorted(map(Path, paths)):
        report = json.loads(path.read_text())
        for row in report["results"]:
            observations.setdefault(row["observation_id"], row)
    return [observations[k] for k in sorted(observations)]


def export_csv(path, rows):
    with Path(path).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["observation_id", "operation", "category", "dtype", "optimization", "status", "max_abs", "max_rel", "regression"])
        for r in rows:
            metric = r.get("comparison", {})
            writer.writerow([r["observation_id"], r["operation"], r["category"], r["inputs"]["x"]["dtype"],
                             r["optimization"], r["status"], metric.get("max_abs"), metric.get("max_rel"), r["regression"]])


def export_site(report_path, destination):
    report_path, destination = Path(report_path).resolve(), Path(destination).resolve()
    report = json.loads(report_path.read_text())
    write_json(destination / "data" / "latest.json", report)
    for row in report["results"]:
        cid = row["case_id"]
        shutil.copytree(report_path.parent / "artifacts" / cid, destination / "artifacts" / cid, dirs_exist_ok=True)
        obs = row["observation_id"]
        shutil.copytree(report_path.parent / "observations" / obs, destination / "observations" / obs, dirs_exist_ok=True)
    (destination / "reports").mkdir(exist_ok=True)
    shutil.copy2(report_path.parent / "report.md", destination / "reports" / "report.md")
    export_csv(destination / "reports" / "results.csv", report["results"])
