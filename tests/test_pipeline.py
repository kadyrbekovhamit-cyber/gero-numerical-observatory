import copy
import json
import numpy as np
import zipfile
from gero_stability.compare import Tolerance
from gero_stability.cases import make_case
from gero_stability.runner import run_suite, baseline_state
from gero_stability.storage import aggregate, export_site, npz_bytes
from gero_stability.cli import main


def test_real_run_replay_dedup_export_and_baseline(tmp_path):
    case = make_case("Softmax", {"x": np.array([[1., 2., 3.], [3., 2., 1.]], np.float32)})
    a = run_suite([case, case], tmp_path / "a", warmup=0, repeats=1)
    b = run_suite([case], tmp_path / "b", warmup=0, repeats=1, baseline=tmp_path / "a/latest.json")
    assert a["summary"] == {"pass": 2}
    assert all(r["regression"] == "unchanged" for r in b["results"])
    assert len(aggregate([tmp_path / "a/latest.json", tmp_path / "b/latest.json"])) == 2
    export_site(tmp_path / "a/latest.json", tmp_path / "site")
    assert (tmp_path / "site/data/latest.json").is_file()
    assert (tmp_path / "site" / a["results"][0]["artifacts"]["bundle"]).is_file()
    artifact = tmp_path / "a/artifacts" / a["results"][0]["case_id"]
    assert main(["replay", str(artifact), "--output", str(tmp_path / "replay")]) == 0
    (artifact / "model.onnx").write_bytes(b"corrupted")
    import pytest
    with pytest.raises(SystemExit): main(["replay", str(artifact), "--output", str(tmp_path / "invalid")])


def test_regression_requires_comparable_passing_baseline():
    env = {"onnx": "1", "onnxruntime": "2", "machine": "arm64"}
    row = {"comparison_key": "same", "status": "divergence"}
    baseline = {"environment": env, "results": [{"comparison_key": "same", "status": "pass"}]}
    assert baseline_state(row, None, env) == "not_assessed"
    assert baseline_state(row, baseline, {**env, "onnxruntime": "3"}) == "regression"
    assert baseline_state(row, baseline, {**env, "onnx": "2"}) == "incomparable_environment"
    assert baseline_state(row, baseline, {**env, "machine": "x86_64"}) == "incomparable_environment"
    assert baseline_state({**row, "comparison_key": "different"}, baseline, env) == "new_coverage"


def test_npz_deterministic_bytes_and_order():
    a, b = np.array([1., 2.]), np.array([3., 4.])
    assert npz_bytes({"a": a, "b": b}) == npz_bytes({"b": b, "a": a})


def test_bundle_restores_custom_tolerance_and_optimization(tmp_path):
    case = make_case("Softmax", {"x": np.array([[1., 2., 3.]], np.float32)})
    report = run_suite([case], tmp_path / "run", optimizations=("all",), tolerance=Tolerance(.001, .0001), warmup=0, repeats=1)
    row = report["results"][0]
    with zipfile.ZipFile(tmp_path / "run" / row["artifacts"]["bundle"]) as archive:
        archive.extractall(tmp_path / "unpacked")
    assert main(["replay", str(tmp_path / "unpacked"), "--output", str(tmp_path / "replay")]) == 0
    replay = json.loads((tmp_path / "replay/latest.json").read_text())
    assert len(replay["results"]) == 1
    assert replay["results"][0]["comparison_key"] == row["comparison_key"]


def test_both_backends_nonfinite_does_not_become_pass_or_divergence(tmp_path):
    case = make_case("Softmax", {"x": np.array([[np.nan, np.nan]], np.float32)})
    report = run_suite([case], tmp_path, optimizations=("disabled",), warmup=0, repeats=1)
    assert report["summary"] == {"invariant_violation": 1}
    assert not report["results"][0]["comparison"]["passed"]
