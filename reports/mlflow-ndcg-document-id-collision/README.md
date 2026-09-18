# MLflow document-ID collision: reproduction package

Independent GERO research by Xamit Kadirbekov,18September2026. Vendor notified first: https://github.com/mlflow/mlflow/issues/25965. Candidate local only; no upstream acceptance.

Use Python3.12 in an isolated environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-repro.txt
.venv/bin/python mlflow_minimal_repro.py
GERO_REPLAY_OUTPUT=rerun .venv/bin/python mlflow_ndcg_probe.py
```

Expected full replay failures release/current/candidate/restored:214/214/0/214 out of1810 synthetic scenarios. The script executes the real released API and saved complete current module, with numerical workers limited to1. The source module and candidate correction are included. `evidence/requirements-frozen.txt` records all dependencies in the original research environment. The short requirements list pins the direct dependencies; additional resolved dependencies may change.

Read REPORT_EN.md for metric deprecation, duplicate policy, source pins, prior work and limitations. This is one ID-collision defect, not214bugs. No hosted service/model/customer-impact measurement. No full upstream suite or performance benchmark.

The unchanged upstream test function replay has its own receipt; it is not claimed to be a full pytest module execution. AI assisted the research and writing, but all reported numerical outputs were actually executed.
