# Portable reproduction

Use a Python environment with Apple MLX 0.32.2. Copy `evidence` to a disposable directory because the scripts write results beside themselves. No MLX main build or network access is needed. Run sequentially:

```sh
cp -R evidence /tmp/mlx-groupnorm-reproduction
cd /tmp/mlx-groupnorm-reproduction
python3 build_patch.py
python3 reproduce.py
python3 regression.py
python3 compatibility.py
python3 validate_artifacts.py
```

Numerical scripts select CPU and one numerical thread. Main and compatibility CPU limits are 30 and 5 seconds. Expected recorded summaries: 332 forward + 24 gradient scenarios, 1424 assertions per variant, 45 baseline failures and zero candidate failures; 25 common compatibility checks and 16 additional candidate checks pass. See the report for numerical-comparison counts, failed prototypes, scope and remaining limitations.

Publication preparation only verified static evidence, row counts and patch identity; it did not rerun numerical suites. Frozen original outputs remain the evidence for numerical claims.
