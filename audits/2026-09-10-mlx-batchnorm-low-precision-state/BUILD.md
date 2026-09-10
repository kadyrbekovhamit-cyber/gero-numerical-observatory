# Portable reproduction

Use an environment with Apple MLX 0.32.2 installed. No build of MLX main is required. Copy the evidence directory to a disposable working directory before rerunning, because the original scripts write their result JSON files beside themselves.

```sh
cp -R evidence /tmp/mlx-batchnorm-reproduction
cd /tmp/mlx-batchnorm-reproduction
python3 build_patch.py
python3 probe.py
python3 regression.py
python3 compatibility.py
python3 variance_boundary.py
python3 validate_artifacts.py
```

Run commands sequentially. Numerical scripts explicitly choose CPU, set numerical thread counts to one and limit process CPU time to 30 seconds. Use the Python executable from your MLX environment. No network access is needed for these commands.

Expected recorded summaries: main 2760 assertions per variant, 102 baseline failures and zero candidate failures; compatibility 29 baseline checks with 18 failures, 33 candidate checks with none; boundary five checks per variant, four baseline failures and none after. Assertions include API properties as well as numerical comparisons.

Publication preparation verified hashes, result counts, Python syntax and patch application. It did not perform a new numerical run. Preserved numerical logs describe the original audit only. Do not run fetch scripts to reproduce the frozen results; they record earlier source and duplicate-search acquisition.
