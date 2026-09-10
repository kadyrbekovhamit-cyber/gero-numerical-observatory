# Reproduce the clipping audit

Use a Python environment with Apple MLX 0.32.2 and its dependencies on a supported CPU platform. The checked environment was macOS on Apple silicon. The baseline and patch use the actual installed MLX runtime; there is no native rebuild and no GPU computation.

From this archive directory:

```sh
python run_reproduction.py --output /tmp/mlx-clip-reproduction
```

Alternatively, select an existing MLX environment explicitly:

```sh
python3 run_reproduction.py --python /path/to/mlx-env/bin/python --output /tmp/mlx-clip-reproduction
```

The output directory must not exist. The runner first verifies the original evidence hashes, copies the saved inputs into that directory, and runs `regression.py`, `compatibility.py` and `validate_artifacts.py` sequentially. It compares every new numerical row with the archived rows, ignoring only elapsed process CPU time. The originals remain unchanged. No network is required once the runtime is installed.

The numerical scripts set OMP, OpenBLAS, MKL, vecLib and NumExpr thread counts to one before importing MLX, explicitly select `mx.cpu` and limit CPU time to 30 seconds per process. This is a thread limit, not physical CPU affinity or a claim about other processes on the computer.

Expected results: main before 1701 checks / 230 failures; main after 1701 / 0. Compatibility before 31 / 17; after 39 / 0, including eight additional preservation controls. The archived artifact validator reports 66 / 0.

The candidate is imported in isolation. Nothing is installed over the user's MLX package. To review the proposed production change, inspect `evidence/clip-grad-norm-range.patch`; the artifact validator applies it only in a temporary copy and compares its body with the tested candidate.
