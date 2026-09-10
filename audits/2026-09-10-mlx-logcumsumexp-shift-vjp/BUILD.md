# Reproduction
## Installed package
Use an Apple-silicon Python environment with mlx==0.32.2. Run python reproduce.py. MLX requires Metal availability during import, but this script explicitly selects CPU and sets numerical threads to one. It records observations; successful script exit alone does not mean the implementation passed.
## Native checks
Use Apple Clang, a source checkout at ce916dbbcaa88e433b6fd1e60a17f766d49c27fe, and a compatible CPU-only MLX build. Set MLX_SOURCE_ROOT to the checkout and MLX_CPU_BUILD to the build root containing compile_commands.json and mlx-build/libmlx.a.

Run sequentially in this audit directory:

    python3 prepare_prerequisites.py
    python3 known-float64-exp/build_and_test.py
    python3 build_and_test.py
    python3 validate_artifacts.py

The commands overwrite local generated logs, objects and executables, so use a separate working copy. Pristine and repaired exp/VJP variants run against the same CPU archive. Expected baseline nonzero exits are checked by the runner, while combined and higher-order tests must return zero.

The source archive does not bundle a precompiled libmlx.a. build-support retains the earlier minimal build support as a starting point. Rebuilding that complete archive from scratch was not performed during publication. Exact versions, commands and archive hashes are preserved in the JSON evidence.

The new recurrence replaces the earlier zero-cotangent VJP prototype. Do not apply that prior VJP patch first. The separate known-exp repair is required for strict float64 validation.
