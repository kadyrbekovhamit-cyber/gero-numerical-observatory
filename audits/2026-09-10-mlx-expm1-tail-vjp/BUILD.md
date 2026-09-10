# Native CPU build setup

The publication rerun reuses a pre-existing CPU archive at base
`ce916dbbcaa88e433b6fd1e60a17f766d49c27fe`. It compiles primitives.cpp and unary.cpp in two variants and relinks their four combinations before that archive. It is not a fresh
build of main. `build-results.json` preserves commands with path placeholders.

To prepare an analogous archive on an Apple-silicon Mac with CMake ≥3.25
and Apple command-line developer tools:

```sh
git clone https://github.com/ml-explore/mlx.git mlx-source
git -C mlx-source checkout ce916dbbcaa88e433b6fd1e60a17f766d49c27fe
cmake -S build-support -B cpu-build \
  -DAUDIT_MLX_SOURCE="$PWD/mlx-source" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DBUILD_SHARED_LIBS=OFF -DMLX_BUILD_CPU=ON \
  -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_GGUF=OFF -DMLX_BUILD_SAFETENSORS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF
cmake --build cpu-build --target mlx --parallel 1
export MLX_SOURCE_ROOT="$PWD/mlx-source"
export MLX_CPU_BUILD="$PWD/cpu-build"
python3 build_and_test.py
```

The setup recipe is supplied for portability from the recorded CMake
configuration; this fresh archive build was **not executed** for publication.
Dependency retrieval needs network access. Compiler/SDK/dependency differences
can change archive hashes and floating-point behavior; retain your own logs.
The runner's tested path uses the existing archive whose SHA-256 is recorded
in native-source-metadata.json. It writes generated objects, binaries and fresh logs under rerun/. These large generated files are excluded from the evidence archive. The stored run-*.json files retain the original audit results.
