# Reproduce on a native CPU build

Use a disposable working directory. Requirements: Git, Python 3, CMake 3.25+,
a C++20 compiler, and MLX's CPU dependencies. The recorded host is macOS arm64
with AppleClang 17 and Accelerate. No result on Linux, Windows or GPU is claimed.

Place this package in `audit/`, then clone a separate source checkout alongside it:

```sh
git clone https://github.com/ml-explore/mlx.git mlx-source
git -C mlx-source checkout ce916dbbcaa88e433b6fd1e60a17f766d49c27fe
cmake -S audit -B audit-build -DMLX_SOURCE="$PWD/mlx-source" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O0 -DNDEBUG" \
  -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_GGUF=OFF -DMLX_BUILD_SAFETENSORS=OFF -DMLX_USE_CCACHE=OFF
cmake --build audit-build --target layernorm_regression -j 1
python3 audit/collect_results.py audit-build/layernorm_regression --output before
```

The baseline executable deliberately exits **1** when assertions fail. The
recorded result is 385 checks / 106 failures. Do not join the baseline command
to the next step with `&&`. CMake may download nlohmann/json 3.11.3 and fmt 12.1.0.
The local verification used existing copies of these dependencies through
`FETCHCONTENT_SOURCE_DIR_JSON` and `FETCHCONTENT_SOURCE_DIR_FMT`.

Apply the candidate only in that disposable source checkout:

```sh
python3 audit/apply_candidate.py mlx-source
cmake --build audit-build --target layernorm_regression -j 1
python3 audit/collect_results.py audit-build/layernorm_regression --output after
```

Expected on the recorded host: 385 checks / 0 failures, exit 0. The source hash
guard prevents silently applying the edit to a different revision. Results are
written to the requested directories; published evidence is not overwritten.

The suite checks values, output dtype, unit-weight equivalence, JVP and VJP.
It does not benchmark speed. Please report your exact revision, compiler,
platform and backend with any independent reproduction.
