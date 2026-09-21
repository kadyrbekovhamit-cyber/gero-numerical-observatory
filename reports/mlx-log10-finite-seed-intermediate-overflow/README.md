# Complete reproduction packet

Download and extract [gero-mlx-log10-impact-evidence-2026-09-21.zip](gero-mlx-log10-impact-evidence-2026-09-21.zip) first. The ZIP contains the runnable sources and expected outputs; loose files are browsing excerpts.

# Reproduce the MLX log10 evidence

The packet contains a deliberately extreme loss-scaling experiment, not a deployment report. Read `REPORT_EN.md` before using the experimental patch: it retains four known boundary errors.

## Contents

- `evidence/primitive`: 294 inputs, two derivative directions, three layouts; original, candidate and restored raw outputs; independent 80/120-digit oracle and direct binary32 rounding checker.
- `evidence/model`: predeclared experiment, normalized digit data and split IDs, trained checkpoint, 54 one-step branch results, saved gradients/weights/predictions, independent analytic references and control comparison.
- `code`: actual C++ MLX drivers and standard-library Python replay.
- `source`: complete pinned MLX source archive and required fmt/json headers with their original licenses. No prebuilt executable is distributed.
- `evidence/provenance`: build configuration and source/object/library hashes used for the reported experiment.

## Build on macOS with Xcode command-line tools

Extract `source/mlx-59d600b5e64c238427d0f8d897ab7c682ef4d3d2.tar.gz` to a local source directory. Use CMake and Ninja from official distributions. A clean complete CPU build can be configured as follows (replace the paths with local absolute paths):

```sh
cmake -S /path/to/mlx-source -B /path/to/mlx-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_GGUF=OFF -DMLX_BUILD_SAFETENSORS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF -DMLX_USE_CCACHE=OFF
cmake --build /path/to/mlx-build --parallel 1
```

CMake may fetch its declared fmt/json dependencies. The included headers are fmt 12.1.0 and nlohmann/json 3.11.3; licenses remain in `source`. Locate the resulting complete `libmlx.a` (the reported parent build placed it in `build/mlx-build/libmlx.a`). No GPU is needed. This replay driver uses Apple's Accelerate framework and was validated on macOS; it is not presented as a tested Linux/Windows runner.

Then, from the extracted evidence packet:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 code/replay.py --source /path/to/mlx-source --library /path/to/libmlx.a --output /path/to/new-replay-directory
```

The output directory must not already exist. The runner validates every packet checksum and source primitive identity, freshly compiles the full original/candidate primitive translation units and both drivers, links against the supplied full library, recomputes the high-precision oracle, reruns the primitive grid, retrains the checkpoint, repeats all 54 one-step branches and checks stored raw outputs exactly. It retains `REPLAY_RECEIPT.json` and compile/run logs. The model reference uses independent binary64 analytic gradients with absolute tolerance `1e-7`.

Exact historical bit comparisons are deliberately strict. A different compiler/CPU/library build may change low bits; if a comparison fails, inspect raw outputs and independent references. Do not silently loosen the tolerances or claim historical reproduction.

The reported local archive replay reuses the previously completed pinned CPU library and freshly recompiles the full changed/restored primitive translation units and drivers. It is **not** a claim that the whole MLX library was rebuilt from scratch for this case. The installed release Python package did not execute; release-source identity is separate evidence.

Original acquisition scripts under `evidence` retain original machine paths for provenance. Use `code/replay.py` for the path-parameterized replay. `run_impact.py` documents dataset acquisition but is not required for replay because the exported split data and attribution are included.

No issue, publication, video or real-world loss is proved by this local packet. Publication and disclosure receipts are maintained separately.
