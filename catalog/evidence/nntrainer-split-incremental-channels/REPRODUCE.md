# Replay the original, corrected and restored C++ layer

Requirements: macOS arm64, Python >=3.9, C++17 compiler, Meson and Ninja already available. This adapter was built for the recorded Mac environment; other platforms require their own build configuration and have not been validated here. Allow at least 600 MiB of free working space. No Docker, GPU, download, installation or audio playback is required by the runner.

From this package directory, run:

```sh
python3 -B reproduce.py --output /absolute/path/to/new-replay
```

Optionally add `--tools-bin /path/to/existing/bin` to use an existing Meson/Ninja environment. Never name the package directory or a previous result directory as the output. An existing output directory is rejected.

The script reads only the included source archives and local build tools, configures `--wrap-mode=nodownload`, builds one native worker at a time, and limits numerical library threads to one. It verifies the candidate patch roundtrip before building the original core. Each subsequent probe compiles the real SplitLayer translation unit, links the freshly built core and runs all scenarios. The candidate variant changes only the target translation unit used by the probe.

Expected receipt: `REPLAY_RECEIPT.json` with `status=clean_portable_replay_verified`. Each variant has 576 scenarios and 19,296 output coordinates. Incremental mismatch counts must be 4,320, zero and 4,320; full-forward and input-preservation failure counts must all be zero. The new raw TSVs must match the archived SHA256 values exactly. All 603 required native source/header files remain unchanged, along with all other included manifest-matching source files.

The output includes compiler/Meson/Ninja versions, every build/probe command, logs, variant summaries, the four-coordinate minimal example, and executable/library hashes. The library hash may vary with compiler and build path; it is recorded for that run and is not required to equal the old binary hash. Raw-output hashes and source hashes are the fixed reproduction gates.

The retained `evidence/clean-*` files are historical outputs from 15 September. Their absolute local paths and `published=false` fields describe that experiment's time and are not current publication status. This standalone runner does not depend on those old paths and does not rewrite those files.

The sentinel `-99999` is deliberately placed in output buffers before each incremental call. It is evidence that a coordinate was left unwritten, not a claimed production output. Exact integer-index and Fraction arithmetic defines the independent expected result; no floating-point tolerance is applied.
