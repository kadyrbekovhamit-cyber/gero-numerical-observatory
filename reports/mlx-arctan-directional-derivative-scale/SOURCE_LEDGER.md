# Source ledger

- MLX source pin: `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`, official GitHub repository.
- Executed code: C++ CPU `float32` `ArcTan::jvp`/VJP through MLX autodiff.
- Mathematical oracle: GERO `check_arctan.py`, 160-decimal `mpmath`, direct binary32 rounding.
- Release comparison: official `v0.32.2` source; runtime not executed.
- Related official records: issues 3778, PRs 3779 and 4227.
- Graphics for companion video: original GERO vector/raster assets only.
- Voice for companion video: Microsoft edge-tts `en-US-JennyNeural`, synthetic, rate `-3%`; no playback during production.
