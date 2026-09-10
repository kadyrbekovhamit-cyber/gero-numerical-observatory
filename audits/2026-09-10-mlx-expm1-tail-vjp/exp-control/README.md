# Known float64 exp control

This scalar-libm CPU control addresses the already public [MLX issue #3047](https://github.com/ml-explore/mlx/issues/3047). It is a dependency used to isolate VJP correctness, **not a new finding**. The test overrides unary.cpp before an existing CPU archive. Other translation units using the header are not rebuilt. No performance qualification or upstream acceptance is claimed.

Apply `float64-exp.patch` to the pinned source only with the primary VJP patch. `build_and_test.py` in the parent directory builds both controls sequentially without changing the source checkout.
