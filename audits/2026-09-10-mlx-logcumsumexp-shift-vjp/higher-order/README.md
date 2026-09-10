# Preserved higher-order regression
This is the test source from the earlier report:
https://www.gero.uz/research/articles/mlx-logcumsumexp-hessian-at-zero.html

The new recurrence plus double-exp repair passes 477 comparisons over 186 scenarios. The recurrence replaces the previous VJP prototype: do not stack both VJP patches. prepare_prerequisites.py compiles pristine primitives.cpp and this regression source. No precompiled binary is distributed.
