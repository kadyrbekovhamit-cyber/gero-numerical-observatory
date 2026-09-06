#!/usr/bin/env python3
"""Reproduce uncertainty bounds after observing zero failures.

The exact bound is the one-sided Clopper-Pearson upper confidence bound for
zero observed failures. The Wilson value is the upper endpoint of a two-sided
score interval. They answer slightly different reporting questions, so both
are printed and labelled explicitly.
"""

from __future__ import annotations

import argparse
import math
from statistics import NormalDist


def exact_one_sided_upper_zero(n: int, confidence: float = 0.95) -> float:
    """Return p_U such that P(X=0 | p_U, n) = 1-confidence."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    alpha = 1.0 - confidence
    return 1.0 - alpha ** (1.0 / n)


def wilson_two_sided_upper_zero(n: int, confidence: float = 0.95) -> float:
    """Return the upper endpoint of a two-sided Wilson interval for x=0."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z2 = z * z
    return z2 / (n + z2)


def minimum_zero_failure_sample(target_upper: float, confidence: float = 0.95) -> int:
    """Smallest n whose exact one-sided upper bound is at most target_upper."""
    if not 0.0 < target_upper < 1.0:
        raise ValueError("target_upper must lie strictly between 0 and 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    alpha = 1.0 - confidence
    return math.ceil(math.log(alpha) / math.log(1.0 - target_upper))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--samples", type=int, nargs="+", default=[2, 10, 30, 59, 299])
    args = parser.parse_args()

    print("n,observed_rate,exact_one_sided_upper,wilson_two_sided_upper")
    for n in args.samples:
        exact = exact_one_sided_upper_zero(n, args.confidence)
        wilson = wilson_two_sided_upper_zero(n, args.confidence)
        print(f"{n},0.000000,{exact:.6f},{wilson:.6f}")

    print(
        f"minimum_n_for_5_percent={minimum_zero_failure_sample(0.05, args.confidence)}"
    )
    print(
        f"minimum_n_for_1_percent={minimum_zero_failure_sample(0.01, args.confidence)}"
    )


if __name__ == "__main__":
    main()

