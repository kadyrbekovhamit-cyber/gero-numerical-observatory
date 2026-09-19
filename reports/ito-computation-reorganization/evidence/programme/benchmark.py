#!/usr/bin/env python3
"""Small, serial, standard-library stochastic-integral benchmark.

Run from any directory: python3 programme/benchmark.py
No network, subprocesses, BLAS, workers, adaptive stopping, or parameter fitting.
Ground-truth bridge areas never enter estimator arguments.
"""

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify import prices  # noqa: E402 -- imported module has a guarded entry point

SEEDS = (17, 43, 101, 271, 577)
PATHS = 256
FINE_N = 64
GRIDS = (8, 16, 32, 64)
T = 1.0
TIMING_PATHS = 64
TIMING_REPEATS = 5
REPEATED_QUERIES = 3
T_CRITICAL_DF4 = 2.7764451051977987


def t_left(w, h, x):
    total = 0.0
    for i in range(len(w) - 1):
        total += (i * h) * (w[i + 1] - w[i])
    return total


def t_optimal(w, h, x):
    total = 0.0
    for i in range(len(w) - 1):
        total += ((i + 0.5) * h) * (w[i + 1] - w[i])
    return total


def w_left(w, h, x):
    total = 0.0
    for i in range(len(w) - 1):
        total += w[i] * (w[i + 1] - w[i])
    return total


def w_left_qv(w, h, x):
    qv = 0.0
    for i in range(len(w) - 1):
        d = w[i + 1] - w[i]
        qv += d * d
    return 0.5 * (w[-1] * w[-1] - qv)


def w_milstein(w, h, x):
    total = 0.0
    for i in range(len(w) - 1):
        d = w[i + 1] - w[i]
        total += w[i] * d + 0.5 * (d * d - h)
    return total


def w_exact_endpoint(w, h, x):
    return 0.5 * (w[-1] * w[-1] - (len(w) - 1) * h)


def w2_left(w, h, x):
    total = 0.0
    for i in range(len(w) - 1):
        total += w[i] * w[i] * (w[i + 1] - w[i])
    return total


def w2_milstein(w, h, x):
    total = 0.0
    for i in range(len(w) - 1):
        a = w[i]
        d = w[i + 1] - a
        total += a * a * d + a * (d * d - h)
    return total


def w2_optimal_expanded(w, h, x):
    total = 0.0
    for i in range(len(w) - 1):
        a = w[i]
        d = w[i + 1] - a
        total += a * a * d + a * (d * d - h) + d * d * d / 3 - h * d / 2
    return total


def w2_optimal_compressed(w, h, x):
    endpoint_sum = 0.0
    for i in range(len(w) - 1):
        endpoint_sum += w[i] + w[i + 1]
    terminal = w[-1]
    return terminal * terminal * terminal / 3 - h * endpoint_sum / 2


def gbm_left(w, h, x):
    total = 0.0
    for i in range(len(w) - 1):
        total += x[i] * (w[i + 1] - w[i])
    return total


def gbm_milstein(w, h, x):
    total = 0.0
    for i in range(len(w) - 1):
        d = w[i + 1] - w[i]
        total += x[i] * d + 0.5 * x[i] * (d * d - h)
    return total


def gbm_exact_terminal(w, h, x):
    return x[-1] - 1.0


METHODS = {
    "t_left": ("t", t_left),
    "t_optimal": ("t", t_optimal),
    "w_left": ("w", w_left),
    "w_left_qv": ("w", w_left_qv),
    "w_milstein": ("w", w_milstein),
    "w_exact_endpoint": ("w", w_exact_endpoint),
    "w2_left": ("w2", w2_left),
    "w2_milstein": ("w2", w2_milstein),
    "w2_optimal_expanded": ("w2", w2_optimal_expanded),
    "w2_optimal_compressed": ("w2", w2_optimal_compressed),
    "gbm_left": ("gbm", gbm_left),
    "gbm_milstein": ("gbm", gbm_milstein),
    "gbm_exact_terminal": ("gbm", gbm_exact_terminal),
}


def confidence(values):
    """Descriptive t interval from five independent seed-level statistics."""
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    half = T_CRITICAL_DF4 * se
    return {"mean": mean, "standard_error": se, "ci95_approx": [mean - half, mean + half]}


def path_metrics(estimates, targets):
    errors = [a - b for a, b in zip(estimates, targets)]
    mse = statistics.fmean(e * e for e in errors)
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "bias": statistics.fmean(errors),
        "error_variance_unbiased": statistics.variance(errors),
        "estimator_variance_unbiased": statistics.variance(estimates),
    }


def summarize_seed_metrics(rows):
    summary = {key: confidence([row[key] for row in rows]) for key in rows[0] if key != "rmse"}
    mse = summary["mse"]
    summary["rmse"] = {
        "value": math.sqrt(mse["mean"]),
        "ci95_approx_from_mse": [math.sqrt(max(0.0, v)) for v in mse["ci95_approx"]],
    }
    return summary


def generate_dataset(seeds=SEEDS, paths=PATHS):
    """True fine bridge integrals, independent of fine-grid increments.

    A_i = integral fine Brownian bridge dt ~ N(0,h^3/12). A_i are
    mutually independent and independent of all sampled increments.
    Drawing these areas makes integral(W dt) exact in joint distribution,
    not a finest-grid discretization masquerading as ground truth.
    """
    result = {}
    h = T / FINE_N
    for seed in seeds:
        rng = random.Random(seed)
        trajectories = []
        for unused in range(paths):
            w = [0.0]
            integral_terms = []
            for i in range(FINE_N):
                a = w[-1]
                b = a + rng.gauss(0.0, math.sqrt(h))
                area = rng.gauss(0.0, math.sqrt(h * h * h / 12.0))
                integral_terms.append(h * (a + b) / 2.0 + area)
                w.append(b)
            iw = math.fsum(integral_terms)
            terminal = w[-1]
            oracles = {"t": T * terminal - iw,
                       "w": (terminal * terminal - T) / 2,
                       "w2": terminal * terminal * terminal / 3 - iw,
                       "gbm": math.expm1(terminal - T / 2)}
            trajectories.append((tuple(w), oracles))
        result[seed] = trajectories
    return result


def prepare_endpoints(dataset, n):
    stride = FINE_N // n
    return {seed: [fine[::stride] for fine, oracle in paths] for seed, paths in dataset.items()}


def prepare_gbm(endpoints, n):
    h = T / n
    return {seed: [tuple(math.exp(v - i * h / 2) for i, v in enumerate(w)) for w in paths]
            for seed, paths in endpoints.items()}


def analytic_mse(name, n):
    h = T / n
    if name == "t_left":
        return T ** 3 / (3 * n * n)
    if name in ("t_optimal", "w2_optimal_expanded", "w2_optimal_compressed"):
        return T ** 3 / (12 * n * n)
    if name in ("w_left", "w_left_qv"):
        return T * T / (2 * n)
    if name in ("w_milstein", "w_exact_endpoint", "gbm_exact_terminal"):
        return 0.0
    if name == "w2_left":
        return T ** 3 / n
    if name == "w2_milstein":
        return T ** 3 / (n * n)
    if name == "gbm_left":
        return math.fsum(math.exp(i * h) for i in range(n)) * (math.expm1(h) - h)
    if name == "gbm_milstein":
        return math.fsum(math.exp(i * h) for i in range(n)) * (math.expm1(h) - h - h * h / 2)
    raise KeyError(name)


def batch_evaluate(fn, endpoints, gbm_nodes, h, queries=1):
    checksum = 0.0
    for query in range(queries):
        for w, x in zip(endpoints, gbm_nodes):
            checksum += fn(w, h, x)
    return checksum


def sample_summary(samples):
    return {"median": statistics.median(samples), "min": min(samples),
            "max": max(samples), "samples": samples}


def profile_methods(endpoints, gbm_nodes, n):
    """Same frozen 64 paths, alternate forward/reverse orders, no tracer timing."""
    samples = {name: [] for name in METHODS}
    checksums = {}
    names = list(METHODS)
    h = T / n
    for repeat in range(TIMING_REPEATS):
        order = names if repeat % 2 == 0 else list(reversed(names))
        for name in order:
            fn = METHODS[name][1]
            start = time.perf_counter()
            checksums[name] = batch_evaluate(fn, endpoints, gbm_nodes, h, REPEATED_QUERIES)
            samples[name].append(time.perf_counter() - start)
    result = {}
    for name in names:
        # A separate pass measures temporary allocations; input arrays already exist.
        tracemalloc.start()
        checksum = batch_evaluate(METHODS[name][1], endpoints, gbm_nodes, h, REPEATED_QUERIES)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result[name] = {"seconds": sample_summary(samples[name]),
                        "peak_extra_python_bytes": peak,
                        "retained_extra_python_bytes": current,
                        "checksum": checksum}
    expanded = samples["w2_optimal_expanded"]
    compressed = samples["w2_optimal_compressed"]
    ratios = [a / b for a, b in zip(expanded, compressed)]
    return result, sample_summary(ratios)


def profile_setup():
    """Timing subset setup is regenerated separately; no reuse hidden in e2e."""
    generation = []
    endpoints = {n: [] for n in GRIDS}
    gbm = {n: [] for n in GRIDS}
    terminal_gbm = []
    for repeat in range(TIMING_REPEATS):
        start = time.perf_counter()
        dataset = generate_dataset((SEEDS[0],), TIMING_PATHS)
        generation.append(time.perf_counter() - start)
        for n in GRIDS:
            start = time.perf_counter()
            selected = prepare_endpoints(dataset, n)
            endpoints[n].append(time.perf_counter() - start)
            start = time.perf_counter()
            prepare_gbm(selected, n)
            gbm[n].append(time.perf_counter() - start)
        start = time.perf_counter()
        [math.exp(fine[-1] - T / 2) for fine, oracle in dataset[SEEDS[0]]]
        terminal_gbm.append(time.perf_counter() - start)
    # Dataset allocation measured in its own pass, never during timed calculations.
    tracemalloc.start()
    dataset = generate_dataset((SEEDS[0],), TIMING_PATHS)
    generation_current, generation_peak = tracemalloc.get_traced_memory()
    selected = prepare_endpoints(dataset, FINE_N)
    prepared = prepare_gbm(selected, FINE_N)
    all_current, all_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"generation_with_validation_oracles_seconds": sample_summary(generation),
            "endpoint_precompute_seconds": {n: sample_summary(v) for n, v in endpoints.items()},
            "gbm_all_nodes_precompute_seconds": {n: sample_summary(v) for n, v in gbm.items()},
            "gbm_terminal_only_precompute_seconds": sample_summary(terminal_gbm),
            "memory_timing_subset": {"dataset_peak_python_bytes": generation_peak,
                                     "dataset_retained_python_bytes": generation_current,
                                     "dataset_plus_n64_gbm_peak_python_bytes": all_peak,
                                     "dataset_plus_n64_gbm_retained_python_bytes": all_current}}


def memoization_control(w, x, n):
    """Generic identical-query caching benefits either algebraic representation."""
    answer = {}
    for name in ("w2_optimal_expanded", "w2_optimal_compressed"):
        samples = []
        setup_samples = []
        query_samples = []
        for repeat in range(TIMING_REPEATS):
            start = time.perf_counter()
            cache = [METHODS[name][1](a, T / n, b) for a, b in zip(w, x)]
            midpoint = time.perf_counter()
            checksum = 0.0
            for query in range(REPEATED_QUERIES):
                for value in cache:
                    checksum += value
            end = time.perf_counter()
            setup_samples.append(midpoint - start)
            query_samples.append(end - midpoint)
            samples.append(end - start)
        answer[name] = {"precompute_seconds": sample_summary(setup_samples),
                        "cached_query_seconds": sample_summary(query_samples),
                        "precompute_plus_queries_seconds": sample_summary(samples),
                        "cached_values": len(cache), "checksum": checksum}
    return answer


def operation_model(n):
    """Source-level real arithmetic model, not CPU instructions or bit complexity."""
    return {
        "w2_optimal_expanded": {"add_sub": 6 * n, "multiply": 7 * n, "divide": 2 * n,
                                "grid_value_reads_without_cache": 2 * n,
                                "integrand_value_formations": n,
                                "derivative_coefficient_formations": n,
                                "extra_space": "O(1), excluding inputs"},
        "w2_optimal_compressed": {"add_sub": 2 * n + 1, "multiply": 3, "divide": 2,
                                  "grid_value_reads_without_cache": 2 * n + 1,
                                  "integrand_value_formations": 0,
                                  "derivative_coefficient_formations": 0,
                                  "extra_space": "O(1), excluding inputs"},
        "all_methods_integrand_value_formations": {
            "t_left": n, "t_optimal": n, "w_left": n, "w_left_qv": 0,
            "w_milstein": n, "w_exact_endpoint": 0, "w2_left": n,
            "w2_milstein": n, "w2_optimal_expanded": n,
            "w2_optimal_compressed": 0, "gbm_left": n,
            "gbm_milstein": n, "gbm_exact_terminal": 1},
        "gbm_all_nodes_exp_calls_in_shared_setup": n + 1,
        "gbm_exact_terminal_exp_calls_needed_standalone": 1,
        "literal_H_callback_calls": 0,
        "note": "Closed formulas are inlined. Value formations differ from costly callback calls. Python indexing/loop overhead, integer operations, and allocation are not counted as real arithmetic. No asymptotic sublinear claim for arbitrary endpoint data.",
    }


def black_scholes(dataset):
    s, k, r, sigma, maturity = 100.0, 100.0, 0.05, 0.2, 1.0
    exact = prices(s, k, r, sigma, maturity)[0]
    discount = math.exp(-r * maturity)

    def payoff(z):
        return discount * max(s * math.exp((r - sigma * sigma / 2) * maturity + sigma * z) - k, 0.0)

    rows = []
    for seed in SEEDS:
        terminal = [fine[-1] for fine, oracle in dataset[seed]]
        plain = [payoff(z) for z in terminal]
        # 128 independent pair means, 256 payoff calls, same cost in payoff calls.
        pairs = [(payoff(z) + payoff(-z)) / 2 for z in terminal[:PATHS // 2]]
        rows.append({"seed": seed, "plain_price": statistics.fmean(plain),
                     "antithetic_price": statistics.fmean(pairs),
                     "plain_estimator_variance_estimate": statistics.variance(plain) / PATHS,
                     "antithetic_estimator_variance_estimate": statistics.variance(pairs) / len(pairs),
                     "plain_error": statistics.fmean(plain) - exact,
                     "antithetic_error": statistics.fmean(pairs) - exact,
                     "payoff_calls_each_method": PATHS,
                     "independent_units_plain": PATHS,
                     "independent_units_antithetic": len(pairs)})
    timing = {"plain": [], "antithetic": []}
    terminal = [fine[-1] for fine, oracle in dataset[SEEDS[0]]]
    for repeat in range(TIMING_REPEATS):
        order = ("plain", "antithetic") if repeat % 2 == 0 else ("antithetic", "plain")
        for name in order:
            start = time.perf_counter()
            if name == "plain":
                checksum = math.fsum(payoff(z) for z in terminal) / PATHS
            else:
                checksum = math.fsum((payoff(z) + payoff(-z)) / 2 for z in terminal[:PATHS // 2]) / (PATHS // 2)
            timing[name].append(time.perf_counter() - start)
    metrics = {}
    for name in ("plain", "antithetic"):
        errors = [row[name + "_error"] for row in rows]
        metrics[name] = {"price": confidence([row[name + "_price"] for row in rows]),
                         "bias": confidence(errors),
                         "mse_across_seed_price_estimates": confidence([v * v for v in errors]),
                         "mean_within_seed_variance_estimate": confidence([row[name + "_estimator_variance_estimate"] for row in rows]),
                         "seconds_excluding_shared_dataset_generation": sample_summary(timing[name])}
    differences = [row["antithetic_error"] ** 2 - row["plain_error"] ** 2 for row in rows]
    return {"parameters": {"S": s, "K": k, "r": r, "sigma": sigma, "T": maturity},
            "analytical_call": exact, "per_seed": rows, "summary": metrics,
            "paired_squared_error_difference_antithetic_minus_plain": confidence(differences),
            "discretization_error": "None: exact GBM terminal simulation. Only Monte Carlo and floating point errors.",
            "budget": "256 payoff calls per method per seed. Antithetic uses 128 independent normals, plain 256. Shared Brownian generation is reported separately; the antithetic method does not intrinsically need a path grid.",
            "scope": "Known BSM control; neither a change to its formula nor a claim about market prices."}


def dominates(a, b):
    return (a["seconds"] <= b["seconds"] and a["rmse"] <= b["rmse"]
            and (a["seconds"] < b["seconds"] or a["rmse"] < b["rmse"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("benchmark-results.json"))
    args = parser.parse_args()
    all_start = time.perf_counter()
    start = time.perf_counter()
    dataset = generate_dataset()
    full_generation_seconds = time.perf_counter() - start
    setup = profile_setup()
    accuracy = []
    timing = []
    identities = {"w_left_vs_qv": 0.0, "w2_expanded_vs_compressed": 0.0,
                  "w_milstein_vs_exact": 0.0, "gbm_terminal_vs_oracle": 0.0}
    identity_scaled = dict(identities)
    exact_oracle_max = 0.0
    deterministic_gates = []
    statistical_checks = []
    for n in GRIDS:
        endpoints = prepare_endpoints(dataset, n)
        gbm = prepare_gbm(endpoints, n)
        per_method = {name: [] for name in METHODS}
        for seed in SEEDS:
            estimates = {name: [] for name in METHODS}
            targets = {case: [oracle[case] for fine, oracle in dataset[seed]] for case in ("t", "w", "w2", "gbm")}
            for w, x in zip(endpoints[seed], gbm[seed]):
                values = {name: fn(w, T / n, x) for name, (case, fn) in METHODS.items()}
                for name, value in values.items():
                    estimates[name].append(value)
                for label, a, b in (
                    ("w_left_vs_qv", values["w_left"], values["w_left_qv"]),
                    ("w2_expanded_vs_compressed", values["w2_optimal_expanded"], values["w2_optimal_compressed"]),
                    ("w_milstein_vs_exact", values["w_milstein"], values["w_exact_endpoint"])):
                    identities[label] = max(identities[label], abs(a - b))
                    identity_scaled[label] = max(identity_scaled[label], abs(a - b) / max(1.0, abs(a), abs(b)))
            for name, (case, fn) in METHODS.items():
                metrics = path_metrics(estimates[name], targets[case])
                per_method[name].append(metrics)
                if analytic_mse(name, n) == 0.0:
                    exact_oracle_max = max(exact_oracle_max, max(abs(a - b) / max(1.0, abs(a), abs(b)) for a, b in zip(estimates[name], targets[case])))
            diff = [abs(a - b) for a, b in zip(estimates["gbm_exact_terminal"], targets["gbm"])]
            identities["gbm_terminal_vs_oracle"] = max(identities["gbm_terminal_vs_oracle"], max(diff))
            identity_scaled["gbm_terminal_vs_oracle"] = max(identity_scaled["gbm_terminal_vs_oracle"], max(abs(a - b) / max(1.0, abs(a), abs(b)) for a, b in zip(estimates["gbm_exact_terminal"], targets["gbm"])))
        profiled, ratios = profile_methods(endpoints[SEEDS[0]][:TIMING_PATHS], gbm[SEEDS[0]][:TIMING_PATHS], n)
        for name, rows in per_method.items():
            summary = summarize_seed_metrics(rows)
            expected = analytic_mse(name, n)
            interval = summary["mse"]["ci95_approx"]
            check = "exact_identity_tolerance" if expected == 0 else ("inside_approx_seed_interval" if interval[0] <= expected <= interval[1] else "outside_approx_seed_interval_NOT_code_failure")
            accuracy.append({"n": n, "method": name, "case": METHODS[name][0],
                             "per_seed": [{"seed": seed, **row} for seed, row in zip(SEEDS, rows)],
                             "summary": summary, "analytic_mse": expected,
                             "statistical_diagnostic": check})
            statistical_checks.append({"n": n, "method": name, "analytic_mse": expected, "diagnostic": check})
        for name, row in profiled.items():
            generation = setup["generation_with_validation_oracles_seconds"]["samples"]
            endpoint_setup = setup["endpoint_precompute_seconds"][n]["samples"]
            if name == "gbm_exact_terminal":
                x_setup = setup["gbm_terminal_only_precompute_seconds"]["samples"]
            elif name.startswith("gbm_"):
                x_setup = setup["gbm_all_nodes_precompute_seconds"][n]["samples"]
            else:
                x_setup = [0.0] * TIMING_REPEATS
            kernel = row["seconds"]["samples"]
            row["generation_setup_plus_queries_seconds_accounted"] = sample_summary([a + b + c + d for a, b, c, d in zip(generation, endpoint_setup, x_setup, kernel)])
            row["endpoint_setup_plus_queries_seconds_accounted"] = sample_summary([a + b + c for a, b, c in zip(endpoint_setup, x_setup, kernel)])
        timing.append({"n": n, "paths": TIMING_PATHS, "queries_per_path": REPEATED_QUERIES,
                       "methods": profiled, "paired_expanded_over_compressed_time_ratio": ratios,
                       "memoization_control": memoization_control(endpoints[SEEDS[0]][:TIMING_PATHS], gbm[SEEDS[0]][:TIMING_PATHS], n),
                       "operation_model": operation_model(n)})
    tolerance = 2e-12
    deterministic_gates = [{"name": name, "max_absolute_difference": identities[name],
                            "max_scaled_difference": value, "scaled_tolerance": tolerance,
                            "pass": value <= tolerance} for name, value in identity_scaled.items()]
    deterministic_gates.append({"name": "all_exact_methods_vs_oracles", "max_scaled_difference": exact_oracle_max,
                                "scaled_tolerance": tolerance, "pass": exact_oracle_max <= tolerance})
    # An empirical frontier is descriptive; timing and MSE uncertainty are not ignored in claims.
    frontier = {}
    for case in ("t", "w", "w2", "gbm"):
        points = []
        for row in accuracy:
            if row["case"] == case:
                trow = next(v for v in timing if v["n"] == row["n"])["methods"][row["method"]]
                points.append({"method": row["method"], "n": row["n"],
                               "rmse": row["summary"]["rmse"]["value"],
                               "seconds": trow["endpoint_setup_plus_queries_seconds_accounted"]["median"]})
        frontier[case] = {"all_measured_points": points,
                          "nondominated_observed_points": [p for p in points if not any(dominates(q, p) for q in points)],
                          "scope": "Same 64-path, 3-query timing budget with grid setup; RMSE from all 1280 paths. No interpolation, extrapolation, tuning, or rigorous frontier claim."}
    result = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(gate["pass"] for gate in deterministic_gates) else "FAIL",
        "config": {"seeds": SEEDS, "paths_per_seed": PATHS, "fine_n": FINE_N, "grids": GRIDS,
                   "T": T, "timing_paths": TIMING_PATHS, "timing_repeats": TIMING_REPEATS,
                   "repeated_queries": REPEATED_QUERIES, "t_critical_df4": T_CRITICAL_DF4,
                   "processes": 1, "numerical_threads": 1},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "machine": platform.machine(), "timer": "time.perf_counter"},
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (Path(__file__).resolve(), ROOT / "verify.py")},
        "full_dataset_generation_seconds": full_generation_seconds,
        "setup_profiles": setup,
        "accuracy": accuracy,
        "timing": timing,
        "deterministic_gates": deterministic_gates,
        "statistical_checks": statistical_checks,
        "empirical_time_error_frontiers": frontier,
        "black_scholes_control": black_scholes(dataset),
        "limitations": [
            "Five independent seed means give approximate descriptive t intervals, not finite-sample coverage guarantees. Multiple diagnostics have no multiplicity correction.",
            "Statistical misses of analytic MSE are not deterministic test failures. GBM and polynomial tails make tiny samples unstable.",
            "Timing uses a small frozen subset, five alternating repetitions, and no thermal control. Ratios are local observations, not universal speed claims.",
            "Accounted end-to-end times sum independently measured generation, precompute and kernel samples; they are not a single contiguous wall-clock measurement. Generation includes fine bridge areas used only for validation.",
            "tracemalloc reports Python allocations, not RSS or allocator/OS memory. Query peaks exclude pre-existing endpoint inputs.",
            "The GBM example integrates exact node values of an SDE solution; it does not measure error from numerically propagating that SDE.",
            "Both W² candidate formulas compute the same known conditional expectation; compression is constant-factor algebra, not a new integral or sublinear algorithm.",
            "Caching an identical query works for every deterministic estimator and is explicitly included as a generic control.",
            "Floating point algebra identities are checked at scaled tolerance. This is not a stability proof on adversarial or extreme inputs.",
        ],
    }
    result["total_elapsed_seconds_before_json_write"] = time.perf_counter() - all_start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "elapsed_seconds": result["total_elapsed_seconds_before_json_write"],
                      "accuracy_rows": len(accuracy), "deterministic_gates": len(deterministic_gates),
                      "statistical_interval_misses": sum("outside" in row["diagnostic"] for row in statistical_checks)}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
