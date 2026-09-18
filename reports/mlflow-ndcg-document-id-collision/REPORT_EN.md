# MLflow NDCG can score an all-irrelevant retrieval above zero

In a synthetic local test of **MLflow 3.16.1**, `mlflow.metrics.ndcg_at_k(3)` returned **0.6309297535714574** for targets `["a_bc574ae_2"]` and predictions `["a", "a", "z"]`. Every retrieved original ID is irrelevant, so NDCG@3 should be **0**. Renaming the unseen target to `"b"` restores zero without changing any relevance relationship.

The finding was reported through [MLflow issue #25965](https://github.com/mlflow/mlflow/issues/25965). The correction described below is a **local candidate**; no upstream acceptance is claimed.

## Reproduce the result

The tested environment used official `mlflow-skinny==3.16.1`, Python 3.12.14, pandas 3.0.6, NumPy 2.5.3 and scikit-learn 1.9.1.

```python
import pandas as pd
from mlflow.metrics import ndcg_at_k

metric = ndcg_at_k(3)
for targets in [["a_bc574ae_2"], ["b"]]:
    result = metric.eval_fn(
        pd.Series([["a", "a", "z"]]),
        pd.Series([targets]),
    )
    print(result.scores[0])
```

Observed output:

```text
0.6309297535714574
0.0
```

Both expected scores are zero: all retrieved positions have relevance zero and the ideal ranking has positive gain.

## Why the identifier matters

[The metric contract](https://github.com/mlflow/mlflow/blob/35758a9a954422200034b949176879c2cbbe4a8a/mlflow/metrics/__init__.py#L300) defines binary relevance and treats repeated relevant IDs as distinct occurrences. Its [duplicate-expansion helper](https://github.com/mlflow/mlflow/blob/35758a9a954422200034b949176879c2cbbe4a8a/mlflow/metrics/metric_definitions.py#L409) creates a name such as `a_bc574ae_2` for the second `a`. A generated name can equal a real document ID and change the subsequent relevance/index mapping.

A second example demonstrates the same root cause. With target `["a"]`, predictions `["a", "a_bc574ae_2", "a"]` score **1.0**, whereas `["a", "b", "a"]` score **0.9197207891481877**. Both have positional relevance `[1,0,1]`. Under this metric's duplicate policy, the independent expected value is:

```text
(1 + 1/log2(4)) / (1 + 1/log2(3)) ≈ 0.9197207891481876
```

The local candidate identifies each occurrence by `(original_id, occurrence_number)` and checks relevance against an immutable original target set.

## What was verified

The official release and a separately loaded complete module from commit `35758a9a954422200034b949176879c2cbbe4a8a` were executed using real dependencies. That pinned module is byte-identical to the installed release module; a complete current-master checkout was not installed.

Across **1,810 synthetic scenarios**, including 905 renaming pairs, the failure counts were:

| Tested implementation | Failing scenarios |
| --- | ---: |
| Official release / pinned current module | 214 / 214 |
| Local candidate correction | 0 |
| Original implementation restored | 214 |

All **1,596 previously passing scenarios remained exactly unchanged**. A fresh process with `PYTHONHASHSEED=17` reproduced all four observation files byte-for-byte. The independent oracle calculates positional DCG directly, without scikit-learn or generated string identifiers.

One unchanged upstream `test_ndcg_at_k` function passed with original, candidate and restored helpers. The full pytest module and MLflow test suite were not run.

## Scope and prior reports

This API has been **deprecated since 3.4.0**, but remains available in the tested 3.16.1 release and inspected source. The work did not test newer GenAI scorers, hosted Databricks services, real retrieval models or customer impact. No model inference was needed.

The bounded duplicate review distinguished earlier irrelevant-duplicate issue #12361 / PR #12447 and short-retrieval issue #24541 and associated proposals. The main nonempty tests here use `k <= len(predictions)` to isolate the ID collision. No exact collision report was identified in that review; this is not a claim that nobody previously knew it.

These are **214 failing test scenarios for one root cause**, not 214 separate bugs. Zero failures in the candidate's finite test set does not establish correctness for every input or production performance.

Research and report preparation were AI-assisted. The reported outputs came from the actual MLflow implementation. Reproduction, observations, candidate patch and receipts are retained with the evidence package.
