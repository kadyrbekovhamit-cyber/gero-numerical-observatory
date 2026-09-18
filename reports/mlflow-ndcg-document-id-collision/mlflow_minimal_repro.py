import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MLFLOW_ENABLE_TELEMETRY"] = "false"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import pandas as pd
from mlflow.metrics import ndcg_at_k

metric = ndcg_at_k(3)
for targets, predictions in [
    (["a_bc574ae_2"], ["a", "a", "z"]),
    (["b"], ["a", "a", "z"]),
    (["a"], ["a", "a_bc574ae_2", "a"]),
    (["a"], ["a", "b", "a"]),
]:
    print(targets, predictions, metric.eval_fn(pd.Series([predictions]), pd.Series([targets])).scores)
