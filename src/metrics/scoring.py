import numpy as np
import pandas as pd
from typing import Dict


def mse(mu: np.ndarray, mu_hat: np.ndarray) -> float:
    return float(np.mean((mu - mu_hat) ** 2))


def medae(mu: np.ndarray, mu_hat: np.ndarray) -> float:
    return float(np.median(np.abs(mu - mu_hat)))


def summary(mu: np.ndarray, mu_hat: np.ndarray) -> Dict[str, float]:
    return {"MSE": mse(mu, mu_hat), "MedAE": medae(mu, mu_hat)}


def aggregate_metrics(csv_paths: list[str | pd.DataFrame]) -> pd.DataFrame:
    if not csv_paths:
        raise ValueError("no inputs")
    dfs = [
        pd.read_parquet(p) if isinstance(p, str) else p.copy()
        for p in csv_paths
    ]
    return pd.concat(dfs, ignore_index=True)
