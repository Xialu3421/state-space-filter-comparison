import numpy as np
from numpy.random import Generator, default_rng
from typing import Tuple
from pathlib import Path
import pandas as pd

from ..config import ModelParams


def draw_eps(rng: Generator, params: ModelParams, size: int) -> np.ndarray:
    if params.eps_dist == "gaussian":
        return rng.normal(0.0, params.sigma_eps, size)
    if params.eps_dist == "student_t":
        return params.sigma_eps * rng.standard_t(params.nu, size=size)
    raise ValueError(params.eps_dist)


def draw_eta(rng: Generator, params: ModelParams, size: int) -> np.ndarray:
    l = params.sigma_eta
    if params.eta_dist == "uniform":
        return rng.uniform(-l, l, size)
    if params.eta_dist == "beta_sym":
        a = params.alpha
        raw = rng.beta(a, a, size)
        return l * (2.0 * raw - 1.0)
    raise ValueError(params.eta_dist)


def simulate_series(
    params: ModelParams, T: int, seed: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    rng = default_rng(seed)
    mu = np.empty(T)
    y = np.empty(T)

    var0 = params.sigma_eta**2 / (1 - params.phi**2)
    mu[0] = rng.normal(0.0, np.sqrt(var0))

    eps = draw_eps(rng, params, T)
    eta = draw_eta(rng, params, T)

    y[0] = mu[0] + eps[0]
    for t in range(1, T):
        mu[t] = params.phi * mu[t - 1] + eta[t]
        y[t] = mu[t] + eps[t]

    return mu, y


def save_simulation(mu: np.ndarray, y: np.ndarray, out: Path) -> None:
    df = pd.DataFrame({"mu": mu, "y": y})
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
