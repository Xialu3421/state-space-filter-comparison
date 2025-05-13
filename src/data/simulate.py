import numpy as np
from numpy.random import Generator, default_rng
from typing import Tuple
from pathlib import Path
import pandas as pd
from ..config import ModelParams

def draw_eps(rng: Generator, params: ModelParams, size: int) -> np.ndarray:
    match params.eps_dist:
        case "gaussian":
            return rng.normal(0.0, params.sigma_eps, size)
        case "student_t":
            return params.sigma_eps * rng.standard_t(df=params.nu, size=size)
        case _:
            raise ValueError(f"Unknown eps_dist {params.eps_dist}")

def draw_eta(rng: Generator, params: ModelParams, size: int) -> np.ndarray:
    ℓ = params.sigma_eta
    match params.eta_dist:
        case "uniform":
            return rng.uniform(-ℓ, ℓ, size)
        case "beta_sym":
            α = params.alpha
            raw = rng.beta(α, α, size)
            return ℓ * (2.0 * raw - 1.0)  # rescale to [-ℓ, ℓ]
        case _:
            raise ValueError(f"Unknown eta_dist {params.eta_dist}")

def simulate_series(params: ModelParams, T: int, seed: int | None = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate one path (μ_t, y_t)_{t=1..T}.
    Returns
    -------
    mu : (T,) latent state
    y  : (T,) observation
    """
    rng = default_rng(seed)
    mu = np.empty(T)
    y = np.empty(T)
    # variance of stationary μ if φ≠0 else use sigma_eta
    var0 = params.sigma_eta**2 / (1 - params.phi**2) if abs(params.phi) < 1 else params.sigma_eta**2
    mu[0] = rng.normal(0.0, np.sqrt(var0))
    eps = draw_eps(rng, params, T)
    eta = draw_eta(rng, params, T)
    y[0] = mu[0] + eps[0]
    for t in range(1, T):
        mu[t] = params.phi * mu[t-1] + eta[t]
        y[t] = mu[t] + eps[t]
    return mu, y

def save_simulation(mu: np.ndarray, y: np.ndarray, out: Path) -> None:
    """Store a single simulated path as parquet for reproducibility."""
    df = pd.DataFrame({"mu": mu, "y": y})
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
