from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

EPS_DIST = Literal["gaussian", "student_t"]
ETA_DIST = Literal["uniform", "beta_sym"]

@dataclass(frozen=True, slots=True)
class ModelParams:
    """Static parameters of the state-space model."""
    phi: float                  # |phi| < 1
    sigma_eps: float            # scale of ε_t
    sigma_eta: float            # scale of η_t (for uniform/beta this is ℓ)
    eps_dist: EPS_DIST = "gaussian"
    eta_dist: ETA_DIST = "uniform"
    nu: int | None = None       # ν for Student-t (None → Gaussian)
    alpha: float | None = None  # α for symmetric Beta (None → Uniform)

@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """One Monte-Carlo experiment setting."""
    model: ModelParams
    T: int = 200                # series length
    n_rep: int = 500            # Monte-Carlo replications
    seed: int = 1               # base RNG seed
    results_dir: Path = Path("results/metrics")
    plots_dir: Path = Path("results/figures")
    
    def make_dirs(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
