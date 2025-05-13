from dataclasses import dataclass
from pathlib import Path
from typing import Literal

EPS_DIST = Literal["gaussian", "student_t"]
ETA_DIST = Literal["uniform", "beta_sym"]


@dataclass(frozen=True, slots=True)
class ModelParams:
    phi: float
    sigma_eps: float
    sigma_eta: float
    eps_dist: EPS_DIST = "gaussian"
    eta_dist: ETA_DIST = "uniform"
    nu: int | None = None       # Student-t df
    alpha: float | None = None  # symmetric-Beta shape


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    model: ModelParams
    T: int = 200
    n_rep: int = 500
    seed: int = 1
    results_dir: Path = Path("results/metrics")
    plots_dir: Path = Path("results/figures")

    def make_dirs(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
