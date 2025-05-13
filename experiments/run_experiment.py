import time, uuid, json
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import ExperimentSpec, ModelParams
from src.data.simulate import simulate_series
from src.filters.kalman import KalmanFilter
from src.filters.robust_kf import RobustHuberKF
from src.filters.student_t_kf import StudentTFilter
from src.filters.particle_filter import BootstrapParticleFilter
from src.metrics.scoring import summary
from src.utils.plotting import plot_simulation

FILTERS = {
    "KF": KalmanFilter(),
    "Huber": RobustHuberKF(),
    "T-KF": StudentTFilter(),
    "PF": BootstrapParticleFilter(N=2000),
}


def run(spec: ExperimentSpec) -> None:
    spec.make_dirs()
    rows = []
    base_rng = np.random.SeedSequence(spec.seed)

    for rep in range(spec.n_rep):
        seed = int(base_rng.spawn(1)[0].entropy)
        mu, y = simulate_series(spec.model, spec.T, seed=seed)

        mu_hat_dict = {}
        for name, flt in FILTERS.items():
            if name == "T-KF" and spec.model.eps_dist != "student_t":
                continue
            mu_hat = flt.filter(y, spec.model).mu_hat
            mu_hat_dict[name] = mu_hat
            rows.append(
                dict(rep=rep, filter=name, **spec.model.__dict__, **summary(mu, mu_hat))
            )

        if rep == 0:
            uid = uuid.uuid4().hex[:6]
            plot_simulation(
                y,
                mu,
                mu_hat_dict,
                out=spec.plots_dir / f"sim_{uid}.png",
                title=f"{spec.model.eps_dist}/{spec.model.eta_dist}, rep0",
            )

    df = pd.DataFrame(rows)
    fn = time.strftime("metrics_%Y%m%d-%H%M%S.parquet")
    df.to_parquet(spec.results_dir / fn, index=False)
    print(f"Saved {df.shape[0]} rows → {fn}")


if __name__ == "__main__":
    spec = ExperimentSpec(
        model=ModelParams(
            phi=0.9,
            sigma_eps=1.0,
            sigma_eta=0.3,
            eps_dist="student_t",
            nu=5,
            eta_dist="beta_sym",
            alpha=2.0,
        ),
        T=200,
        n_rep=10,
    )
    run(spec)
