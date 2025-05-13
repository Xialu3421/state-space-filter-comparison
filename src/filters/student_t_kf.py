import numpy as np
from typing import Final
from .base import StateSpaceFilter, FilterResult
from ..config import ModelParams


class StudentTFilter(StateSpaceFilter):
    def filter(self, y: np.ndarray, params: ModelParams) -> FilterResult:
        if params.eps_dist != "student_t":
            raise ValueError("eps_dist must be 'student_t'")
        if params.nu is None or params.nu <= 2:
            raise ValueError("nu must be > 2")

        T: Final[int] = y.size
        mu_f = np.empty(T)
        V_f = np.empty(T)

        V_prev = params.sigma_eta**2 / (1 - params.phi**2)
        mu_prev = 0.0
        var_eps = params.sigma_eps**2
        nu = float(params.nu)

        for t in range(T):
            mu_pred = params.phi * mu_prev
            V_pred = params.phi**2 * V_prev + params.sigma_eta**2

            e = y[t] - mu_pred
            S = V_pred + var_eps
            r2 = (e * e) / S
            w = (nu + 1.0) / (nu + r2)

            K = V_pred / S
            mu_curr = mu_pred + w * K * e
            V_curr = w * (1.0 - K) * V_pred

            mu_f[t] = mu_curr
            V_f[t] = V_curr
            mu_prev, V_prev = mu_curr, V_curr

        return FilterResult(mu_hat=mu_f, V_hat=V_f)
