import numpy as np
from typing import Final
from .base import StateSpaceFilter, FilterResult
from ..config import ModelParams


class RobustHuberKF(StateSpaceFilter):
    def __init__(self, c: float = 1.345):
        self.c = float(c)

    @staticmethod
    def _psi(r: np.ndarray, c: float) -> np.ndarray:
        return np.clip(r, -c, c)

    def filter(self, y: np.ndarray, params: ModelParams) -> FilterResult:
        T: Final[int] = y.size
        mu_f = np.empty(T)
        V_f = np.empty(T)

        V_prev = params.sigma_eta**2 / (1 - params.phi**2)
        mu_prev = 0.0
        var_eps = params.sigma_eps**2
        c = self.c

        for t in range(T):
            mu_pred = params.phi * mu_prev
            V_pred = params.phi**2 * V_prev + params.sigma_eta**2

            S = V_pred + var_eps
            std = np.sqrt(S)
            r = (y[t] - mu_pred) / std
            psi = self._psi(r, c)

            K = V_pred / S
            mu_curr = mu_pred + K * std * psi

            psi_prime = 1.0 if abs(r) <= c else 0.0
            A = 1.0 - K * psi_prime
            V_curr = A * V_pred * A + (K * std) ** 2 * (psi ** 2)

            mu_f[t] = mu_curr
            V_f[t] = V_curr
            mu_prev, V_prev = mu_curr, V_curr

        return FilterResult(mu_hat=mu_f, V_hat=V_f)
