import numpy as np
from typing import Final
from .base import StateSpaceFilter, FilterResult
from ..config import ModelParams

class KalmanFilter(StateSpaceFilter):
    def filter(self, y: np.ndarray, params: ModelParams) -> FilterResult:
        T: Final[int] = y.size
        mu_f = np.empty(T)
        V_f  = np.empty(T)
        # initial guesses
        V_prev = params.sigma_eta**2 / (1 - params.phi**2)
        mu_prev = 0.0
        for t in range(T):
            # prediction
            mu_pred = params.phi * mu_prev
            V_pred  = params.phi**2 * V_prev + params.sigma_eta**2
            # update
            S = V_pred + params.sigma_eps**2
            K = V_pred / S                                # Kalman gain
            mu_curr = mu_pred + K * (y[t] - mu_pred)
            V_curr  = (1 - K) * V_pred
            mu_f[t] = mu_curr
            V_f[t]  = V_curr
            mu_prev, V_prev = mu_curr, V_curr
        return FilterResult(mu_hat=mu_f, V_hat=V_f)
