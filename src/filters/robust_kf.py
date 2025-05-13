import numpy as np
from typing import Final
from .base import StateSpaceFilter, FilterResult
from ..config import ModelParams


class RobustHuberKF(StateSpaceFilter):
    """
    Robust Kalman filter using the Masreliez–Martin correction with a Huber
    influence function.  When the standardised innovation |r_t| ≤ c the filter
    reduces to the classical Gaussian KF; otherwise the update is damped.

    References
    ----------
    * Masreliez, C. (1975) "Approximate non‐Gaussian filtering."
    * Martin, R., Thompson, W. & Robertson, T. (1983) "Robust Kalman Filtering."
    """

    def __init__(self, c: float = 1.345):
        """
        Parameters
        ----------
        c : float, default 1.345
            Huber tuning constant (≈ 95 % efficiency for Gaussian noise).
        """
        self.c = float(c)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _huber(r: float | np.ndarray, c: float) -> float | np.ndarray:
        """Huber ψ(r) = clip(r, −c, c)."""
        return np.clip(r, -c, c)

    # ------------------------------------------------------------------ filter
    def filter(self, y: np.ndarray, params: ModelParams) -> FilterResult:
        T: Final[int] = y.size
        mu_f = np.empty(T)
        V_f  = np.empty(T)

        # Initial state moments (stationary variance for |φ|<1)
        V_prev = params.sigma_eta**2 / (1 - params.phi**2)
        mu_prev = 0.0

        var_eps = params.sigma_eps**2
        c = self.c

        for t in range(T):
            # 1. Prediction step
            mu_pred = params.phi * mu_prev
            V_pred  = params.phi**2 * V_prev + params.sigma_eta**2

            # 2. Innovation and robust score
            S   = V_pred + var_eps                  # innovation variance
            std = np.sqrt(S)
            r   = (y[t] - mu_pred) / std            # standardised residual
            psi = self._huber(r, c)

            # 3. Robust correction (Masreliez–Martin)
            K = V_pred / S                          # classical Kalman gain
            mu_curr = mu_pred + K * (std * psi)     # state update

            # 4. Posterior variance adjustment
            #     For Huber: ψ'(r)=1 if |r|≤c else 0
            psi_deriv = 1.0 if abs(r) <= c else 0.0
            A = 1.0 - K * psi_deriv
            V_curr = A * V_pred * A + (K * std)**2 * (psi**2)

            mu_f[t] = mu_curr
            V_f[t]  = V_curr
            mu_prev, V_prev = mu_curr, V_curr

        return FilterResult(mu_hat=mu_f, V_hat=V_f)
