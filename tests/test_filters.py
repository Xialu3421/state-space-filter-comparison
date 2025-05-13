import numpy as np
import pytest
from src.data.simulate import simulate_series
from src.filters.kalman import KalmanFilter
from src.filters.robust_kf import RobustHuberKF
from src.filters.student_t_kf import StudentTFilter
from src.filters.particle_filter import BootstrapParticleFilter
from src.metrics.scoring import mse

KF = KalmanFilter()
RKF = RobustHuberKF()
TKF = StudentTFilter()
PF  = BootstrapParticleFilter(N=500, resample_threshold=0.3)

@pytest.mark.parametrize("T", [100])
def test_kf_recovers_state(gaussian_params, T):
    mu, y = simulate_series(gaussian_params, T=T, seed=2)
    mu_hat = KF.filter(y, gaussian_params).mu_hat
    assert mse(mu, mu_hat) < 0.5   # loose but useful sanity bound

def test_student_t_vs_kf_converges(gaussian_params):
    """When ν→∞ the Student-t filter ≈ Gaussian KF."""
    params = gaussian_params | {"eps_dist":"student_t", "nu":1_000_000}
    mu, y = simulate_series(params, T=200, seed=3)
    mu_kf = KF.filter(y, params).mu_hat
    mu_t  = TKF.filter(y, params).mu_hat
    assert np.allclose(mu_kf, mu_t, atol=1e-2)

def test_robust_downweights_outlier(student_params):
    """Massive outlier should affect Robust KF < classic KF."""
    mu, y = simulate_series(student_params, T=50, seed=4)
    y[20] += 30  # big spike
    err_kf  = abs(KF.filter(y, student_params).mu_hat[20] - mu[20])
    err_rkf = abs(RKF.filter(y, student_params).mu_hat[20] - mu[20])
    assert err_rkf < err_kf

def test_particle_filter_runs(gaussian_params):
    mu, y = simulate_series(gaussian_params, T=60, seed=5)
    mu_hat = PF.filter(y, gaussian_params).mu_hat
    assert mu_hat.shape == (60,)
