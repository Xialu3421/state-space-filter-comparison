import numpy as np
from src.data.simulate import simulate_series

def test_simulation_shapes(gaussian_params):
    mu, y = simulate_series(gaussian_params, T=250, seed=0)
    assert mu.shape == (250,)
    assert y.shape == (250,)

def test_stationary_variance(gaussian_params):
    # Large T → empirical var of μ_t should be close to sigma_eta^2 / (1-phi^2)
    T = 10_000
    mu, _ = simulate_series(gaussian_params, T=T, seed=1)
    var_theory = gaussian_params.sigma_eta**2 / (1 - gaussian_params.phi**2)
    assert np.isclose(mu.var(), var_theory, rtol=0.1)
