import numpy as np
from scipy import stats
from .base import StateSpaceFilter, FilterResult
from ..config import ModelParams
from ..data.simulate import draw_eta


class BootstrapParticleFilter(StateSpaceFilter):
    def __init__(self, N: int = 1000, resample_threshold: float = 0.5):
        self.N = N
        self.thresh = resample_threshold

    def filter(self, y: np.ndarray, params: ModelParams) -> FilterResult:
        T = y.size
        rng = np.random.default_rng()
        particles = rng.normal(0.0, params.sigma_eta / np.sqrt(1 - params.phi**2), self.N)
        weights = np.full(self.N, 1 / self.N)
        mu_hat = np.empty(T)

        for t in range(T):
            eta = draw_eta(rng, params, self.N)
            particles = params.phi * particles + eta

            ll = stats.norm.logpdf(y[t] - particles, scale=params.sigma_eps)
            weights *= np.exp(ll - ll.max())
            weights /= weights.sum()

            mu_hat[t] = np.dot(weights, particles)

            ess = 1.0 / (weights ** 2).sum()
            if ess < self.thresh * self.N:
                idx = rng.choice(self.N, self.N, p=weights)
                particles = particles[idx]
                weights.fill(1 / self.N)

        return FilterResult(mu_hat=mu_hat, V_hat=None)
