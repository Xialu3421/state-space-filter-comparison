import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from src.config import ModelParams

@pytest.fixture(scope="session")
def gaussian_params():
    return ModelParams(phi=0.8, sigma_eps=1.0, sigma_eta=0.5)

@pytest.fixture(scope="session")
def student_params():
    return ModelParams(
        phi=0.8,
        sigma_eps=1.0,
        sigma_eta=0.5,
        eps_dist="student_t",
        nu=5,
    )

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(123)
