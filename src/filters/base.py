from abc import ABC, abstractmethod
import numpy as np
from typing import NamedTuple

class FilterResult(NamedTuple):
    mu_hat: np.ndarray          # state estimates μ̂_t|t
    V_hat: np.ndarray | None    # state variances (optional)

class StateSpaceFilter(ABC):
    """Base class – every filter gets (y, params) and returns FilterResult."""
    @abstractmethod
    def filter(self, y: np.ndarray, params) -> FilterResult: ...
