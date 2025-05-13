from abc import ABC, abstractmethod
import numpy as np
from typing import NamedTuple


class FilterResult(NamedTuple):
    mu_hat: np.ndarray
    V_hat: np.ndarray | None


class StateSpaceFilter(ABC):
    @abstractmethod
    def filter(self, y: np.ndarray, params) -> FilterResult: ...
