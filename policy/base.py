from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Policy(ABC):
    @abstractmethod
    def initialize(self, model, data) -> None:
        """Prepare the policy for a specific MuJoCo model."""
        raise NotImplementedError

    @abstractmethod
    def compute_control(self, state: np.ndarray) -> np.ndarray:
        """Return the control input vector for the current state."""
        raise NotImplementedError
