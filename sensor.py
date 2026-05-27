from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorModel:
    """Simple measurement model for injecting state noise into rollouts."""

    noise_std: np.ndarray | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.noise_std is None:
            self.noise_std = np.zeros(6, dtype=float)
        else:
            self.noise_std = np.asarray(self.noise_std, dtype=float).reshape(-1)
        self.rng = np.random.default_rng(self.seed)

    def measure(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        if self.noise_std is None or np.allclose(self.noise_std, 0.0):
            return state.copy()
        noise = self.rng.normal(loc=0.0, scale=self.noise_std, size=state.shape)
        return state + noise
