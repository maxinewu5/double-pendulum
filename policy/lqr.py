from __future__ import annotations

import numpy as np

from .base import Policy
from .utils import compute_lqr_gain_at_reference


class LQRPolicy(Policy):
    def __init__(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        control_limit: float | None = None,
        reference_state: np.ndarray | None = None,
    ):
        self.Q = Q
        self.R = R
        self.K = None
        self.control_limit = control_limit
        self.reference_state = reference_state

    def initialize(self, model, data) -> None:
        self.K = compute_lqr_gain_at_reference(
            model=model,
            data=data,
            Q=self.Q,
            R=self.R,
            reference_state=self.reference_state,
        )

    def compute_control(self, state: np.ndarray) -> np.ndarray:
        if self.K is None:
            raise RuntimeError("Policy must be initialized before use.")

        if self.reference_state is None:
            error_state = state
        else:
            error_state = state - self.reference_state

        u = -self.K @ error_state
        if self.control_limit is not None:
            u = np.clip(u, -self.control_limit, self.control_limit)
        return u
