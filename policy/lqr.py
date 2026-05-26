from __future__ import annotations

import numpy as np

from .base import Policy
from .utils import compute_lqr_gain_at_reference, wrap_angle_error


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
            error_state = np.array(state, dtype=float, copy=True)
            reference_state = np.zeros_like(state)
        else:
            error_state = np.array(state - self.reference_state, dtype=float, copy=True)
            reference_state = self.reference_state

        # The MuJoCo state uses angular generalized coordinates in qpos[1:3].
        # Wrap those error terms so states near 2*pi are treated as near upright
        # rather than catastrophically far from the reference.
        if error_state.shape[0] >= 3:
            error_state[1] = wrap_angle_error(state[1], reference_state[1])
            error_state[2] = wrap_angle_error(state[2], reference_state[2])

        u = -self.K @ error_state
        if self.control_limit is not None:
            u = np.clip(u, -self.control_limit, self.control_limit)
        return u
