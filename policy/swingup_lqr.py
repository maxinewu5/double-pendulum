from __future__ import annotations

import numpy as np

from .base import Policy
from .lqr import LQRPolicy
from .swingup import SwingUpPolicy
from .utils import interpret_planar_state, wrap_angle_error


class SwingUpLQRPolicy(Policy):
    """
    Wrapper policy that uses shuttle-style swing-up first, then hands off to LQR
    once the system enters a configurable capture region.
    """

    def __init__(
        self,
        swingup_policy: SwingUpPolicy,
        lqr_policy: LQRPolicy,
        q1_capture_deg: float = 5.0,
        q2_capture_deg: float = 5.0,
        q1_dot_capture: float = 1.0,
        q2_abs_dot_capture: float = 1.5,
        x_capture_limit: float = 2.0,
        h_lim: int = 0,
    ):
        self.swingup_policy = swingup_policy
        self.lqr_policy = lqr_policy
        self.q1_capture_rad = np.deg2rad(q1_capture_deg)
        self.q2_capture_rad = np.deg2rad(q2_capture_deg)
        self.q1_dot_capture = q1_dot_capture
        self.q2_abs_dot_capture = q2_abs_dot_capture
        self.x_capture_limit = x_capture_limit
        self.h_lim = max(0, int(h_lim))

        self.active_policy = "swingup"
        self.switch_step: int | None = None
        self.debug_history: list[dict[str, float | str | bool]] = []
        self.control_step = 0
        self.capture_hold_count = 0

    def initialize(self, model, data) -> None:
        self.swingup_policy.initialize(model, data)
        self.lqr_policy.initialize(model, data)
        self.active_policy = "swingup"
        self.switch_step = None
        self.debug_history = []
        self.control_step = 0
        self.capture_hold_count = 0

    def _capture_condition(self, state: np.ndarray) -> tuple[bool, dict[str, float | bool]]:
        interpreted_state = interpret_planar_state(state)

        if self.lqr_policy.reference_state is None:
            reference_state = np.zeros_like(state)
        else:
            reference_state = self.lqr_policy.reference_state
        interpreted_reference = interpret_planar_state(reference_state)

        q1_error = wrap_angle_error(interpreted_state["q1"], interpreted_reference["q1"])
        q2_abs_error = wrap_angle_error(interpreted_state["q2_abs"], interpreted_reference["q2_abs"])
        q1_dot = interpreted_state["q1_dot"]
        q2_abs_dot = interpreted_state["q2_abs_dot"]
        x = interpreted_state["x"]

        is_captured = (
            abs(q1_error) <= self.q1_capture_rad
            and abs(q2_abs_error) <= self.q2_capture_rad
            and abs(q1_dot) <= self.q1_dot_capture
            and abs(q2_abs_dot) <= self.q2_abs_dot_capture
            and abs(x) <= self.x_capture_limit
        )

        return is_captured, {
            "q1_error_deg": np.rad2deg(q1_error),
            "q2_abs_error_deg": np.rad2deg(q2_abs_error),
            "q1_dot": q1_dot,
            "q2_abs_dot": q2_abs_dot,
            "x": x,
            "captured": is_captured,
        }

    def compute_control(self, state: np.ndarray) -> np.ndarray:
        is_captured, capture_debug = self._capture_condition(state)

        if self.active_policy == "swingup":
            if is_captured:
                self.capture_hold_count += 1
            else:
                self.capture_hold_count = 0

        if self.active_policy == "swingup" and self.capture_hold_count > self.h_lim:
            self.active_policy = "lqr"
            self.switch_step = self.control_step

        if self.active_policy == "swingup":
            control = self.swingup_policy.compute_control(state)
        else:
            control = self.lqr_policy.compute_control(state)

        self.debug_history.append(
            {
                "active_policy": self.active_policy,
                "captured": bool(capture_debug["captured"]),
                "q1_error_deg": float(capture_debug["q1_error_deg"]),
                "q2_abs_error_deg": float(capture_debug["q2_abs_error_deg"]),
                "q1_dot": float(capture_debug["q1_dot"]),
                "q2_abs_dot": float(capture_debug["q2_abs_dot"]),
                "x": float(capture_debug["x"]),
                "capture_hold_count": float(self.capture_hold_count),
            }
        )
        self.control_step += 1
        return control
