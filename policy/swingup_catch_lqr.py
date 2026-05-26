from __future__ import annotations

import numpy as np

from .base import Policy
from .lqr import LQRPolicy
from .swingup import SwingUpPolicy
from .utils import interpret_planar_state, wrap_angle_error


class SwingUpCatchLQRPolicy(Policy):
    """
    Three-stage controller:
    1. shuttle-style swing-up
    2. optional catch / counter-balance mode near upright
    3. LQR stabilization once angles and velocities are small enough
    """

    def __init__(
        self,
        swingup_policy: SwingUpPolicy,
        lqr_policy: LQRPolicy,
        catch_q1_angle_deg: float = 12.0,
        catch_q2_angle_deg: float = 12.0,
        catch_cart_limit: float | None = None,
        catch_force_limit: float = 12.0,
        catch_k_q1dot: float = 1.0,
        catch_k_q2dot: float = 1.0,
        catch_k_q1: float | None = None,
        catch_k_q2: float | None = None,
        catch_k_x: float | None = None,
        catch_k_xdot: float | None = None,
        lqr_q1_angle_deg: float = 5.0,
        lqr_q2_angle_deg: float = 5.0,
        lqr_q1_dot_thresh: float = 1.0,
        lqr_q2_dot_thresh: float = 1.5,
        lqr_x_limit: float | None = None,
    ):
        self.swingup_policy = swingup_policy
        self.lqr_policy = lqr_policy

        self.catch_q1_angle_rad = np.deg2rad(catch_q1_angle_deg)
        self.catch_q2_angle_rad = np.deg2rad(catch_q2_angle_deg)
        self.catch_cart_limit = catch_cart_limit
        self.catch_force_limit = catch_force_limit
        self.catch_k_q1dot = catch_k_q1dot
        self.catch_k_q2dot = catch_k_q2dot
        self.catch_k_q1 = catch_k_q1
        self.catch_k_q2 = catch_k_q2
        self.catch_k_x = catch_k_x
        self.catch_k_xdot = catch_k_xdot

        self.lqr_q1_angle_rad = np.deg2rad(lqr_q1_angle_deg)
        self.lqr_q2_angle_rad = np.deg2rad(lqr_q2_angle_deg)
        self.lqr_q1_dot_thresh = lqr_q1_dot_thresh
        self.lqr_q2_dot_thresh = lqr_q2_dot_thresh
        self.lqr_x_limit = lqr_x_limit

        self.active_policy = "swingup"
        self.switch_step: int | None = None
        self.catch_step: int | None = None
        self.control_step = 0
        self.debug_history: list[dict[str, float | str | bool]] = []

    def initialize(self, model, data) -> None:
        self.swingup_policy.initialize(model, data)
        self.lqr_policy.initialize(model, data)
        self.active_policy = "swingup"
        self.switch_step = None
        self.catch_step = None
        self.control_step = 0
        self.debug_history = []

    def _get_reference_interpretation(self, state: np.ndarray) -> dict[str, float]:
        if self.lqr_policy.reference_state is None:
            reference_state = np.zeros_like(state)
        else:
            reference_state = self.lqr_policy.reference_state
        return interpret_planar_state(reference_state)

    def _compute_tracking_terms(self, state: np.ndarray) -> dict[str, float]:
        interpreted_state = interpret_planar_state(state)
        interpreted_reference = self._get_reference_interpretation(state)

        q1_error = wrap_angle_error(interpreted_state["q1"], interpreted_reference["q1"])
        q2_abs_error = wrap_angle_error(interpreted_state["q2_abs"], interpreted_reference["q2_abs"])

        return {
            "x": interpreted_state["x"],
            "x_dot": interpreted_state["x_dot"],
            "q1_error": q1_error,
            "q2_abs_error": q2_abs_error,
            "q1_dot": interpreted_state["q1_dot"],
            "q2_abs_dot": interpreted_state["q2_abs_dot"],
        }

    def _in_catch_region(self, terms: dict[str, float]) -> bool:
        if abs(terms["q1_error"]) > self.catch_q1_angle_rad:
            return False
        if abs(terms["q2_abs_error"]) > self.catch_q2_angle_rad:
            return False
        if self.catch_cart_limit is not None and abs(terms["x"]) > self.catch_cart_limit:
            return False
        return True

    def _in_lqr_region(self, terms: dict[str, float]) -> bool:
        if abs(terms["q1_error"]) > self.lqr_q1_angle_rad:
            return False
        if abs(terms["q2_abs_error"]) > self.lqr_q2_angle_rad:
            return False
        if abs(terms["q1_dot"]) > self.lqr_q1_dot_thresh:
            return False
        if abs(terms["q2_abs_dot"]) > self.lqr_q2_dot_thresh:
            return False
        if self.lqr_x_limit is not None and abs(terms["x"]) > self.lqr_x_limit:
            return False
        return True

    def _compute_catch_control(self, terms: dict[str, float]) -> np.ndarray:
        u = 0.0
        u += -self.catch_k_q1dot * terms["q1_dot"]
        u += -self.catch_k_q2dot * terms["q2_abs_dot"]
        if self.catch_k_q1 is not None:
            u += -self.catch_k_q1 * terms["q1_error"]
        if self.catch_k_q2 is not None:
            u += -self.catch_k_q2 * terms["q2_abs_error"]
        if self.catch_k_x is not None:
            u += -self.catch_k_x * terms["x"]
        if self.catch_k_xdot is not None:
            u += -self.catch_k_xdot * terms["x_dot"]
        u = float(np.clip(u, -self.catch_force_limit, self.catch_force_limit))
        return np.array([u], dtype=float)

    def compute_control(self, state: np.ndarray) -> np.ndarray:
        terms = self._compute_tracking_terms(state)
        in_catch_region = self._in_catch_region(terms)
        in_lqr_region = self._in_lqr_region(terms)

        if self.active_policy == "swingup" and in_catch_region:
            self.active_policy = "catch"
            self.catch_step = self.control_step
        if self.active_policy == "catch" and in_lqr_region:
            self.active_policy = "lqr"
            self.switch_step = self.control_step

        if self.active_policy == "swingup":
            control = self.swingup_policy.compute_control(state)
        elif self.active_policy == "catch":
            control = self._compute_catch_control(terms)
        else:
            control = self.lqr_policy.compute_control(state)

        self.debug_history.append(
            {
                "active_policy": self.active_policy,
                "in_catch_region": in_catch_region,
                "in_lqr_region": in_lqr_region,
                "q1_error_deg": np.rad2deg(terms["q1_error"]),
                "q2_abs_error_deg": np.rad2deg(terms["q2_abs_error"]),
                "q1_dot": terms["q1_dot"],
                "q2_abs_dot": terms["q2_abs_dot"],
                "x": terms["x"],
                "x_dot": terms["x_dot"],
                "u_catch_raw": float(control[0]) if self.active_policy == "catch" else np.nan,
            }
        )
        self.control_step += 1
        return control
