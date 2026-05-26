from __future__ import annotations

import numpy as np

from .base import Policy
from .utils import interpret_planar_state


class SwingUpPolicy(Policy):
    def __init__(
        self,
        control_limit: float | None = None,
        initial_kick_force: float = 0.0,
        initial_kick_duration_s: float = 0.0,
        x_target: float = 0.8,
        brake_margin: float = 0.2,
        x_tolerance: float = 0.05,
        stop_velocity_threshold: float = 0.15,
        launch_force: float = 8.0,
        brake_force: float = 12.0,
        hold_kp: float = 20.0,
        hold_kd: float = 8.0,
        apex_velocity_threshold: float = 0.2,
    ):
        self.control_limit = control_limit
        self.initial_kick_force = initial_kick_force
        self.initial_kick_duration_s = initial_kick_duration_s
        self.x_target = x_target
        self.brake_margin = brake_margin
        self.x_tolerance = x_tolerance
        self.stop_velocity_threshold = stop_velocity_threshold
        self.launch_force = launch_force
        self.brake_force = brake_force
        self.hold_kp = hold_kp
        self.hold_kd = hold_kd
        self.apex_velocity_threshold = apex_velocity_threshold
        self.dt: float | None = None
        self.control_step = 0
        self.initial_kick_steps_remaining = 0
        self.mode = "drive_right"
        self.debug_history: list[dict[str, float | str]] = []

        # First-link parameters from the MJCF model. We keep the energy model
        # focused on link1 for now so we can validate it before adding link2.
        self.g = 9.81
        self.link1_length = 0.6
        self.link1_com = self.link1_length / 2.0
        self.link1_mass = 0.35
        self.link1_inertia_com = (1.0 / 12.0) * self.link1_mass * self.link1_length**2
        self.direction_velocity_threshold = 0.05

    def initialize(self, model, data) -> None:
        self.dt = model.opt.timestep
        self.control_step = 0
        self.mode = self._select_start_mode(np.concatenate([data.qpos.copy(), data.qvel.copy()]))
        self.debug_history = []
        self.initial_kick_steps_remaining = max(
            0,
            int(round(self.initial_kick_duration_s / self.dt)),
        )

    def _clip_control(self, u: float) -> np.ndarray:
        if self.control_limit is not None:
            u = float(np.clip(u, -self.control_limit, self.control_limit))
        return np.array([u], dtype=float)

    def compute_first_link_energy_breakdown(self, state: np.ndarray) -> dict[str, float]:
        """
        Compute the first link's kinetic, potential, and total energy.

        Angle convention:
        - q1 = 0 means link1 points straight up
        - q1 = pi or -pi means link1 points straight down
        """
        interpreted_state = interpret_planar_state(state)
        q1 = interpreted_state["q1"]
        q1_dot = interpreted_state["q1_dot"]

        com_speed_sq = (self.link1_com * q1_dot) ** 2
        kinetic = (
            0.5 * self.link1_mass * com_speed_sq
            + 0.5 * self.link1_inertia_com * q1_dot**2
        )
        potential = self.g * self.link1_mass * self.link1_com * np.cos(q1)
        total = kinetic + potential
        target_total = self.g * self.link1_mass * self.link1_com

        return {
            "q1": q1,
            "q1_dot": q1_dot,
            "kinetic": kinetic,
            "potential": potential,
            "total": total,
            "target_total": target_total,
        }

    def compute_first_link_target_energy(self) -> float:
        """Return the first-link total energy at the upright apex (q1 = 0)."""
        return self.g * self.link1_mass * self.link1_com

    def _is_near_apex(self, q1_dot: float) -> bool:
        return abs(q1_dot) <= self.apex_velocity_threshold

    def _is_stopped_at_target(self, x: float, x_dot: float, target_x: float) -> bool:
        return (
            abs(x - target_x) <= self.x_tolerance
            and abs(x_dot) <= self.stop_velocity_threshold
        )

    def _should_exit_brake_right(self, x: float, x_dot: float, target_x: float) -> bool:
        if self._is_stopped_at_target(x, x_dot, target_x):
            return True
        # If we have passed the right target and are now moving back left, the
        # braking phase has done its job well enough to hand off to hold mode.
        if x >= target_x and x_dot <= 0.0:
            return True
        # If the cart is already moving slowly after entering the braking region,
        # let hold mode take over and actively regulate the final position.
        if x >= target_x - self.brake_margin and abs(x_dot) <= self.stop_velocity_threshold:
            return True
        return False

    def _should_exit_brake_left(self, x: float, x_dot: float, target_x: float) -> bool:
        if self._is_stopped_at_target(x, x_dot, target_x):
            return True
        if x <= target_x and x_dot >= 0.0:
            return True
        if x <= target_x + self.brake_margin and abs(x_dot) <= self.stop_velocity_threshold:
            return True
        return False

    def _hold_control(self, x: float, x_dot: float, target_x: float) -> float:
        return -self.hold_kp * (x - target_x) - self.hold_kd * x_dot

    def _choose_swing_direction(self, x_dot: float, q1_dot: float) -> str:
        if q1_dot >= self.direction_velocity_threshold:
            return "right"
        if q1_dot <= -self.direction_velocity_threshold:
            return "left"
        if x_dot >= self.direction_velocity_threshold:
            return "right"
        if x_dot <= -self.direction_velocity_threshold:
            return "left"
        return "right" if self.initial_kick_force >= 0.0 else "left"

    def _select_start_mode(self, state: np.ndarray) -> str:
        interpreted_state = interpret_planar_state(state)
        direction = self._choose_swing_direction(
            x_dot=interpreted_state["x_dot"],
            q1_dot=interpreted_state["q1_dot"],
        )
        return "drive_right" if direction == "right" else "drive_left"

    def _record_debug(self, state: np.ndarray, raw_u: float, phase: str, target_x: float | None) -> None:
        interpreted_state = interpret_planar_state(state)
        self.debug_history.append(
            {
                "mode": self.mode,
                "phase": phase,
                "x": interpreted_state["x"],
                "x_dot": interpreted_state["x_dot"],
                "q1": interpreted_state["q1"],
                "q1_dot": interpreted_state["q1_dot"],
                "target_x": np.nan if target_x is None else target_x,
                "u_raw": raw_u,
            }
        )

    def _compute_swingup_control(self, state: np.ndarray) -> float:
        interpreted_state = interpret_planar_state(state)
        x = interpreted_state["x"]
        x_dot = interpreted_state["x_dot"]
        q1_dot = interpreted_state["q1_dot"]

        right_target = self.x_target
        left_target = -self.x_target

        if self.mode == "drive_right":
            if x >= right_target - self.brake_margin:
                self.mode = "brake_right"
                u = -self.brake_force
                self._record_debug(state, u, "brake", right_target)
                return u
            u = self.launch_force
            self._record_debug(state, u, "drive", right_target)
            return u

        if self.mode == "brake_right":
            if self._should_exit_brake_right(x, x_dot, right_target):
                self.mode = "hold_right"
                u = self._hold_control(x, x_dot, right_target)
                self._record_debug(state, u, "hold", right_target)
                return u
            u = -self.brake_force
            self._record_debug(state, u, "brake", right_target)
            return u

        if self.mode == "hold_right":
            if self._is_near_apex(q1_dot):
                self.mode = "drive_left"
                u = -self.launch_force
                self._record_debug(state, u, "launch_left", right_target)
                return u
            u = self._hold_control(x, x_dot, right_target)
            self._record_debug(state, u, "hold", right_target)
            return u

        if self.mode == "drive_left":
            if x <= left_target + self.brake_margin:
                self.mode = "brake_left"
                u = self.brake_force
                self._record_debug(state, u, "brake", left_target)
                return u
            u = -self.launch_force
            self._record_debug(state, u, "drive", left_target)
            return u

        if self.mode == "brake_left":
            if self._should_exit_brake_left(x, x_dot, left_target):
                self.mode = "hold_left"
                u = self._hold_control(x, x_dot, left_target)
                self._record_debug(state, u, "hold", left_target)
                return u
            u = self.brake_force
            self._record_debug(state, u, "brake", left_target)
            return u

        if self.mode == "hold_left":
            if self._is_near_apex(q1_dot):
                self.mode = "drive_right"
                u = self.launch_force
                self._record_debug(state, u, "launch_right", left_target)
                return u
            u = self._hold_control(x, x_dot, left_target)
            self._record_debug(state, u, "hold", left_target)
            return u

        raise RuntimeError(f"Unknown swing-up mode: {self.mode}")

    def compute_control(self, state: np.ndarray) -> np.ndarray:
        if self.dt is None:
            raise RuntimeError("Policy must be initialized before use.")
        if self.initial_kick_steps_remaining > 0:
            self.initial_kick_steps_remaining -= 1
            self._record_debug(state, self.initial_kick_force, "initial_kick", None)
            control = self._clip_control(self.initial_kick_force)
        else:
            if self.control_step == 0 or (
                self.control_step == max(0, int(round(self.initial_kick_duration_s / self.dt)))
                and self.initial_kick_duration_s > 0.0
            ):
                self.mode = self._select_start_mode(state)
            control = self._clip_control(self._compute_swingup_control(state))
        self.control_step += 1
        return control
