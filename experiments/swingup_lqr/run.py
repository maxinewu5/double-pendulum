"""
Run the shuttle swing-up policy with an LQR handoff wrapper and save a debug plot
that shows both the swing-up behavior and the capture-to-LQR transition.

Run this on macOS with:
  .venv/bin/mjpython double-pendulum/experiments/swingup_lqr/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import MJCF_MODEL
from policy import LQRPolicy, SwingUpLQRPolicy, SwingUpPolicy, wrap_angle
from rollout import RolloutConfig, RolloutScenario, run_rollout

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "swingup_lqr"


def _build_swingup_lqr_plot(
    time_s: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    swingup_debug_history: list[dict[str, float | str]],
    wrapper_debug_history: list[dict[str, float | str | bool]],
    kinetic: np.ndarray,
    potential: np.ndarray,
    total: np.ndarray,
    target_total: float,
    switch_time_s: float | None,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(15, 17), sharex=True)
    axes = axes.ravel()

    q1 = states[:, 1]
    q2_rel = states[:, 2]
    x = states[:, 0]
    x_dot = states[:, 3]
    q1_dot = states[:, 4]
    q2_rel_dot = states[:, 5]
    q2_abs = q1 + q2_rel
    q2_abs_dot = q1_dot + q2_rel_dot
    q1_deg = np.rad2deg(q1)
    q2_abs_deg = np.rad2deg(q2_abs)

    mode_names = [
        "drive_right",
        "brake_right",
        "hold_right",
        "drive_left",
        "brake_left",
        "hold_left",
    ]
    mode_to_idx = {name: idx for idx, name in enumerate(mode_names)}
    swing_mode_trace = np.full_like(time_s, np.nan, dtype=float)
    target_x = np.full_like(time_s, np.nan, dtype=float)
    raw_u = np.full_like(time_s, np.nan, dtype=float)
    for idx, entry in enumerate(swingup_debug_history[: len(time_s)]):
        swing_mode_trace[idx] = mode_to_idx.get(str(entry["mode"]), -1)
        target_x[idx] = float(entry["target_x"])
        raw_u[idx] = float(entry["u_raw"])

    active_policy_trace = np.asarray(
        [0.0 if str(entry["active_policy"]) == "swingup" else 1.0 for entry in wrapper_debug_history[: len(time_s)]],
        dtype=float,
    )
    q1_error_deg = np.asarray([float(entry["q1_error_deg"]) for entry in wrapper_debug_history[: len(time_s)]], dtype=float)
    q2_abs_error_deg = np.asarray([float(entry["q2_abs_error_deg"]) for entry in wrapper_debug_history[: len(time_s)]], dtype=float)
    capture_x = np.asarray([float(entry["x"]) for entry in wrapper_debug_history[: len(time_s)]], dtype=float)
    capture_q1_dot = np.asarray([float(entry["q1_dot"]) for entry in wrapper_debug_history[: len(time_s)]], dtype=float)
    capture_q2_abs_dot = np.asarray([float(entry["q2_abs_dot"]) for entry in wrapper_debug_history[: len(time_s)]], dtype=float)

    axes[0].plot(time_s, kinetic, label="kinetic")
    axes[0].plot(time_s, potential, label="potential")
    axes[0].plot(time_s, total, label="total")
    axes[0].axhline(target_total, color="tab:red", linestyle="--", label="upright target")
    axes[0].set_title("First-Link Energy")
    axes[0].set_ylabel("J")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s, q1_deg, label="q1")
    axes[1].plot(time_s, q2_abs_deg, label="q2 abs")
    axes[1].axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0, label="upright")
    axes[1].axhline(360.0, color="tab:red", linestyle=":", linewidth=1.0, label="upright + 360")
    axes[1].set_title("Absolute Link Angles")
    axes[1].set_ylabel("deg")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_s, q1_dot, label="q1_dot")
    axes[2].plot(time_s, q2_abs_dot, label="q2_abs_dot")
    axes[2].set_title("Absolute Link Angular Velocities")
    axes[2].set_ylabel("rad/s")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time_s, x, label="x")
    axes[3].plot(time_s, x_dot, label="x_dot")
    axes[3].plot(time_s, target_x, label="target_x", alpha=0.7)
    axes[3].set_title("Cart Position, Velocity, and Target")
    axes[3].set_ylabel("m, m/s")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(time_s, raw_u, label="u_raw")
    axes[4].plot(time_s, controls, label="u_applied", linestyle="--")
    axes[4].set_title("Applied Control")
    axes[4].set_ylabel("N")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    axes[5].step(time_s, swing_mode_trace, where="post")
    axes[5].set_title("Swing-Up State Machine Mode")
    axes[5].set_ylabel("mode")
    axes[5].set_yticks(list(mode_to_idx.values()), mode_names)
    axes[5].grid(True, alpha=0.3)

    axes[6].plot(time_s, q1_error_deg, label="q1 error [deg]")
    axes[6].plot(time_s, q2_abs_error_deg, label="q2 abs error [deg]")
    axes[6].axhline(5.0, color="tab:red", linestyle="--", linewidth=1.0)
    axes[6].axhline(-5.0, color="tab:red", linestyle="--", linewidth=1.0)
    axes[6].set_title("Capture Angle Errors")
    axes[6].set_ylabel("deg")
    axes[6].legend()
    axes[6].grid(True, alpha=0.3)

    axes[7].plot(time_s, capture_q1_dot, label="q1_dot")
    axes[7].plot(time_s, capture_q2_abs_dot, label="q2_abs_dot")
    axes[7].set_title("Capture Velocity Terms")
    axes[7].set_ylabel("rad/s")
    axes[7].legend()
    axes[7].grid(True, alpha=0.3)

    axes[8].step(time_s, active_policy_trace, where="post")
    axes[8].set_title("Active Policy")
    axes[8].set_ylabel("policy")
    axes[8].set_yticks([0.0, 1.0], ["swingup", "lqr"])
    axes[8].grid(True, alpha=0.3)

    axes[9].plot(time_s, capture_x, label="x")
    axes[9].axhline(1.5, color="tab:red", linestyle="--", linewidth=1.0, label="x capture limit")
    axes[9].axhline(-1.5, color="tab:red", linestyle="--", linewidth=1.0)
    axes[9].set_title("Capture Cart Position")
    axes[9].set_ylabel("m")
    axes[9].legend()
    axes[9].grid(True, alpha=0.3)

    if switch_time_s is not None:
        for axis in axes:
            axis.axvline(
                switch_time_s,
                color="tab:green",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
            )

    for axis in axes:
        axis.set_xlabel("time [s]")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_post_handoff_plot(
    time_s: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    switch_time_s: float,
    output_path: Path,
) -> None:
    handoff_idx = int(np.searchsorted(time_s, switch_time_s))
    post_time_s = time_s[handoff_idx:]
    post_states = states[handoff_idx:]
    post_controls = controls[handoff_idx:]

    q1_wrapped_deg = np.rad2deg(np.array([wrap_angle(state[1]) for state in post_states], dtype=float))
    q2_abs_wrapped_deg = np.rad2deg(
        np.array([wrap_angle(state[1] + state[2]) for state in post_states], dtype=float)
    )
    q1_dot = post_states[:, 4]
    q2_abs_dot = post_states[:, 4] + post_states[:, 5]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(post_time_s, q1_wrapped_deg, label="q1 wrapped")
    axes[0].plot(post_time_s, q2_abs_wrapped_deg, label="q2 abs wrapped")
    axes[0].axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0, label="upright")
    axes[0].set_ylim(-10.0, 10.0)
    axes[0].set_title("Post-Handoff Wrapped Link Angles")
    axes[0].set_ylabel("deg")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(post_time_s, q1_dot, label="q1_dot")
    axes[1].plot(post_time_s, q2_abs_dot, label="q2_abs_dot")
    axes[1].axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
    axes[1].set_ylim(-3.0, 3.0)
    axes[1].set_title("Post-Handoff Angular Velocities")
    axes[1].set_ylabel("rad/s")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(post_time_s, post_controls, label="u_applied")
    axes[2].axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
    axes[2].set_title("Post-Handoff Applied Control")
    axes[2].set_ylabel("N")
    axes[2].set_xlabel("time [s]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    # tuned swing up policy
    swingup_policy = SwingUpPolicy(
        control_limit=30.0,
        initial_kick_force=15.0,
        initial_kick_duration_s=0.5,
        x_target=0.4,
        brake_margin=0.5,
        launch_force=18.0,
        brake_force=15.0,
        apex_velocity_threshold=0.2,
        hold_kd=8.0,
    )

    # tuned lqr parameters
    Q = np.diag([
        8.5,   # cart position
        10.0,   # q1
        10.0,   # q2_rel
        4.5,   # cart velocity
        1.5,   # q1_dot
        1.0,   # q2_rel_dot
    ])
    R = np.array([[1.0]])

    lqr_policy = LQRPolicy(
        Q=Q,
        R=R,
        control_limit=60.0,
        reference_state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )

    policy = SwingUpLQRPolicy(
        swingup_policy=swingup_policy,
        lqr_policy=lqr_policy,
        q1_capture_deg=5.0,
        q2_capture_deg=5.0,
        q1_dot_capture=2.0,
        q2_abs_dot_capture=2.0,
        x_capture_limit=2.0,
        h_lim=2,
    )

    metrics = run_rollout(
        model_xml=MJCF_MODEL,
        policy=policy,
        scenario=RolloutScenario(
            init_qpos=np.array([0.0, np.deg2rad(180), np.deg2rad(0.0)]),
            init_qvel=np.array([0.0, 0.0, 0.0]),
            name="swingup_lqr",
        ),
        config=RolloutConfig(render=True, logging=True),
    )

    print("LQR K:")
    print(lqr_policy.K)

    states = np.asarray(metrics["state_history"], dtype=float)
    controls = np.asarray(metrics["control_history"], dtype=float)
    dt = 0.005
    time_s = np.arange(states.shape[0], dtype=float) * dt

    energy_breakdown = [swingup_policy.compute_first_link_energy_breakdown(state) for state in states]
    kinetic = np.asarray([entry["kinetic"] for entry in energy_breakdown], dtype=float)
    potential = np.asarray([entry["potential"] for entry in energy_breakdown], dtype=float)
    total = np.asarray([entry["total"] for entry in energy_breakdown], dtype=float)
    target_total = float(energy_breakdown[0]["target_total"])

    switch_time_s = None
    if policy.switch_step is not None:
        switch_time_s = policy.switch_step * dt

    output_path = OUTPUT_DIR / "swingup_lqr_debug.png"
    _build_swingup_lqr_plot(
        time_s=time_s,
        states=states,
        controls=controls,
        swingup_debug_history=swingup_policy.debug_history,
        wrapper_debug_history=policy.debug_history,
        kinetic=kinetic,
        potential=potential,
        total=total,
        target_total=target_total,
        switch_time_s=switch_time_s,
        output_path=output_path,
    )
    post_handoff_output_path = OUTPUT_DIR / "swingup_lqr_post_handoff.png"
    if switch_time_s is not None:
        _build_post_handoff_plot(
            time_s=time_s,
            states=states,
            controls=controls,
            switch_time_s=switch_time_s,
            output_path=post_handoff_output_path,
        )

    print(f"saved swing-up + LQR debug plot to {output_path.resolve()}")
    if switch_time_s is not None:
        print(f"saved post-handoff debug plot to {post_handoff_output_path.resolve()}")
    print(f"first-link upright target energy: {target_total:.4f} J")
    if switch_time_s is not None:
        print(f"LQR handoff at t={switch_time_s:.4f} s")
    else:
        print("LQR handoff did not occur during this rollout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
