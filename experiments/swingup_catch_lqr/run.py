"""
Run the shuttle swing-up policy with an intermediate catch phase and an LQR
handoff, then save a debug plot showing all three controller phases.

Run this on macOS with:
  .venv/bin/mjpython double-pendulum/experiments/swingup_catch_lqr/run.py
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
from policy import LQRPolicy, SwingUpCatchLQRPolicy, SwingUpPolicy
from rollout import RolloutConfig, RolloutScenario, run_rollout

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "swingup_catch_lqr"


def _build_swingup_catch_lqr_plot(
    time_s: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    swingup_debug_history: list[dict[str, float | str]],
    wrapper_debug_history: list[dict[str, float | str | bool]],
    kinetic: np.ndarray,
    potential: np.ndarray,
    total: np.ndarray,
    target_total: float,
    catch_time_s: float | None,
    switch_time_s: float | None,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(6, 2, figsize=(15, 20), sharex=True)
    axes = axes.ravel()

    q1 = states[:, 1]
    q2_rel = states[:, 2]
    x = states[:, 0]
    x_dot = states[:, 3]
    q1_dot = states[:, 4]
    q2_rel_dot = states[:, 5]
    q2_abs = q1 + q2_rel
    q2_abs_dot = q1_dot + q2_rel_dot

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

    policy_map = {"swingup": 0.0, "catch": 1.0, "lqr": 2.0}
    active_policy_trace = np.asarray(
        [policy_map.get(str(entry["active_policy"]), np.nan) for entry in wrapper_debug_history[: len(time_s)]],
        dtype=float,
    )
    q1_error_deg = np.asarray([float(entry["q1_error_deg"]) for entry in wrapper_debug_history[: len(time_s)]], dtype=float)
    q2_abs_error_deg = np.asarray([float(entry["q2_abs_error_deg"]) for entry in wrapper_debug_history[: len(time_s)]], dtype=float)
    track_x = np.asarray([float(entry["x"]) for entry in wrapper_debug_history[: len(time_s)]], dtype=float)
    track_x_dot = np.asarray([float(entry["x_dot"]) for entry in wrapper_debug_history[: len(time_s)]], dtype=float)
    track_q1_dot = np.asarray([float(entry["q1_dot"]) for entry in wrapper_debug_history[: len(time_s)]], dtype=float)
    track_q2_abs_dot = np.asarray([float(entry["q2_abs_dot"]) for entry in wrapper_debug_history[: len(time_s)]], dtype=float)
    catch_u = np.asarray([float(entry["u_catch_raw"]) for entry in wrapper_debug_history[: len(time_s)]], dtype=float)
    in_catch_region = np.asarray(
        [1.0 if bool(entry["in_catch_region"]) else 0.0 for entry in wrapper_debug_history[: len(time_s)]],
        dtype=float,
    )
    in_lqr_region = np.asarray(
        [1.0 if bool(entry["in_lqr_region"]) else 0.0 for entry in wrapper_debug_history[: len(time_s)]],
        dtype=float,
    )

    axes[0].plot(time_s, kinetic, label="kinetic")
    axes[0].plot(time_s, potential, label="potential")
    axes[0].plot(time_s, total, label="total")
    axes[0].axhline(target_total, color="tab:red", linestyle="--", label="upright target")
    axes[0].set_title("First-Link Energy")
    axes[0].set_ylabel("J")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s, q1, label="q1")
    axes[1].plot(time_s, q2_abs, label="q2 abs")
    axes[1].axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0, label="upright")
    axes[1].axhline(2.0 * np.pi, color="tab:red", linestyle=":", linewidth=1.0, label="upright + 2pi")
    axes[1].set_title("Absolute Link Angles")
    axes[1].set_ylabel("rad")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_s, q1_dot, label="q1_dot")
    axes[2].plot(time_s, q2_abs_dot, label="q2_abs_dot")
    axes[2].set_title("Absolute Link Angular Velocities")
    axes[2].set_ylim(-5, 5)
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

    axes[4].plot(time_s, raw_u, label="swingup_u_raw")
    axes[4].plot(time_s, catch_u, label="catch_u", alpha=0.8)
    axes[4].plot(time_s, controls, label="u_applied", linestyle="--")
    axes[4].set_title("Applied Control by Phase")
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
    axes[6].set_title("Angle Errors Relative to LQR Target")
    axes[6].set_ylabel("deg")
    axes[6].set_ylim(-45.0, 45.0)
    axes[6].legend()
    axes[6].grid(True, alpha=0.3)

    axes[7].plot(time_s, track_q1_dot, label="q1_dot")
    axes[7].plot(time_s, track_q2_abs_dot, label="q2_abs_dot")
    axes[7].set_title("Velocity Terms for Catch/LQR Logic")
    axes[7].set_ylabel("rad/s")
    axes[7].set_ylim(-10.0, 10.0)
    axes[7].legend()
    axes[7].grid(True, alpha=0.3)

    axes[8].step(time_s, active_policy_trace, where="post")
    axes[8].set_title("Active Policy")
    axes[8].set_ylabel("policy")
    axes[8].set_yticks([0.0, 1.0, 2.0], ["swingup", "catch", "lqr"])
    axes[8].grid(True, alpha=0.3)

    axes[9].step(time_s, in_catch_region, where="post", label="in catch region")
    axes[9].step(time_s, in_lqr_region, where="post", label="in lqr region")
    axes[9].set_title("Region Tests")
    axes[9].set_ylabel("bool")
    axes[9].set_yticks([0.0, 1.0], ["False", "True"])
    axes[9].legend()
    axes[9].grid(True, alpha=0.3)

    axes[10].plot(time_s, track_x, label="x")
    axes[10].plot(time_s, track_x_dot, label="x_dot")
    axes[10].set_title("Tracking Terms: Cart State")
    axes[10].set_ylabel("m, m/s")
    axes[10].legend()
    axes[10].grid(True, alpha=0.3)

    axes[11].plot(time_s, q2_rel, label="q2_rel")
    axes[11].set_title("Relative Link2 Joint Angle")
    axes[11].set_ylabel("rad")
    axes[11].legend()
    axes[11].grid(True, alpha=0.3)

    markers = []
    if catch_time_s is not None:
        markers.append((catch_time_s, "tab:orange"))
    if switch_time_s is not None:
        markers.append((switch_time_s, "tab:green"))
    for marker_time, color in markers:
        for axis in axes:
            axis.axvline(
                marker_time,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
            )

    for axis in axes:
        axis.set_xlabel("time [s]")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    swingup_policy = SwingUpPolicy(
        control_limit=30.0,
        initial_kick_force=15.0,
        initial_kick_duration_s=0.5,
        x_target=0.5,
        brake_margin=0.3,
        launch_force=19.7,
        brake_force=17.0,
    )
    lqr_policy = LQRPolicy(
        Q=np.diag([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        R=np.array([[20.0]]),
        control_limit=30.0,
        reference_state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    policy = SwingUpCatchLQRPolicy(
        swingup_policy=swingup_policy,
        lqr_policy=lqr_policy,
        catch_q1_angle_deg=5.0,
        catch_q2_angle_deg=5.0,
        catch_cart_limit=None,
        catch_force_limit=40.0,
        catch_k_q1dot=6.0,
        catch_k_q2dot=3.0,
        catch_k_q1=None,
        catch_k_q2=None,
        catch_k_x=None,
        catch_k_xdot=None,
        lqr_q1_angle_deg=5.0,
        lqr_q2_angle_deg=5.0,
        lqr_q1_dot_thresh=2.0,
        lqr_q2_dot_thresh=2.0,
        lqr_x_limit=None,
    )

    metrics = run_rollout(
        model_xml=MJCF_MODEL,
        policy=policy,
        scenario=RolloutScenario(
            init_qpos=np.array([0.0, np.deg2rad(180), np.deg2rad(0.0)]),
            init_qvel=np.array([0.0, 0.0, 0.0]),
            name="swingup_catch_lqr",
        ),
        config=RolloutConfig(render=True, logging=True),
    )

    states = np.asarray(metrics["state_history"], dtype=float)
    controls = np.asarray(metrics["control_history"], dtype=float)
    dt = 0.005
    time_s = np.arange(states.shape[0], dtype=float) * dt

    energy_breakdown = [swingup_policy.compute_first_link_energy_breakdown(state) for state in states]
    kinetic = np.asarray([entry["kinetic"] for entry in energy_breakdown], dtype=float)
    potential = np.asarray([entry["potential"] for entry in energy_breakdown], dtype=float)
    total = np.asarray([entry["total"] for entry in energy_breakdown], dtype=float)
    target_total = float(energy_breakdown[0]["target_total"])

    catch_time_s = None
    if policy.catch_step is not None:
        catch_time_s = policy.catch_step * dt
    switch_time_s = None
    if policy.switch_step is not None:
        switch_time_s = policy.switch_step * dt

    output_path = OUTPUT_DIR / "swingup_catch_lqr_debug.png"
    _build_swingup_catch_lqr_plot(
        time_s=time_s,
        states=states,
        controls=controls,
        swingup_debug_history=swingup_policy.debug_history,
        wrapper_debug_history=policy.debug_history,
        kinetic=kinetic,
        potential=potential,
        total=total,
        target_total=target_total,
        catch_time_s=catch_time_s,
        switch_time_s=switch_time_s,
        output_path=output_path,
    )

    print(f"saved swing-up + catch + LQR debug plot to {output_path.resolve()}")
    print(f"first-link upright target energy: {target_total:.4f} J")
    if catch_time_s is not None:
        print(f"entered catch mode at t={catch_time_s:.4f} s")
    else:
        print("catch mode did not occur during this rollout")
    if switch_time_s is not None:
        print(f"LQR handoff at t={switch_time_s:.4f} s")
    else:
        print("LQR handoff did not occur during this rollout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
