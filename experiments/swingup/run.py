"""
Render the same cart + double pendulum setup so the plots have a visual match.

Run this on macOS with:
  .venv/bin/mjpython double-pendulum/experiments/swingup/run.py
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
from rollout import run_rollout, RolloutConfig, RolloutScenario
from model import MJCF_MODEL
from policy import SwingUpPolicy, analyze_capture_region_dwell

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "swingup"
Q1_CAPTURE_LIMIT_RAD = 0.08
Q2_ABS_CAPTURE_LIMIT_RAD = 0.08
Q1_DOT_CAPTURE_LIMIT = 2.0
Q2_ABS_DOT_CAPTURE_LIMIT = 2.0


def _build_energy_debug_plot(
    time_s: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    kinetic: np.ndarray,
    potential: np.ndarray,
    total: np.ndarray,
    target_total: float,
    debug_history: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
    axes = axes.ravel()

    q1 = states[:, 1]
    q2_rel = states[:, 2]
    x = states[:, 0]
    x_dot = states[:, 3]
    q1_dot = states[:, 4]
    q2_rel_dot = states[:, 5]
    q2_abs = q1 + q2_rel
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
    mode_trace = np.asarray([mode_to_idx.get(str(entry["mode"]), -1) for entry in debug_history], dtype=float)
    target_x = np.asarray([float(entry["target_x"]) for entry in debug_history], dtype=float)
    raw_u = np.asarray([float(entry["u_raw"]) for entry in debug_history], dtype=float)

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
    axes[2].plot(time_s, q2_rel_dot, label="q2_rel_dot")
    axes[2].set_title("Joint Angular Velocities")
    axes[2].set_ylabel("rad/s")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time_s, x, label="x")
    axes[3].plot(time_s, x_dot, label="x_dot")
    axes[3].set_title("Cart Kinematics")
    axes[3].set_ylabel("m, m/s")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(time_s, q2_rel, label="q2_rel")
    axes[4].set_title("Relative Link2 Joint Angle")
    axes[4].set_ylabel("rad")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(time_s, raw_u, label="u_raw")
    axes[5].plot(time_s, controls, label="u_applied", linestyle="--")
    axes[5].set_title("Applied Control")
    axes[5].set_ylabel("N")
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)

    axes[6].plot(time_s, target_x, label="target_x")
    axes[6].plot(time_s, x, label="x", alpha=0.8)
    axes[6].set_title("Cart Position vs Target Waypoint")
    axes[6].set_ylabel("m")
    axes[6].legend()
    axes[6].grid(True, alpha=0.3)

    axes[7].step(time_s, mode_trace, where="post")
    axes[7].set_title("Shuttle State Machine Mode")
    axes[7].set_ylabel("mode")
    axes[7].set_yticks(list(mode_to_idx.values()), mode_names)
    axes[7].grid(True, alpha=0.3)

    for axis in axes:
        axis.set_xlabel("time [s]")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    # policy v1: swing up, but too fast by upright, so it overshoots
    # policy = SwingUpPolicy(
    #     control_limit=30.0,
    #     initial_kick_force=15.0,
    #     initial_kick_duration_s=0.5,
    #     x_target=0.5,
    #     brake_margin=0.3,
    #     launch_force=20.0,
    #     brake_force=17.0,
    # )

    policy = SwingUpPolicy(
        control_limit=30.0,
        initial_kick_force=15.0,
        initial_kick_duration_s=0.5,
        x_target=0.4,
        brake_margin=0.5,
        launch_force=18.0,
        brake_force=15.0,
        apex_velocity_threshold=0.4
    )
    
    metrics = run_rollout(
        model_xml=MJCF_MODEL,
        policy=policy,
        scenario=RolloutScenario(
            init_qpos=np.array([0.0, np.deg2rad(180), np.deg2rad(0.0)]),
            init_qvel=np.array([0.0, 0.0, 0.0]),
            name="swingup",
        ),
        config=RolloutConfig(render=True, logging=True),
    )

    states = np.asarray(metrics["state_history"], dtype=float)
    controls = np.asarray(metrics["control_history"], dtype=float)
    dt = 0.005
    time_s = np.arange(states.shape[0], dtype=float) * dt
    dwell_metrics = analyze_capture_region_dwell(
        states,
        q1_error_limit_rad=Q1_CAPTURE_LIMIT_RAD,
        q2_abs_error_limit_rad=Q2_ABS_CAPTURE_LIMIT_RAD,
        q1_dot_limit=Q1_DOT_CAPTURE_LIMIT,
        q2_abs_dot_limit=Q2_ABS_DOT_CAPTURE_LIMIT,
        dt=dt,
    )

    energy_breakdown = [policy.compute_first_link_energy_breakdown(state) for state in states]
    kinetic = np.asarray([entry["kinetic"] for entry in energy_breakdown], dtype=float)
    potential = np.asarray([entry["potential"] for entry in energy_breakdown], dtype=float)
    total = np.asarray([entry["total"] for entry in energy_breakdown], dtype=float)
    target_total = float(energy_breakdown[0]["target_total"])

    output_path = OUTPUT_DIR / "swingup_energy_debug.png"
    _build_energy_debug_plot(
        time_s=time_s,
        states=states,
        controls=controls,
        kinetic=kinetic,
        potential=potential,
        total=total,
        target_total=target_total,
        debug_history=policy.debug_history,
        output_path=output_path,
    )
    print(f"saved energy debug plot to {output_path.resolve()}")
    print(f"first-link upright target energy: {target_total:.4f} J")
    print(
        "capture-region thresholds: "
        f"|q1_error| <= {Q1_CAPTURE_LIMIT_RAD:.3f} rad, "
        f"|q2_abs_error| <= {Q2_ABS_CAPTURE_LIMIT_RAD:.3f} rad, "
        f"|q1_dot| <= {Q1_DOT_CAPTURE_LIMIT:.3f} rad/s, "
        f"|q2_abs_dot| <= {Q2_ABS_DOT_CAPTURE_LIMIT:.3f} rad/s"
    )
    print(
        "capture-region dwell: "
        f"entered={dwell_metrics['entered_region']}, "
        f"first_entry_time_s={dwell_metrics['first_entry_time_s']:.3f}, "
        f"total_time_in_region_s={dwell_metrics['total_time_in_region_s']:.3f}, "
        f"longest_streak_time_s={dwell_metrics['longest_streak_time_s']:.3f}, "
        f"best_normalized_score={dwell_metrics['best_normalized_score']:.3f}"
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
