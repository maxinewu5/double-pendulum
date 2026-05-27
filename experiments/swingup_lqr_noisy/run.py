"""
Run the swing-up + LQR pipeline with optional Gaussian sensor noise.

Run this on macOS with:
  .venv/bin/mjpython double-pendulum/experiments/swingup_lqr_noisy/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from model import MJCF_MODEL
from policy import LQRPolicy, SwingUpLQRPolicy, SwingUpPolicy
from rollout import RolloutConfig, RolloutScenario, run_rollout


def main() -> int:
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

    Q = np.diag([8.5, 10.0, 10.0, 4.5, 1.5, 1.0])
    R = np.array([[1.0]])

    lqr_policy = LQRPolicy(
        Q=Q,
        R=R,
        control_limit=60.0,
        reference_state=np.zeros(6),
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

    noise_std = np.array([
        0.002,  # cart position [m]
        0.003,  # q1 [rad]
        0.003,  # q2_rel [rad]
        0.01,   # cart velocity [m/s]
        0.02,   # q1_dot [rad/s]
        0.02,   # q2_rel_dot [rad/s]
    ])

    # noise_std = np.array([0.005, 0.008, 0.008, 0.03, 0.05, 0.05])

    metrics = run_rollout(
        model_xml=MJCF_MODEL,
        policy=policy,
        scenario=RolloutScenario(
            init_qpos=np.array([0.0, np.deg2rad(180.0), np.deg2rad(0.0)]),
            init_qvel=np.array([0.0, 0.0, 0.0]),
            name="swingup_lqr_noisy",
        ),
        config=RolloutConfig(
            render=True,
            logging=True,
            sensor_noise_std=noise_std,
            sensor_noise_seed=0,
        ),
    )

    print(f"recovered: {metrics['recovered']}")
    print(f"settling_time: {metrics['settling_time']}")
    print(f"peak_cart_displacement: {metrics['peak_cart_displacement']:.4f}")
    print(f"total_control_effort: {metrics['total_control_effort']:.4f}")
    print(f"noise_std: {noise_std}")
    if policy.switch_step is not None:
        print(f"handoff_step: {policy.switch_step}")
    else:
        print("handoff_step: none")

    return 0


if __name__ == "__main__":
    sys.exit(main())
