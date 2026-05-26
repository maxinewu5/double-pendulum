"""
Render the same cart + double pendulum setup so the plots have a visual match.

Run this on macOS with:
  .venv/bin/mjpython double-pendulum/experiments/lqr/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from rollout import run_rollout, RolloutConfig, RolloutScenario
from model import MJCF_MODEL
from policy import LQRPolicy

def main() -> int:
    policy = LQRPolicy(
        Q=np.diag([8.5, 10.0, 10.0, 4.5, 1.5, 1.0]),
        R=np.array([[1.0]])
    )
    run_rollout(
        model_xml=MJCF_MODEL,
        policy=policy,  # LQR policy will be computed inside the rollout function
        scenario=RolloutScenario(
            # change the initial angles and velocities to be further from the upright position
            init_qpos=np.array([0.0, np.deg2rad(-12), np.deg2rad(-6)]),
            init_qvel=np.array([0.0, 0.0, 0.0]),
            name="lqr",
        ),
        config=RolloutConfig(render=True, logging=True),
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
