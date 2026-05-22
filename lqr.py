"""
Render the same cart + double pendulum setup so the plots have a visual match.

Run this on macOS with:
  .venv/bin/mjpython lqr.py
"""

from __future__ import annotations

import sys
import control as ct
import numpy as np
from rollout import run_rollout, RolloutConfig, RolloutScenario
from model import MJCF_MODEL
from policy import LQRPolicy

def main() -> int:
    policy = LQRPolicy(
        Q=np.diag([50.0, 10.0, 10.0, 5.0, 2.0, 2.0]), 
        R=np.array([[20.0]])
    )
    run_rollout(
        model_xml=MJCF_MODEL,
        policy=policy,  # LQR policy will be computed inside the rollout function
        scenario=RolloutScenario(
            # change the initial angles and velocities to be further from the upright position
            init_qpos=np.array([0.0, np.deg2rad(30), np.deg2rad(-30)]),
            init_qvel=np.array([0.0, 0.0, 0.0]),
            name="lqr",
        ),
        config=RolloutConfig(render=True, logging=True),
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
