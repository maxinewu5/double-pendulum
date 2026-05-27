"""
Shared rollout utilities for evaluating LQR controllers on the MuJoCo model.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from policy import Policy
from sensor import SensorModel

try:
    import mujoco
    from mujoco import viewer
except ImportError:  # pragma: no cover - depends on local environment
    mujoco = None
    viewer = None

@dataclass
class RolloutScenario:
    # defines the initial condition / test case for a rollout
    init_qpos: np.ndarray
    init_qvel: np.ndarray
    name: str = "default"

@dataclass
class RolloutConfig:
    # defines how the rollout is executed and when it is considered settled
    render: bool = False
    logging: bool = False
    max_steps: int = 2000
    angle_tolerance_rad: float = 0.05
    angular_velocity_tolerance: float = 0.1
    settle_hold_time: float = 1.0
    stop_early: bool = False
    sensor_noise_std: np.ndarray | None = None
    sensor_noise_seed: int | None = None


def run_rollout(model_xml, policy: Policy, scenario: RolloutScenario, config: RolloutConfig | None = None):
    if config is None:
        config = RolloutConfig()
    if mujoco is None:
        raise SystemExit("Install MuJoCo first with: pip install mujoco")
    if config.render and viewer is None:
        raise SystemExit("MuJoCo viewer is unavailable in this environment")

    # setup the model from the xml string and create a data instance that holds state
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)

    # set initial state of system
    data.qpos[:] = scenario.init_qpos
    data.qvel[:] = scenario.init_qvel

    # call forward to compute any derived quantities
    mujoco.mj_forward(model, data)

    first_settled_step = None
    settling_time = None
    recovered = False

    control_history = []
    total_control_effort = 0.0
    state_history = []
    max_angle_error = 0.0
    peak_cart_displacement = 0.0
    measured_state_history = []

    sensor = SensorModel(
        noise_std=config.sensor_noise_std,
        seed=config.sensor_noise_seed,
    )

    # initialize the control policy for this specific MuJoCo model/state
    policy.initialize(model, data)

    if config.render:
        rollout_context = viewer.launch_passive(model, data)
    else:
        rollout_context = None

    # setup optional viewer context to render the rollout
    if rollout_context is None:
        sim_viewer = None
        viewer_running = True
        context_manager = None
    else:
        context_manager = rollout_context.__enter__()
        sim_viewer = context_manager
        sim_viewer.cam.distance = 8.0
        sim_viewer.cam.azimuth = 90
        sim_viewer.cam.elevation = -15
        sim_viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        viewer_running = True

    try:
        for step in range(config.max_steps):
            if sim_viewer is not None and not sim_viewer.is_running():
                viewer_running = False
                break

            # x: the full state vector of the system
            # x = [cart_pos, theta1, theta2, cart_vel, theta1_vel, theta2_vel]
            x = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            measured_x = sensor.measure(x)
            state_history.append(x)
            measured_state_history.append(measured_x)
            max_angle_error = max(max_angle_error, abs(x[1]), abs(x[2]))
            peak_cart_displacement = max(peak_cart_displacement, abs(x[0]))

            # compute whether the system is settled based on whether the angles are upright
            # and angular velocity is low
            is_settled = (
                abs(x[1]) < config.angle_tolerance_rad
                and abs(x[2]) < config.angle_tolerance_rad
                and abs(x[4]) < config.angular_velocity_tolerance
                and abs(x[5]) < config.angular_velocity_tolerance
            )
            if not recovered and is_settled:
                if first_settled_step is None:
                    first_settled_step = step
                else:
                    if (step - first_settled_step) * model.opt.timestep >= config.settle_hold_time:
                        settling_time = step * model.opt.timestep
                        recovered = True
                        if config.stop_early:
                            break
            else:
                first_settled_step = None

            # ask the active control policy for the input to apply at this state
            u = np.asarray(policy.compute_control(measured_x), dtype=float).reshape(-1)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)

            # keep the signed applied actuator command for diagnostics/plotting
            control_history.append(float(data.ctrl[0]))
            total_control_effort += float(np.dot(data.ctrl, data.ctrl)) * model.opt.timestep

            # log step and control data as needed
            if config.logging: 
                if step % 10 == 0:
                    print(
                        f"step {step}: "
                        f"t={step * model.opt.timestep: .4f}, "
                        f"x={data.qpos[0]: .4f}, "
                        f"theta1={data.qpos[1]: .4f}, "
                        f"theta2={data.qpos[2]: .4f}, "
                        f"theta1dot={data.qvel[1]: .4f}, "
                        f"theta2dot={data.qvel[2]: .4f}, "
                        f"u={data.ctrl[0]: .4f}"
                    )

            # advance the sim as needed
            if sim_viewer is not None:
                sim_viewer.sync()
                time.sleep(model.opt.timestep)
    finally:
        if rollout_context is not None:
            rollout_context.__exit__(None, None, None)

    return {
        "recovered": recovered,
        "settling_time": settling_time,
        "peak_angle_error": max_angle_error,
        "total_control_effort": total_control_effort,
        "peak_cart_displacement": peak_cart_displacement,
        "viewer_closed": not viewer_running,
        "state_history": state_history,
        "measured_state_history": measured_state_history,
        "control_history": control_history,
    }
