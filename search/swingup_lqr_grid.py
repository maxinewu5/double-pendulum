from __future__ import annotations

import sys
import numpy as np

from model import MJCF_MODEL
from policy import LQRPolicy, SwingUpLQRPolicy, SwingUpPolicy, interpret_planar_state, wrap_angle_error
from rollout import RolloutConfig, RolloutScenario, run_rollout
from search.grid_utils import GridSearchResult, run_grid_search, write_results_csv


def _score_best_near_upright_state(
    states: np.ndarray,
    reference_state: np.ndarray,
) -> dict[str, float | int]:
    """
    Score the rollout by finding the timestep whose two-link upright angle error is
    smallest, then measuring angular velocities at that instant.
    """
    reference_interp = interpret_planar_state(reference_state)
    best_index = 0
    best_angle_cost = float("inf")

    for idx, state in enumerate(states):
        interpreted = interpret_planar_state(state)
        q1_error = wrap_angle_error(interpreted["q1"], reference_interp["q1"])
        q2_abs_error = wrap_angle_error(interpreted["q2_abs"], reference_interp["q2_abs"])
        angle_cost = abs(q1_error) + abs(q2_abs_error)
        if angle_cost < best_angle_cost:
            best_angle_cost = angle_cost
            best_index = idx

    best_state = interpret_planar_state(states[best_index])
    q1_error = wrap_angle_error(best_state["q1"], reference_interp["q1"])
    q2_abs_error = wrap_angle_error(best_state["q2_abs"], reference_interp["q2_abs"])
    velocity_cost = abs(best_state["q1_dot"]) + abs(best_state["q2_abs_dot"])
    combined_score = velocity_cost + 0.25 * (abs(q1_error) + abs(q2_abs_error))

    return {
        "best_index": best_index,
        "best_time_s": best_index * 0.005,
        "q1_error_rad": abs(q1_error),
        "q2_abs_error_rad": abs(q2_abs_error),
        "q1_dot_abs": abs(best_state["q1_dot"]),
        "q2_abs_dot_abs": abs(best_state["q2_abs_dot"]),
        "velocity_cost": velocity_cost,
        "angle_cost": best_angle_cost,
        "combined_score": combined_score,
    }


def _evaluate_swingup_lqr_params(params: dict[str, float]) -> dict[str, float | int | bool]:
    reference_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    swingup_policy = SwingUpPolicy(
        control_limit=30.0,
        initial_kick_force=15.0,
        initial_kick_duration_s=0.5,
        x_target=params["x_target"],
        brake_margin=params["brake_margin"],
        launch_force=params["launch_force"],
        brake_force=params["brake_force"],
    )
    lqr_policy = LQRPolicy(
        Q=np.diag([50.0, 10.0, 15.0, 7.0, 10.0, 60.0]),
        R=np.array([[20.0]]),
        control_limit=30.0,
        reference_state=reference_state,
    )
    policy = SwingUpLQRPolicy(
        swingup_policy=swingup_policy,
        lqr_policy=lqr_policy,
        q1_capture_deg=params["q1_capture_deg"],
        q2_capture_deg=params["q2_capture_deg"],
        q1_dot_capture=params["q1_dot_capture"],
        q2_abs_dot_capture=params["q2_abs_dot_capture"],
        x_capture_limit=params["x_capture_limit"],
    )

    metrics = run_rollout(
        model_xml=MJCF_MODEL,
        policy=policy,
        scenario=RolloutScenario(
            init_qpos=np.array([0.0, np.deg2rad(180), np.deg2rad(0.0)]),
            init_qvel=np.array([0.0, 0.0, 0.0]),
            name="swingup_lqr_grid",
        ),
        config=RolloutConfig(render=False, logging=False, max_steps=2500, stop_early=False),
    )

    states = np.asarray(metrics["state_history"], dtype=float)
    score_metrics = _score_best_near_upright_state(states, reference_state)
    return {
        **score_metrics,
        "switched_to_lqr": policy.switch_step is not None,
        "switch_time_s": -1.0 if policy.switch_step is None else policy.switch_step * 0.005,
        "peak_cart_displacement": metrics["peak_cart_displacement"],
        "total_control_effort": metrics["total_control_effort"],
    }


def main() -> int:
    param_grid = {
        "x_target": [0.4, 0.5, 0.6],
        "brake_margin": [0.3, 0.4, 0.5],
        "launch_force": [14.0, 16.0, 18.0],
        "brake_force": [15.0, 17.0],
        "q1_capture_deg": [5.0],
        "q2_capture_deg": [5.0],
        "q1_dot_capture": [1.5, 2.0],
        "q2_abs_dot_capture": [1.5, 2.0],
        "x_capture_limit": [2.0],
    }

    results = run_grid_search(
        param_grid=param_grid,
        evaluate_fn=_evaluate_swingup_lqr_params,
        progress_label="swingup_lqr",
    )
    ranked_results = sorted(results, key=lambda result: result.metrics["combined_score"])

    write_results_csv(
        output_path="double-pendulum/grid_results_swingup_lqr.csv",
        results=ranked_results,
        metric_keys=[
            "best_index",
            "best_time_s",
            "q1_error_rad",
            "q2_abs_error_rad",
            "q1_dot_abs",
            "q2_abs_dot_abs",
            "velocity_cost",
            "angle_cost",
            "combined_score",
            "switched_to_lqr",
            "switch_time_s",
            "peak_cart_displacement",
            "total_control_effort",
        ],
    )

    print("\nTop 10 parameter sets by combined near-upright score:")
    for rank, result in enumerate(ranked_results[:10], start=1):
        print(
            f"{rank}. params={result.params} "
            f"combined_score={result.metrics['combined_score']:.4f}, "
            f"q1_dot_abs={result.metrics['q1_dot_abs']:.4f}, "
            f"q2_abs_dot_abs={result.metrics['q2_abs_dot_abs']:.4f}, "
            f"angle_cost={result.metrics['angle_cost']:.4f}, "
            f"switched_to_lqr={result.metrics['switched_to_lqr']}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
