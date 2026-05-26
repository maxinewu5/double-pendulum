from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from model import MJCF_MODEL
from policy import SwingUpPolicy, interpret_planar_state, wrap_angle_error
from rollout import RolloutConfig, RolloutScenario, run_rollout
from search.grid_utils import run_grid_search, write_results_csv

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "search" / "swingup_grid"

CATCH_Q1_ERROR_LIMIT_RAD = 0.08
CATCH_Q2_ABS_ERROR_LIMIT_RAD = 0.08
CATCH_Q1_DOT_LIMIT = 1.5
CATCH_Q2_ABS_DOT_LIMIT = 1.5
MAX_HANDOFF_TIME_S = 8.0


def _score_best_near_upright_state(states: np.ndarray, reference_state: np.ndarray) -> dict[str, float | int]:
    """
    Find the timestep that jointly minimizes angle and velocity terms relative
    to the approximate handoff thresholds, then mark whether that best point is
    actually feasible for handoff.
    """
    reference_interp = interpret_planar_state(reference_state)
    best_index = 0
    best_normalized_score = float("inf")

    for idx, state in enumerate(states):
        interpreted = interpret_planar_state(state)
        q1_error = wrap_angle_error(interpreted["q1"], reference_interp["q1"])
        q2_abs_error = wrap_angle_error(interpreted["q2_abs"], reference_interp["q2_abs"])
        q1_error_abs = abs(q1_error)
        q2_abs_error_abs = abs(q2_abs_error)
        q1_dot_abs = abs(interpreted["q1_dot"])
        q2_abs_dot_abs = abs(interpreted["q2_abs_dot"])

        time_s = idx * 0.005
        normalized_score = (
            q1_error_abs / CATCH_Q1_ERROR_LIMIT_RAD
            + q2_abs_error_abs / CATCH_Q2_ABS_ERROR_LIMIT_RAD
            + q1_dot_abs / CATCH_Q1_DOT_LIMIT
            + q2_abs_dot_abs / CATCH_Q2_ABS_DOT_LIMIT
        )
        if time_s > MAX_HANDOFF_TIME_S:
            normalized_score += 1e6
        if normalized_score < best_normalized_score:
            best_normalized_score = normalized_score
            best_index = idx

    best_state = interpret_planar_state(states[best_index])
    q1_error = wrap_angle_error(best_state["q1"], reference_interp["q1"])
    q2_abs_error = wrap_angle_error(best_state["q2_abs"], reference_interp["q2_abs"])
    q1_error_abs = abs(q1_error)
    q2_abs_error_abs = abs(q2_abs_error)
    q1_dot_abs = abs(best_state["q1_dot"])
    q2_abs_dot_abs = abs(best_state["q2_abs_dot"])
    velocity_cost = q1_dot_abs + q2_abs_dot_abs
    angle_cost = q1_error_abs + q2_abs_error_abs
    handoff_feasible = (
        q1_error_abs <= CATCH_Q1_ERROR_LIMIT_RAD
        and q2_abs_error_abs <= CATCH_Q2_ABS_ERROR_LIMIT_RAD
        and q1_dot_abs <= CATCH_Q1_DOT_LIMIT
        and q2_abs_dot_abs <= CATCH_Q2_ABS_DOT_LIMIT
        and (best_index * 0.005) <= MAX_HANDOFF_TIME_S
    )

    return {
        "best_index": best_index,
        "best_time_s": best_index * 0.005,
        "q1_error_rad": q1_error_abs,
        "q2_abs_error_rad": q2_abs_error_abs,
        "q1_dot_abs": q1_dot_abs,
        "q2_abs_dot_abs": q2_abs_dot_abs,
        "velocity_cost": velocity_cost,
        "angle_cost": angle_cost,
        "combined_score": best_normalized_score,
        "handoff_feasible": handoff_feasible,
    }


def _evaluate_swingup_params(params: dict[str, float]) -> dict[str, float | int]:
    reference_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    policy = SwingUpPolicy(
        control_limit=30.0,
        initial_kick_force=params["initial_kick_force"],
        initial_kick_duration_s=params["initial_kick_duration_s"],
        x_target=params["x_target"],
        brake_margin=params["brake_margin"],
        launch_force=params["launch_force"],
        brake_force=params["brake_force"],
        hold_kp=params["hold_kp"],
        hold_kd=params["hold_kd"],
        apex_velocity_threshold=params["apex_velocity_threshold"],
    )

    metrics = run_rollout(
        model_xml=MJCF_MODEL,
        policy=policy,
        scenario=RolloutScenario(
            init_qpos=np.array([0.0, np.deg2rad(180), np.deg2rad(0.0)]),
            init_qvel=np.array([0.0, 0.0, 0.0]),
            name="swingup_grid",
        ),
        config=RolloutConfig(render=False, logging=False, max_steps=2500, stop_early=False),
    )

    states = np.asarray(metrics["state_history"], dtype=float)
    score_metrics = _score_best_near_upright_state(states, reference_state)
    return {
        **score_metrics,
        "peak_cart_displacement": metrics["peak_cart_displacement"],
        "total_control_effort": metrics["total_control_effort"],
    }


def main() -> int:
    coarse_sweep_grid = {
        "initial_kick_force": [10.0, 12.0, 15.0],
        "initial_kick_duration_s": [0.3, 0.5],
        "x_target": [0.4, 0.5, 0.6],
        "brake_margin": [0.3, 0.4, 0.5],
        "launch_force": [14.0, 16.0, 18.0],
        "brake_force": [15.0, 17.0],
        "hold_kp": [20.0],
        "hold_kd": [8.0],
        "apex_velocity_threshold": [0.2, 0.4],
    }
    param_grid_2 = {
        "initial_kick_force": [14.5, 14.75, 15.0, 15.25, 15.5],
        "initial_kick_duration_s": [0.475, 0.4875, 0.5, 0.5125, 0.525],
        "x_target": [0.39, 0.395, 0.4, 0.405, 0.41],
        "brake_margin": [0.48, 0.49, 0.5, 0.51, 0.52],
        "launch_force": [18.0],
        "brake_force": [15.0],
        # "launch_force": [17.5, 17.75, 18.0, 18.25, 18.5],
        # "brake_force": [14.5, 14.75, 15.0, 15.25, 15.5],
        "hold_kp": [20.0],
        "hold_kd": [8.0],
        "apex_velocity_threshold": [0.15, 0.175, 0.2, 0.225, 0.25],
    }

    param_grid_3 = {
    "initial_kick_force": [14.5, 15.0, 15.5],
    "initial_kick_duration_s": [0.5],
    "x_target": [0.36, 0.38, 0.4],
    "brake_margin": [0.5, 0.54, 0.58],
    "launch_force": [17.0, 17.5, 18.0],
    "brake_force": [15.0, 15.5, 16.0],
    "hold_kp": [20.0],
    "hold_kd": [8.0],
    "apex_velocity_threshold": [0.1, 0.15, 0.2],
}

    results = run_grid_search(
        param_grid=param_grid_3,
        evaluate_fn=_evaluate_swingup_params,
        progress_label="swingup",
    )
    feasible_results = [
        result for result in results if bool(result.metrics["handoff_feasible"])
    ]
    ranked_results = sorted(
        feasible_results,
        key=lambda result: float(result.metrics["combined_score"]),
    )
    infeasible_count = len(results) - len(feasible_results)
    all_results_ranked = sorted(
        results,
        key=lambda result: (
            not bool(result.metrics["handoff_feasible"]),
            float(result.metrics["combined_score"]),
        ),
    )

    write_results_csv(
        output_path=str(OUTPUT_DIR / "grid_results_swingup.csv"),
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
            "handoff_feasible",
            "peak_cart_displacement",
            "total_control_effort",
        ],
    )

    print(
        "\nCatch-region filter: "
        f"|q1_error| <= {CATCH_Q1_ERROR_LIMIT_RAD:.3f} rad, "
        f"|q2_abs_error| <= {CATCH_Q2_ABS_ERROR_LIMIT_RAD:.3f} rad, "
        f"|q1_dot| <= {CATCH_Q1_DOT_LIMIT:.3f} rad/s, "
        f"|q2_abs_dot| <= {CATCH_Q2_ABS_DOT_LIMIT:.3f} rad/s, "
        f"t <= {MAX_HANDOFF_TIME_S:.3f} s"
    )
    print(f"Feasible candidates written: {len(feasible_results)}/{len(results)}")
    print(f"Infeasible candidates dropped: {infeasible_count}")
    print("\nTop 10 swing-up parameter sets by filtered handoff score:")
    for rank, result in enumerate(ranked_results[:10], start=1):
        print(
            f"{rank}. params={result.params} "
            f"handoff_feasible={result.metrics['handoff_feasible']}, "
            f"combined_score={result.metrics['combined_score']:.4f}, "
            f"q1_dot_abs={result.metrics['q1_dot_abs']:.4f}, "
            f"q2_abs_dot_abs={result.metrics['q2_abs_dot_abs']:.4f}, "
            f"angle_cost={result.metrics['angle_cost']:.4f}"
        )

    if not ranked_results:
        print("\nNo feasible candidates met the handoff filter.")
        print("Closest infeasible candidates:")
        for rank, result in enumerate(all_results_ranked[:10], start=1):
            print(
                f"{rank}. params={result.params} "
                f"handoff_feasible={result.metrics['handoff_feasible']}, "
                f"combined_score={result.metrics['combined_score']:.4f}, "
                f"q1_error_rad={result.metrics['q1_error_rad']:.4f}, "
                f"q2_abs_error_rad={result.metrics['q2_abs_error_rad']:.4f}, "
                f"q1_dot_abs={result.metrics['q1_dot_abs']:.4f}, "
                f"q2_abs_dot_abs={result.metrics['q2_abs_dot_abs']:.4f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
