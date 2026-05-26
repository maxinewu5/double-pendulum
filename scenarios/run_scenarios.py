"""
Validate a controller policy across a set of rollout scenarios.

Run with 'python run_scenarios.py' and it will execute a set of rollouts across different 
initial conditions and log the results to a CSV file for analysis.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from model import MJCF_MODEL
from policy import LQRPolicy
from rollout import RolloutConfig, run_rollout
from scenarios import build_validation_scenarios

def main() -> int:
    # define the LQR policy to validate
    Q = np.diag([15.0, 10.0, 10.0, 5.0, 2.0, 2.0])
    R = np.array([[20.0]])
    policy = LQRPolicy(Q, R, control_limit=30.0)

    scenarios = build_validation_scenarios()
    config = RolloutConfig(render=False, logging=False)
    results = []

    for scenario in scenarios:
        metrics = run_rollout(MJCF_MODEL, policy, scenario, config)
        result = {
            "scenario_name": scenario.name,
            **metrics,
        }
        results.append(result)

        print(
            f"{scenario.name}: "
            f"recovered={metrics['recovered']}, "
            f"settling_time={metrics['settling_time']}, "
            f"effort={metrics['total_control_effort']:.2f}, "
            f"peak_angle_error={metrics['peak_angle_error']:.3f}, "
            f"peak_cart_displacement={metrics['peak_cart_displacement']:.3f}"
        )

    write_results_csv(results, "scenarios/validation_results.csv")

    recovered_results = [result for result in results if result["recovered"]]
    print(
        f"\nRecovered {len(recovered_results)} / {len(results)} scenarios. "
        f"Results written to validation_results.csv"
    )

    return 0

def write_results_csv(results: list[dict], output_path: str) -> None:
    fieldnames = [
        "scenario_name",
        "recovered",
        "settling_time",
        "total_control_effort",
        "peak_angle_error",
        "peak_cart_displacement",
    ]

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fieldnames})

if __name__ == "__main__":
    sys.exit(main())
