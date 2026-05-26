"""
Visualize the zero-velocity LQR angle basin as a 2D map over initial (q1, q2).

Run with:
  python double-pendulum/validate/angle_basin.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import MJCF_MODEL
from policy import LQRPolicy
from rollout import RolloutConfig, RolloutScenario, run_rollout

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validate"
LQR_Q_DIAG = [15.0, 10.0, 10.0, 5.0, 2.0, 2.0]
LQR_R_VALUE = 20.0
EFFORT_PLOT_MAX = 1000.0

LQR_Q_DIAG = [8.5, 10.0, 10.0, 4.5, 1.5, 1.0]
LQR_R_VALUE = 1.0


def _build_angle_basin_scenarios(q1_deg_values: list[float], q2_deg_values: list[float]) -> list[RolloutScenario]:
    scenarios: list[RolloutScenario] = []
    for q1_deg in q1_deg_values:
        for q2_deg in q2_deg_values:
            scenarios.append(
                RolloutScenario(
                    name=f"q1_{q1_deg:+.0f}_q2_{q2_deg:+.0f}",
                    init_qpos=np.array([0.0, np.deg2rad(q1_deg), np.deg2rad(q2_deg)]),
                    init_qvel=np.zeros(3),
                )
            )
    return scenarios


def _write_results_csv(results: list[dict[str, float | bool]], output_path: Path) -> None:
    fieldnames = [
        "q1_deg",
        "q2_deg",
        "recovered",
        "settling_time",
        "total_control_effort",
        "peak_angle_error",
        "peak_cart_displacement",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def _load_results_csv(input_path: Path) -> list[dict[str, float | bool]]:
    results: list[dict[str, float | bool]] = []
    with input_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            settling_time = row["settling_time"]
            results.append(
                {
                    "q1_deg": float(row["q1_deg"]),
                    "q2_deg": float(row["q2_deg"]),
                    "recovered": row["recovered"] == "True",
                    "settling_time": np.nan if settling_time in {"", "nan", "None"} else float(settling_time),
                    "total_control_effort": float(row["total_control_effort"]),
                    "peak_angle_error": float(row["peak_angle_error"]),
                    "peak_cart_displacement": float(row["peak_cart_displacement"]),
                }
            )
    return results


def _plot_heatmap(
    grid: np.ndarray,
    q1_deg_values: list[float],
    q2_deg_values: list[float],
    *,
    title: str,
    colorbar_label: str,
    output_path: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    tick_step_deg: int = 2,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[min(q1_deg_values), max(q1_deg_values), min(q2_deg_values), max(q2_deg_values)],
    )
    ax.set_xlabel("Initial q1 [deg]")
    ax.set_ylabel("Initial q2 [deg]")
    ax.set_title(title)
    ax.set_xticks(list(range(int(min(q1_deg_values)), int(max(q1_deg_values)) + 1, tick_step_deg)))
    ax.set_yticks(list(range(int(min(q2_deg_values)), int(max(q2_deg_values)) + 1, tick_step_deg)))
    ax.grid(False)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_grids_from_results(
    results: list[dict[str, float | bool]],
    q1_deg_values: list[float],
    q2_deg_values: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    success_grid = np.zeros((len(q2_deg_values), len(q1_deg_values)), dtype=float)
    settling_grid = np.full((len(q2_deg_values), len(q1_deg_values)), np.nan, dtype=float)
    effort_grid = np.full((len(q2_deg_values), len(q1_deg_values)), np.nan, dtype=float)

    for result in results:
        q1_deg = float(result["q1_deg"])
        q2_deg = float(result["q2_deg"])
        q1_idx = q1_deg_values.index(int(round(q1_deg)))
        q2_idx = q2_deg_values.index(int(round(q2_deg)))
        recovered = bool(result["recovered"])
        settling_time = result["settling_time"]

        success_grid[q2_idx, q1_idx] = 1.0 if recovered else 0.0
        if recovered and not np.isnan(settling_time):
            settling_grid[q2_idx, q1_idx] = float(settling_time)
            effort_grid[q2_idx, q1_idx] = float(result["total_control_effort"])

    return success_grid, settling_grid, effort_grid


def _render_plots_from_results(
    results: list[dict[str, float | bool]],
    q1_deg_values: list[float],
    q2_deg_values: list[float],
) -> None:
    success_grid, settling_grid, effort_grid = _build_grids_from_results(results, q1_deg_values, q2_deg_values)
    lqr_caption = (
        "LQR parameters: "
        f"Q=diag({LQR_Q_DIAG}), R={LQR_R_VALUE:.1f}"
    )

    _plot_heatmap(
        success_grid,
        q1_deg_values,
        q2_deg_values,
        title=f"LQR Zero-Velocity Angle Basin\n{lqr_caption}",
        colorbar_label="Recovered (1=True, 0=False)",
        output_path=OUTPUT_DIR / "lqr_angle_basin_success.png",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        tick_step_deg=2,
    )
    _plot_heatmap(
        settling_grid,
        q1_deg_values,
        q2_deg_values,
        title=f"LQR Zero-Velocity Settling Time\n{lqr_caption}",
        colorbar_label="Settling time [s]",
        output_path=OUTPUT_DIR / "lqr_angle_basin_settling.png",
        cmap="viridis",
        tick_step_deg=2,
    )
    # _plot_heatmap(
    #     np.where(effort_grid <= EFFORT_PLOT_MAX, effort_grid, np.nan),
    #     q1_deg_values,
    #     q2_deg_values,
    #     title=f"LQR Zero-Velocity Control Effort\n{lqr_caption}",
    #     colorbar_label=f"Total control effort (settled cases only, <= {EFFORT_PLOT_MAX:.0f})",
    #     output_path=OUTPUT_DIR / "lqr_angle_basin_effort.png",
    #     cmap="magma",
    #     tick_step_deg=2,
    # )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    q1_deg_values = list(range(-12, 13, 1))
    q2_deg_values = list(range(-12, 13, 1))
    csv_path = OUTPUT_DIR / "lqr_angle_basin_results.csv"

    if len(sys.argv) > 1 and sys.argv[1] == "--replot":
        results = _load_results_csv(csv_path)
        _render_plots_from_results(results, q1_deg_values, q2_deg_values)
        print(f"re-rendered angle basin plots from {csv_path}")
        return 0

    scenarios = _build_angle_basin_scenarios(q1_deg_values, q2_deg_values)

    Q = np.diag(LQR_Q_DIAG)
    R = np.array([[LQR_R_VALUE]])
    policy = LQRPolicy(Q, R, control_limit=30.0)
    config = RolloutConfig(render=False, logging=False)

    results: list[dict[str, float | bool]] = []

    for scenario in scenarios:
        q1_deg = float(np.rad2deg(scenario.init_qpos[1]))
        q2_deg = float(np.rad2deg(scenario.init_qpos[2]))
        metrics = run_rollout(MJCF_MODEL, policy, scenario, config)

        results.append(
            {
                "q1_deg": q1_deg,
                "q2_deg": q2_deg,
                "recovered": bool(metrics["recovered"]),
                "settling_time": np.nan if metrics["settling_time"] is None else float(metrics["settling_time"]),
                "total_control_effort": float(metrics["total_control_effort"]),
                "peak_angle_error": float(metrics["peak_angle_error"]),
                "peak_cart_displacement": float(metrics["peak_cart_displacement"]),
            }
        )
        print(
            f"(q1={q1_deg:+.0f} deg, q2={q2_deg:+.0f} deg): "
            f"recovered={metrics['recovered']}, settling_time={metrics['settling_time']}"
        )

    _write_results_csv(results, csv_path)
    _render_plots_from_results(results, q1_deg_values, q2_deg_values)

    recovered_count = sum(1 for result in results if bool(result["recovered"]))
    print(
        f"\nRecovered {recovered_count} / {len(scenarios)} zero-velocity angle scenarios. "
        f"Saved outputs to {OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
