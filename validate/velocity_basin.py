"""
Visualize LQR velocity tolerance at a few fixed angle anchors.

Run with:
  python double-pendulum/validate/velocity_basin.py
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
LQR_Q_DIAG = [8.5, 10.0, 10.0, 4.5, 1.5, 1.0]
LQR_R_VALUE = 1.0
EFFORT_PLOT_MAX = 1000.0

# Representative zero-velocity angle anchors near the upright basin.
ANGLE_ANCHORS_DEG = [
    (2.0, 2.0),
    (3.0, 3.0),
    (4.0, 4.0),
    (5.0, 5.0),
    (5.0, -5.0),
]


def _build_velocity_scenarios(
    q1_deg_anchor: float,
    q2_deg_anchor: float,
    q1_dot_values: list[float],
    q2_dot_values: list[float],
) -> list[RolloutScenario]:
    scenarios: list[RolloutScenario] = []
    for q1_dot in q1_dot_values:
        for q2_dot in q2_dot_values:
            scenarios.append(
                RolloutScenario(
                    name=f"q1a_{q1_deg_anchor:+.0f}_q2a_{q2_deg_anchor:+.0f}_q1d_{q1_dot:+.2f}_q2d_{q2_dot:+.2f}",
                    init_qpos=np.array([0.0, np.deg2rad(q1_deg_anchor), np.deg2rad(q2_deg_anchor)]),
                    init_qvel=np.array([0.0, q1_dot, q2_dot]),
                )
            )
    return scenarios


def _write_results_csv(results: list[dict[str, float | bool]], output_path: Path) -> None:
    fieldnames = [
        "q1_anchor_deg",
        "q2_anchor_deg",
        "q1_dot",
        "q2_dot",
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


def _plot_heatmap(
    grid: np.ndarray,
    q1_dot_values: list[float],
    q2_dot_values: list[float],
    *,
    title: str,
    colorbar_label: str,
    output_path: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    tick_step: float = 0.5,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[min(q1_dot_values), max(q1_dot_values), min(q2_dot_values), max(q2_dot_values)],
    )
    ax.set_xlabel("Initial q1_dot [rad/s]")
    ax.set_ylabel("Initial q2_dot [rad/s]")
    ax.set_title(title)
    q1_ticks = np.arange(min(q1_dot_values), max(q1_dot_values) + 1e-9, tick_step)
    q2_ticks = np.arange(min(q2_dot_values), max(q2_dot_values) + 1e-9, tick_step)
    ax.set_xticks(q1_ticks)
    ax.set_yticks(q2_ticks)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    q1_dot_values = list(np.round(np.arange(-2.0, 2.01, 0.25), 2))
    q2_dot_values = list(np.round(np.arange(-2.0, 2.01, 0.25), 2))

    Q = np.diag(LQR_Q_DIAG)
    R = np.array([[LQR_R_VALUE]])
    policy = LQRPolicy(Q, R, control_limit=30.0)
    config = RolloutConfig(render=False, logging=False)
    lqr_caption = f"Q=diag({LQR_Q_DIAG}), R={LQR_R_VALUE:.1f}"

    all_results: list[dict[str, float | bool]] = []

    for q1_anchor_deg, q2_anchor_deg in ANGLE_ANCHORS_DEG:
        scenarios = _build_velocity_scenarios(q1_anchor_deg, q2_anchor_deg, q1_dot_values, q2_dot_values)
        success_grid = np.zeros((len(q2_dot_values), len(q1_dot_values)), dtype=float)
        settling_grid = np.full((len(q2_dot_values), len(q1_dot_values)), np.nan, dtype=float)
        effort_grid = np.full((len(q2_dot_values), len(q1_dot_values)), np.nan, dtype=float)

        for scenario in scenarios:
            metrics = run_rollout(MJCF_MODEL, policy, scenario, config)
            q1_dot = float(scenario.init_qvel[1])
            q2_dot = float(scenario.init_qvel[2])
            q1_idx = q1_dot_values.index(round(q1_dot, 2))
            q2_idx = q2_dot_values.index(round(q2_dot, 2))

            success_grid[q2_idx, q1_idx] = 1.0 if metrics["recovered"] else 0.0
            if metrics["recovered"] and metrics["settling_time"] is not None:
                settling_grid[q2_idx, q1_idx] = float(metrics["settling_time"])
                if metrics["total_control_effort"] <= EFFORT_PLOT_MAX:
                    effort_grid[q2_idx, q1_idx] = float(metrics["total_control_effort"])

            all_results.append(
                {
                    "q1_anchor_deg": q1_anchor_deg,
                    "q2_anchor_deg": q2_anchor_deg,
                    "q1_dot": q1_dot,
                    "q2_dot": q2_dot,
                    "recovered": bool(metrics["recovered"]),
                    "settling_time": np.nan if metrics["settling_time"] is None else float(metrics["settling_time"]),
                    "total_control_effort": float(metrics["total_control_effort"]),
                    "peak_angle_error": float(metrics["peak_angle_error"]),
                    "peak_cart_displacement": float(metrics["peak_cart_displacement"]),
                }
            )

        anchor_slug = f"q1_{q1_anchor_deg:+.0f}_q2_{q2_anchor_deg:+.0f}".replace("+", "p").replace("-", "n")
        title_prefix = f"LQR Velocity Basin at (q1={q1_anchor_deg:+.0f} deg, q2={q2_anchor_deg:+.0f} deg)"
        _plot_heatmap(
            success_grid,
            q1_dot_values,
            q2_dot_values,
            title=f"{title_prefix}\n{lqr_caption}",
            colorbar_label="Recovered (1=True, 0=False)",
            output_path=OUTPUT_DIR / f"lqr_velocity_basin_success_{anchor_slug}.png",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            tick_step=0.5,
        )
        _plot_heatmap(
            settling_grid,
            q1_dot_values,
            q2_dot_values,
            title=f"{title_prefix} Settling Time\n{lqr_caption}",
            colorbar_label="Settling time [s]",
            output_path=OUTPUT_DIR / f"lqr_velocity_basin_settling_{anchor_slug}.png",
            cmap="viridis",
            tick_step=0.5,
        )
        # _plot_heatmap(
        #     effort_grid,
        #     q1_dot_values,
        #     q2_dot_values,
        #     title=f"{title_prefix} Control Effort\n{lqr_caption}",
        #     colorbar_label=f"Total control effort (settled cases only, <= {EFFORT_PLOT_MAX:.0f})",
        #     output_path=OUTPUT_DIR / f"lqr_velocity_basin_effort_{anchor_slug}.png",
        #     cmap="magma",
        #     tick_step=0.5,
        # )
        recovered_count = int(np.nansum(success_grid))
        print(
            f"(q1={q1_anchor_deg:+.0f} deg, q2={q2_anchor_deg:+.0f} deg): "
            f"recovered {recovered_count} / {len(scenarios)} velocity scenarios"
        )

    _write_results_csv(all_results, OUTPUT_DIR / "lqr_velocity_basin_results.csv")
    print(f"\nSaved velocity-basin outputs to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
