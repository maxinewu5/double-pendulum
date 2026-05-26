from __future__ import annotations

import numpy as np

from rollout import RolloutScenario


def _scenario(
    name: str,
    *,
    x: float = 0.0,
    q1_deg: float = 0.0,
    q2_deg: float = 0.0,
    x_dot: float = 0.0,
    q1_dot: float = 0.0,
    q2_dot: float = 0.0,
) -> RolloutScenario:
    return RolloutScenario(
        name=name,
        init_qpos=np.array([x, np.deg2rad(q1_deg), np.deg2rad(q2_deg)]),
        init_qvel=np.array([x_dot, q1_dot, q2_dot]),
    )


def build_q1_angle_scenarios() -> list[RolloutScenario]:
    scenarios: list[RolloutScenario] = []
    for q1_deg in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
        scenarios.append(_scenario(f"q1_pos_{q1_deg:.0f}deg", q1_deg=q1_deg))
        scenarios.append(_scenario(f"q1_neg_{q1_deg:.0f}deg", q1_deg=-q1_deg))
    return scenarios


def build_combined_angle_scenarios() -> list[RolloutScenario]:
    scenarios: list[RolloutScenario] = []
    angle_pairs_deg = [
        (2.0, 2.0),
        (4.0, 2.0),
        (4.0, 4.0),
        (6.0, 4.0),
        (6.0, 6.0),
        (8.0, 4.0),
        (8.0, 6.0),
    ]
    for q1_deg, q2_deg in angle_pairs_deg:
        scenarios.append(_scenario(f"q1p{q1_deg:.0f}_q2p{q2_deg:.0f}", q1_deg=q1_deg, q2_deg=q2_deg))
        scenarios.append(_scenario(f"q1p{q1_deg:.0f}_q2n{q2_deg:.0f}", q1_deg=q1_deg, q2_deg=-q2_deg))
        scenarios.append(_scenario(f"q1n{q1_deg:.0f}_q2p{q2_deg:.0f}", q1_deg=-q1_deg, q2_deg=q2_deg))
        scenarios.append(_scenario(f"q1n{q1_deg:.0f}_q2n{q2_deg:.0f}", q1_deg=-q1_deg, q2_deg=-q2_deg))
    return scenarios


def build_catch_like_scenarios() -> list[RolloutScenario]:
    return [
        _scenario("catch_like_1", q1_deg=5.0, q2_deg=-3.0, q1_dot=0.50, q2_dot=-0.35),
        _scenario("catch_like_2", q1_deg=-5.0, q2_deg=3.0, q1_dot=-0.50, q2_dot=0.35),
        _scenario("catch_like_3", q1_deg=6.0, q2_deg=4.0, q1_dot=0.75, q2_dot=-0.50),
        _scenario("catch_like_4", q1_deg=-6.0, q2_deg=-4.0, q1_dot=-0.75, q2_dot=0.50),
        _scenario("catch_like_5", q1_deg=8.0, q2_deg=-5.0, q1_dot=1.00, q2_dot=-0.60),
        _scenario("catch_like_6", q1_deg=-8.0, q2_deg=5.0, q1_dot=-1.00, q2_dot=0.60),
        _scenario("catch_like_7", q1_deg=4.0, q2_deg=2.0, x=0.10, q1_dot=0.60, q2_dot=-0.40),
        _scenario("catch_like_8", q1_deg=-4.0, q2_deg=-2.0, x=-0.10, q1_dot=-0.60, q2_dot=0.40),
    ]


def build_validation_scenarios() -> list[RolloutScenario]:
    scenarios: list[RolloutScenario] = []
    scenarios.extend(build_q1_angle_scenarios())
    scenarios.extend(build_combined_angle_scenarios())
    scenarios.extend(build_catch_like_scenarios())
    return scenarios
