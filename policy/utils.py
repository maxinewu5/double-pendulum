from __future__ import annotations

import control as ct
import mujoco
import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi] so nearby poses stay numerically nearby."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angle_error(angle: float, reference: float) -> float:
    """Return the shortest signed angular difference angle - reference."""
    return wrap_angle(angle - reference)


def interpret_planar_state(state: np.ndarray) -> dict[str, float]:
    """
    Interpret MuJoCo's planar double-pendulum coordinates.

    state[1] is the absolute angle of link1 from vertical.
    state[2] is the relative joint angle of link2 with respect to link1.
    """
    q1 = wrap_angle(state[1])
    q2_rel = wrap_angle(state[2])
    q1_dot = state[4]
    q2_rel_dot = state[5]
    return {
        "x": state[0],
        "q1": q1,
        "q2_rel": q2_rel,
        "q2_abs": wrap_angle(q1 + q2_rel),
        "x_dot": state[3],
        "q1_dot": q1_dot,
        "q2_rel_dot": q2_rel_dot,
        "q2_abs_dot": q1_dot + q2_rel_dot,
    }


def compute_lqr_gain_at_reference(
    model,
    data,
    Q: np.ndarray,
    R: np.ndarray,
    reference_state: np.ndarray | None,
) -> np.ndarray:
    """
    Linearize the discrete dynamics at the requested reference state and solve LQR.
    """
    n_state = model.nq + model.nv
    n_input = model.nu
    A = np.zeros((n_state, n_state))
    B = np.zeros((n_state, n_input))
    C = np.zeros((0, n_state))
    D = np.zeros((0, n_input))

    original_qpos = np.copy(data.qpos)
    original_qvel = np.copy(data.qvel)
    original_act = np.copy(data.act) if model.na > 0 else None
    original_time = data.time

    if reference_state is None:
        qpos_ref = np.zeros(model.nq)
        qvel_ref = np.zeros(model.nv)
    else:
        qpos_ref = np.asarray(reference_state[: model.nq], dtype=float)
        qvel_ref = np.asarray(reference_state[model.nq : model.nq + model.nv], dtype=float)

    try:
        data.qpos[:] = qpos_ref
        data.qvel[:] = qvel_ref
        if model.na > 0:
            data.act[:] = 0.0
        data.time = 0.0
        mujoco.mj_forward(model, data)
        mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, C, D)
    finally:
        data.qpos[:] = original_qpos
        data.qvel[:] = original_qvel
        if model.na > 0 and original_act is not None:
            data.act[:] = original_act
        data.time = original_time
        mujoco.mj_forward(model, data)

    K, _, _ = ct.dlqr(A, B, Q, R)
    return K
