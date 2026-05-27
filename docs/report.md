# Inverted Double Pendulum on Cart Swing-Up and Balance

## Introduction

The goal of this project was to control a cart-mounted double pendulum in simulation and drive it to the upright position using only a horizontal force applied to the cart. I implemented a two-phase controller: a swing-up policy from rest followed by LQR stabilization near the upright equilibrium. 

## Model and Environment

The system was modeled in MuJoCo as a planar cart-double-pendulum with a single horizontal actuator applied to the cart. The system state and control input are defined as follows:

$$
x =
\begin{bmatrix}
x_{cart} & \theta_1 & \theta_2 & \dot{x}_{cart} & \dot{\theta}_1 & \dot{\theta}_2
\end{bmatrix}^T.
$$

$$
u = \begin{bmatrix} F_{cart} \end{bmatrix}.
$$

<p align="center">
  <img src="assets/system_state_diagram.png" alt="State diagram for the cart-double-pendulum system, showing the cart position, input force, and the two link angles." width="80%">
</p>

*Figure 1. Planar cart-double-pendulum model and state definition.*

> Note: The second joint angle $\theta_2$ is defined relative to link 1 rather than in the world frame, following MuJoCo's joint convention.

### Assumptions

The simulation assumed planar rigid-body dynamics, no out-of-plane dynamics, full state availability, and no sensor noise or external disturbances. These assumptions simplify the controller design and validation in simulation, though the simulated performance will likely overestimate the true real-world performance.

## Controller Design

A key design decision was to split the controller into separate swing-up and stabilization phases. Near the upright equilibrium, LQR is appropriate, but from the hanging rest position the system must first build momentum through a different control strategy. This also introduces a handoff problem: the swing-up controller must bring the system close enough to upright for the local stabilizer to take over successfully.

### Stabilization Phase

For the stabilization near upright phase, I chose an LQR (linear quadratic regulator) controller. LQR is a good fit because near the upright equilibrium, the angles and velocities are small enough that the pendulum's nonlinear dynamics can be approximated by a linear model. Additionally, the controller must regulate multiple coupled state errors, such as the motion of both pendulum links, with one actuator. 

Although the controller works well close to the local upright equilibrium, when the pendulum angles or angular velocities are too large, the linear approximation of the system will no longer hold, so the performance of the LQR controller will become worse when the state is far from the equilibrium.

### Swing-Up Policy

The swing-up controller was more challenging to design for than the near-upright stabilization controller because the controller had to build enough momentum from rest to reach the upright region while staying on the rails and keeping the motion of pendulum link under control. To approach this problem, I several different swing-up strategies with varying levels of feedback.

#### 1. Phase-based Swing Up
I first tried a phase-based swing-up controller that timed cart forces with the pendulum motion to increase oscillation amplitude. In practice, this was difficult to tune because forces that were helpful for one link could still destabilize the coupled motion of the full double pendulum.

#### 2. Energy-based Swing Up
I then tested an energy-based swing-up controller that compared the system's mechanical energy to the upright target energy and used cart motion to add or remove energy as needed. Although this approach was physically intuitive, it was difficult to tune reliably because the effect of a cart force depended on both the system energy and the phase of the two links, so energy could still be injected at unhelpful times.

#### 3. Cart Motion Swing Up

Lastly, I explored a more heuristic controller based on a left-right cart motion was also explored. This controller used a state machine to move the cart from left to right, braking and reversing at controlled times. The main modes were `drive_right`, `brake_right`, `hold_right`, `drive_left`, `brake_left`, and `hold_left`. This design was motivated by a simple physical intuition: move the cart decisively, stop within the rail limit, let the pendulum reach its apex, and reverse at useful times while staying within the rail bounds.

In practice, this approach was significantly easier to tune than the earlier swing-up attempts. Its behavior was also more interpretable in plots because each controller mode had a clear purpose and failure modes were easier to troubleshoot. This controller also tended to produce more stable control and structured cart motion, which helped keep the second link from becoming too chaotic. Although it is less principled and more brittle than a feedback-driven energy-based design, it was easier to visualize and tune, and I was able to successfully develop a version of this controller that produced a swing-up trajectory.

## Results

This section evaluates the local LQR stabilizer and swing-up controller. The results show that the LQR controller provided robust stabilization near the upright equilibrium under a variety of angular and angular velocity conditions. However, the overall system was limited by the less robust swing-up stage and the quality of the handoff into the LQR capture region.

### LQR Stabilization Results

After tuning the weighting matrices $Q$ and $R$, the LQR controller consistently stabilized the pendulum from a broad range of small and moderate perturbations near the upright equilibrium. Increasing the angular state penalties improved recovery speed but also increased control effort, while penalizing cart position helped keep the cart within the rail limits.

To characterize this local basin of attraction in which the LQR controller can drive the system back to equilibrium, I first tested the final controller over a grid of initial angle perturbations with zero initial angular velocity.

<p align="center">
  <img src="../outputs_example/validate/lqr_angle_basin_settling.png" alt="Settling-time map for the zero-velocity LQR angle basin." width="70%">
</p>

*Figure 3. Zero-velocity angle basin for the final LQR controller.*

Across the full grid from $-12^\circ$ to $+12^\circ$ in both $q_1$ and $q_2$, 577 of 625 initial conditions were recovered successfully, corresponding to about $92.3\%$ success. In addition, every tested condition inside $|q_1| \le 9^\circ$ and $|q_2| \le 9^\circ$ was stabilized successfully. Among successful recoveries, settling time ranged from about $1.0\,s$ to $8.18\,s$, with an average of about $2.65\,s$, showing that the LQR controller was a strong local stabilizer when residual velocity was small.

<p align="center">
  <img src="../outputs_example/validate/lqr_velocity_basin_settling_q1_p5_q2_p5.png" alt="Velocity basin for the final LQR controller at a representative angle anchor of q1 equals 5 degrees, q2 equals 5 degrees." width="70%">
</p>

*Figure 4. Velocity basin at the representative anchor $(q_1, q_2) = (5^\circ, 5^\circ)$.*

I then extended the validation by adding initial angular velocity at representative near-upright angle anchors. The basin around $(q_1, q_2) = (5^\circ, 5^\circ)$ was especially useful because it is similar to states reached during swing-up handoff.

At this anchor, the controller recovered 152 of 289 tested velocity combinations. Recovery was strongest near the origin in angular-velocity space and degraded as residual angular velocity increased, showing that residual velocity was a more important limitation than small angle offsets alone. In practice, this means the LQR controller worked best as a catch-and-settle controller for slowly moving near-upright states.

## Swing-up + LQR

[Swing-up + LQR Rollout Video](assets/swingup_lqr.mp4)

The end-to-end controller combined the swing-up controller with LQR stabilization near upright. In nominal runs, this pipeline succeeded: the swing-up controller built momentum from rest, brought the pendulum near the upright region, and transferred control to the LQR stabilizer, which then settled the motion.

The main challenge was not simply reaching upright, but reaching it with sufficiently low angular velocity for a smooth handoff. In many runs, the swing-up policy brought the pendulum close to upright in angle, but the state still carried too much momentum. As a result, the handoff to LQR was often aggressive, requiring large initial control effort, and small differences in timing or incoming momentum could qualitatively change the outcome.

<div style="display: flex; justify-content: center; gap: 12px; align-items: flex-start;">
  <img src="../outputs_example/swingup_lqr/swingup_lqr_debug.png" alt="Debug plot for the nominal swing-up plus LQR rollout." width="48%">
  <img src="../outputs_example/swingup_lqr/swingup_lqr_post_handoff.png" alt="Post-handoff view of the nominal swing-up plus LQR rollout." width="48%">
</div>

*Figure 5. End-to-end and post-handoff views of the nominal swing-up plus LQR rollout.*

These results suggest that the handoff between the swing-up controller and the local stabilization is critical. Near upright, the LQR controller had a broad local angle basin but a more limited velocity basin, so the swing-up stage needs to consistently produce a sufficiently low velocity state to handoff to the stabilization controller.

### Limitations
- The controller assumed perfect state information. In a real system, some states might need to be estimated rather than measured directly, which could introduce additional error.
- The controller also assumed ideal sensors with no noise or delay. In practice, sensing errors would likely reduce robustness, especially for phase-sensitive swing-up logic and near-upright stabilization.
    - Preliminary experiments showed some robustness to light Gaussian measurement noise, but performance degraded as the noise standard deviation increased.
- The heuristic swing-up controller was highly sensitive to parameter changes and initial conditions, which limited its basin of attraction and reduced repeatability.
- The final control pipeline relied on a narrow handoff region between swing-up and LQR, so near-upright angle alone was not sufficient when the incoming angular velocity remained too large.

## Conclusion

This project demonstrated a working two-stage controller for a cart-mounted double pendulum in MuJoCo. The LQR controller provided strong local stabilization near upright, while the heuristic swing-up controller was able to produce successful swing-up trajectories. Overall, the results showed that local stabilization was reliable near upright, but end-to-end robustness was limited by the difficulty of delivering a clean handoff into the LQR capture region.

### Next Steps
- Improve swing-up robustness by using a more feedback-driven controller than can adapt applied force based on the current state
- Add a dedicated catch or damping phase before LQR handoff to reduce angular velocity.
- Test robustness under additional perturbations such as nonzero cart offsets, model mismatch, and measurement noise.

## References
- https://underactuated.mit.edu/lqr.html
- https://www.youtube.com/watch?v=e7669NPbENY
- https://coecsl.ece.illinois.edu/se420/ast_fur96.pdf

## AI Usage
I used Codex to assist with code implementation and to draft parts of the report. I reviewed and approved all code and written material included in the final submission.
