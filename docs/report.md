# Double Pendulum Swing-Up and Balance
Maxine Wu

## Goal
The goal of this project was to control a cart-mounted double pendulum in simulation and drive it to the upright position using a horizontal force applied to the cart. 

## Model and Environment

The plant was modeled in MuJoCo as a planar cart-double-pendulum system consisting of a sliding cart, two pendulum links, and a single motor actuator applying horizontal force to the cart. The cart moves along a bounded rail, and each pendulum link is connected by a hinge joint. 

The system state was defined as follows:
$$
x = [qpos, qvel] =
\begin{bmatrix}
x_{cart} & \theta_1 & \theta_2 & \dot{x}_{cart} & \dot{\theta}_1 & \dot{\theta}_2
\end{bmatrix}^T.
$$

![State diagram for the cart-double-pendulum system, showing the cart position, input force, and the two link angles.](assets/system_state_diagram.png)

*Figure 1. Planar cart-double-pendulum model and state definition. The second joint angle \(\theta_2\) is defined relative to link 1, following the MuJoCo joint convention.*

> Note: We define the second joint angle as the angle relative to the first joint angle rather than the absolute world-frame angle, following MuJoCo's convention defined relative to the first link rather than the absolute world frame. In the rest of the report, we will distinguish between the the relative second-link angle and the absolute orientation of the second link when interpreting controller state.

The control input was a single horizontal cart force:

$$
u = \begin{bmatrix} F_{cart} \end{bmatrix}.
$$

This system is underactuated, since neither pendulum joint is directly actuated.

The main modeling assumptions are below:
- The motion was restricted to a planar 2D setting, so out-of-plane dynamics were ignored.
- The cart and pendulum segments were modeled as rigid bodies with fixed masses and lengths.
- The controller had access to the full simulated state, meaning perfect state measurement was assumed with no sensor noise, delay, or state-estimation error.
- The baseline simulation was deterministic, so disturbances and unmodeled effects were not included.

These assumptions allow us to model the critical dynamics of the double-pendulum system while removing some additional physics considerations such as flexible bodies to keep the project focused on the core control design.

The simulation parameters are as follows:
- cart track range: approximately `[-2.0, 2.0] m`
- actuator limit: `[-50, 50] N`
- simulation timestep: `0.005 s`
- cart mass: `1.0 kg`
- link 1 length / mass: `0.6 m`, `0.35 kg`
- link 2 length / mass: `0.5 m`, `0.25 kg`

## Controller Design

From a control perspective, the problem separates into a couple of distinct parts. The first is local stabilization near the upright equilibrium. The second is a swing-up from rest, where the controller must build pendulum motion to guide the system into the stabilizable region near the upright equilibium. Additionally, a handoff between the different control system might be needed so the swing-up controller can transfer control to the stabilization controller once the system is close to upright, without destabilizing the system.

### Stabilization Near Upright

For designing a control algorithm to stabilize the pendulum near an equilibrium. I primarily considered an LQR (linear quadratic regulator) approach. 

Near the upright equilibrium, the pendulum angles and velocities are small enough that the nonlinear dynamics of the system can be well approximated by a linear model. This makes a linear state-feedback design appropriate for stablization near upright, even though the full pendulum system may be nonlinear (https://underactuated.mit.edu/lqr.html). The controller also needs to be able to optimize across multiple coupled state errors (ie. the angle of the first and second pendulum links) rather than one single error term, which makes LQR suitable.

I briefly considered some alternate control approaches: 
- PID: PID does not work well when there are multiple important input states like we have here with the double pendulum (cart position, first angle, second angle), as PID mostly focuses on reducing one error term. 
- Pole Placement: Another linear control approach similar to LQR, but harder to tune, due to choosing eigenvalues being less interpretable and less intuitive
- More complex control methods could be considred but LQR met the needs of our controller design and was more straightforward to implement.

The local linear dynamics used for LQR are written in discrete-time state-space form as

$$
x_{k+1} = A x_k + B u_k,
$$

where \(x_k\) is the six-dimensional system state and \(u_k\) is the cart-force input. 

The LQR controller then computes a state-feedback law

$$
u_k = -K x_k
$$

that minimizes the quadratic cost

$$
J = \sum_{k=0}^{\infty} \left(x_k^T Q x_k + u_k^T R u_k\right),
$$

where \(Q\) penalizes state error and \(R\) penalizes control effort.

In this project, I leveraged MuJoCo's discrete linearization at the upright reference state to compute the matrices local state-transition matrix \(A\) and the input matrix \(B\). MuJoCo uses a small perturbation size (\(10^{-6}\)) on each of the state values to estimate how the system's next-step dynamics change with respect to the current state and input. Intuitively, MuJoCo computes how the next simulation step changes in response to small perturbations in the current state and input, which produces a discrete-time linear approximation around that operating point.

I chose to use a discrete-time linearization because it was consistent with the simulator, which made the implementation easier and also made the controller design more consistent with how real digital controllers are implemented (feedback updated at discrete intervals rather than in a true continuous nature).

The weighting matrices \(Q\) and \(R\) were then tuned to balance pendulum stabilization against actuator effort. I first tuned these weights manually by observing rollout behavior, then used parameter sweeps and validation runs to compare settling time, recovery success, and control effort across different rollouts.

### Swing-Up Policy

The next part of this project involves swinging up the pendulum from rest.
We need a way to swing the pendulum up. I considered several approaches here:

#### 1. Phase-based Swing Up
A phase-based swing-up strategy first considered, where cart forces were applied according to the pendulum’s motion phase rather than a direct energy estimate. In particular, force pulses were triggered near favorable parts of the swing to increase oscillation amplitude and move the system toward the upright region. However, the phase of a two pendulum system is much more complex than the phase of a single pendulum system, and the controller is highly sensitive to timing. It was difficult to tune this controller to get desired results.

#### 2. Energy-based Swing Up

Energy-based swing-up works by estimating the pendulum’s current mechanical energy, comparing it to the target energy required for the fully upright position, and then applying cart forces to either add energy or remove energy as needed. So if the pendulum's total energy was less than the target energy, we could apply a cart force in a direction that will increase the oscillation amplitude, and if the pendulum's total energy was greater than the target energy, we could reduce the force being applied to damp the motion and slow the pendulum down to avoid overshootingn the target.

The approach seemed promising because energy-based swing-up is a known startegy for balancing underactuated pendulum system. However, in practice, it ended up being pretty difficult to make this robust for a double pendulum system. The two links can exchange energy, and their coupled motion introduces more complex phase-dependent behaviors than in a single pendulum system. 

Approximating the energy of the system with the energy of the first link only proved insufficient, because it didn't consider the phase of the second link. When approximiating the energy of the system with the first and second link, the total energy became much harder to interpret and harder to use for control signals. In practice, the controller was highly sensitive to the timing and phase of the control inputs, and this usually created unstable behavior in the system.

Reference: https://coecsl.ece.illinois.edu/se420/ast_fur96.pdf

#### 3. Cart Motion Swing Up

A heuristic controller was also explored. This controller is inspired by the deliberate left-right cart motion. 

This approach uses a state machine to move the cart between left and right waypoints while braking and reversing at controlled times. The main modes were drive_right, brake_right, hold_right, drive_left, brake_left, and hold_left.

This design was motivated by some physical intuition around what successful swing-up behavior looks like: the cart needs to move to the left/right quickly, stop quickly and allow the pendulum to swing to its apex, then reverse direction at useful times, while remainining within the limits of the rail. 

In practice, this approach was significantly easier to tune than the earlier swing-up attempts. Its behavior was also more interpretable in plots, since each controller mode had a clear purpose and the cart motion could be debugged directly.

This controller also tended to have relatively stable control inputs, which helped keep the second link from becoming too chaotic and putting the entire pendulum system into a state that would be hard to recover from.

## Results

TODO 

### Current Best Result
TODO

### Limitations
- The controller assumed perfect state information. In simulation, the full state is available directly, but in a real system some states would need to be estimated rather than measured exactly.
- The controller also assumed ideal sensors with no noise or delay. Real sensing errors would likely reduce robustness, particularly for phase-sensitive swing-up logic and near-upright stabilization.

## Analysis

TODO

## Conclusion

TODO

### Next Steps
TODO 

## AI Usage
Codex was used to iterate on controller design, implement the controller code, and to draft the report. I was involved in reviewing all of the code and the report writeup, and I take full responsibility for all design decisions.
