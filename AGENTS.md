# Double Pendulum Notes

## Coordinate Convention

- `q1` is the absolute angle of link 1 measured relative to the vertical upright direction.
- `q1 = 0` means link 1 is exactly upright.
- `q1 = pi` or `q1 = -pi` means link 1 points straight down.
- Increasing `q1` corresponds to clockwise rotation.
- Decreasing `q1` corresponds to counterclockwise rotation.
- `q1_dot` is the angular velocity associated with `q1`.
- `q1_dot > 0` means link 1 is rotating clockwise.
- `q1_dot < 0` means link 1 is rotating counterclockwise.

This sign convention matters when interpreting phase logic such as
`np.sign(q1_dot)` in swing-up policies.
