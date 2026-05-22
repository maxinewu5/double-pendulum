"""
Defines the MJCF model for the cart double pendulum system, including: 
- cart and two pendulum links with masses and inertias
- actuator for applying force to the cart
- hinge geometry for the pendulum joints
- rail geometry to constrain the cart's motion 
"""

MJCF_MODEL = """
<mujoco model="cart_double_pendulum">
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="7 5 0.1" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>
    <geom name="rail" type="box" pos="0 0 1.2" size="2.3 0.08 0.04" rgba="0.25 0.25 0.25 1" contype="0" conaffinity="0"/>
    <geom name="rail_left_stop" type="box" pos="-2.25 0 1.28" size="0.03 0.12 0.12" rgba="0.45 0.15 0.15 1" contype="0" conaffinity="0"/>
    <geom name="rail_right_stop" type="box" pos="2.25 0 1.28" size="0.03 0.12 0.12" rgba="0.15 0.45 0.15 1" contype="0" conaffinity="0"/>

    <body name="cart" pos="0 0 1.34">
      <joint name="slider" type="slide" axis="1 0 0" range="-2.0 2.0"/>
      <geom type="box" size="0.22 0.14 0.08" mass="1.0" rgba="0.2 0.4 0.8 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="-0.12 0.10 -0.08" size="0.045 0.025" quat="0.707107 0 0.707107 0" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="-0.12 -0.10 -0.08" size="0.045 0.025" quat="0.707107 0 0.707107 0" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.12 0.10 -0.08" size="0.045 0.025" quat="0.707107 0 0.707107 0" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.12 -0.10 -0.08" size="0.045 0.025" quat="0.707107 0 0.707107 0" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>

      <body name="link1" pos="0 0 0.08">
        <joint name="hinge1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.04" mass="0.35" rgba="0.9 0.4 0.2 1"/>

        <body name="link2" pos="0 0 0.6">
          <joint name="hinge2" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.035" mass="0.25" rgba="0.2 0.7 0.3 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="cart_force" joint="slider" gear="1" ctrllimited="true" ctrlrange="-50 50"/>
  </actuator>
</mujoco>
"""
