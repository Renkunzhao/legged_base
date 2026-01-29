# Legged Base
Utilities for legged robots, including timers, rotation transforms, and interpolation.

## LeggedState
LeggedState represents the state of a legged robot.

- Coordinate/frame transformations. Example: set the base orientation using ZYX Euler angles and retrieve the base rotation matrix:
```cpp
LeggedState legged_state;
legged_state.setBaseRotationFromEulerZYX(euler_zyx);
const auto& R_base = legged_state.base_R();
```

- Assemble generalized position and velocity. For Pinocchio with a floating base:
```cpp
pinocchio::urdf::buildModel(urdfPath, pinocchio::JointModelFreeFlyer(), model_);
```
Conventions:
```text
q_pin = [base_position, base_quaternion (x y z w), joint_positions]

v_pin = [base_linear_velocity (base frame), base_angular_velocity (base frame), joint_velocities]
```

Create custom state layouts:
```cpp
legged_state.createCustomState("q_pin", {"base_pos", "base_quat", "joint_pos"}, jointNames_);
legged_state.createCustomState("v_pin", {"base_lin_vel_B", "base_ang_vel_B", "joint_vel"}, jointNames_);

// Update from sensors, e.g.:
legged_state.setBaseRotationFromQuaternion({x, y, z, w});

auto q_pin = legged_state.custom_state("q_pin");
auto v_pin = legged_state.custom_state("v_pin");
```

For more usage demonstration, see: https://github.com/Renkunzhao/unitree_lowlevel

TODO: Collect conventions across tools (e.g., MuJoCo, Isaac Sim/Gym).

## LeggedModel
Provides kinematics and dynamics functions based on Pinocchio. Requires a URDF and a config.yaml.