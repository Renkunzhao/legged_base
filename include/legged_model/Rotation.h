#pragma once
// Kunzhao Ren, LeggedAI Lab, Inc. 2026

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace LeggedAI {

/**
 * Convention (IMPORTANT):
 * - eulerZYX = [yaw, pitch, roll]  (in radians)
 * - Rotation order: R = Rz(yaw) * Ry(pitch) * Rx(roll)
 * - Quaternion is ordered in Eigen/Pinocchio coeffs() format: [x, y, z, w]
 */
    
/**
 * Compute quaternion from ZYX Euler angles (yaw, pitch, roll).
 *
 * @param eulerZYX ZYX Euler angles: [yaw, pitch, roll]
 * @return Quaternion (x, y, z, w) in Eigen convention
 */
Eigen::Quaterniond eulerZYX2Quat(const Eigen::Vector3d& eulerZYX);

/**
 * Same as eulerZYX2Quat, but returns the quaternion as a vector [x, y, z, w].
 *
 * @param eulerZYX ZYX Euler angles: [yaw, pitch, roll]
 * @return Quaternion vector in [x, y, z, w] order (Eigen/Pinocchio/ROS-compatible)
 */
Eigen::Vector4d eulerZYX2QuatVec(const Eigen::Vector3d& eulerZYX);

/**
 * Convert ZYX Euler angle derivatives to angular velocity expressed in the WORLD frame.
 *
 * Note:
 * - This mapping is singular at pitch = ±pi/2 (gimbal lock).
 *
 * @param eulerZYX ZYX Euler angles: [yaw, pitch, roll]
 * @param eulerZYX_dot Time-derivative of ZYX Euler angles
 * @return Angular velocity ω expressed in WORLD frame
 */
Eigen::Vector3d eulerZYXDot2AngVelW(const Eigen::Vector3d& eulerZYX,
                                   const Eigen::Vector3d& eulerZYX_dot);

/**
 * Inverse mapping of eulerZYXDot2AngVelW:
 * Convert WORLD-frame angular velocity to ZYX Euler angle derivatives.
 *
 * Note:
 * - This mapping is singular at pitch = ±pi/2 (gimbal lock).
 *
 * @param eulerZYX ZYX Euler angles: [yaw, pitch, roll]
 * @param base_ang_vel_W Angular velocity ω expressed in WORLD frame
 * @return Euler angle derivatives eulerZYX_dot
 */
Eigen::Vector3d angVelW2EulerZYXDot(const Eigen::Vector3d& eulerZYX,
                                   const Eigen::Vector3d& base_ang_vel_W);

} // namespace LeggedAI
