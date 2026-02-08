#pragma once
// Kunzhao Ren, LeggedAI Lab, Inc. 2026

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

#include "legged_base/math/types.hpp"

namespace legged_base {

/**
 * Convention:
 * - Euler ZYX: [yaw, pitch, roll] (rad)
 * - R = Rz(yaw) * Ry(pitch) * Rx(roll)
 * - Quaternion coeffs() = [x, y, z, w]  (Eigen/Pinocchio)
 * - ω is WORLD-frame unless stated otherwise
 */

// ----------------------------- helpers -----------------------------

template <typename Derived>
inline void _checkSize(const Eigen::MatrixBase<Derived>& v, int n, const char* name) {
  if (v.size() != n) throw std::invalid_argument(std::string(name) + ": wrong size");
}

// ------------------------------ API --------------------------------

template <typename Scalar>
inline Quat<Scalar> eulerZYX2Quat(const Vec3<Scalar>& eulerZYX) {
  const Eigen::AngleAxis<Scalar> yaw  (eulerZYX(0), Vec3<Scalar>::UnitZ());
  const Eigen::AngleAxis<Scalar> pitch(eulerZYX(1), Vec3<Scalar>::UnitY());
  const Eigen::AngleAxis<Scalar> roll (eulerZYX(2), Vec3<Scalar>::UnitX());
  return yaw * pitch * roll;
}

template <typename Scalar>
inline Vec4<Scalar> eulerZYX2QuatVec(const Vec3<Scalar>& eulerZYX) {
  return eulerZYX2Quat<Scalar>(eulerZYX).coeffs(); // [x y z w]
}

// Accept any Eigen 4-vector expression: Vec4, tail(4), segment(.,4), etc.
template <typename Derived>
inline Vec3<typename Derived::Scalar>
quat2eulerZYX(const Eigen::MatrixBase<Derived>& quat_xyzw) {
  using Scalar = typename Derived::Scalar;
  _checkSize(quat_xyzw, 4, "quat2eulerZYX");

  const Vec4<Scalar> qv = quat_xyzw.eval();              // ensure contiguous
  Quat<Scalar> q(qv[3], qv[0], qv[1], qv[2]);            // w,x,y,z
  q.normalize();

  Vec3<Scalar> euler = q.toRotationMatrix().eulerAngles(2, 1, 0); // [yaw pitch roll]

  // Pinocchio-style pitch unwrap (keep same behavior)
  const Scalar pi = Scalar(M_PI);
  if (euler[1] < -pi / Scalar(2)) euler[1] += Scalar(2) * pi;

  if (euler[1] > pi / Scalar(2)) {
    euler[1] = pi - euler[1];
    euler[2] += (euler[2] < Scalar(0) ? pi : -pi);
    euler[0] -= pi;
  }
  return euler;
}

template <typename Scalar>
inline Vec3<Scalar> eulerZYXDot2AngVelW(const Vec3<Scalar>& eulerZYX,
                                       const Vec3<Scalar>& eulerZYX_dot) {
  const Scalar yaw   = eulerZYX(0);
  const Scalar pitch = eulerZYX(1);

  const Scalar sz = std::sin(yaw);
  const Scalar cz = std::cos(yaw);
  const Scalar sy = std::sin(pitch);
  const Scalar cy = std::cos(pitch);

  const Scalar dyaw   = eulerZYX_dot(0);
  const Scalar dpitch = eulerZYX_dot(1);
  const Scalar droll  = eulerZYX_dot(2);

  return Vec3<Scalar>(
      -sz * dpitch + cy * cz * droll,
       cz * dpitch + cy * sz * droll,
       dyaw - sy * droll);
}

template <typename Scalar>
inline Vec3<Scalar> angVelW2EulerZYXDot(const Vec3<Scalar>& eulerZYX,
                                       const Vec3<Scalar>& ang_vel_W) {
  const Scalar yaw   = eulerZYX(0);
  const Scalar pitch = eulerZYX(1);

  const Scalar sz = std::sin(yaw);
  const Scalar cz = std::cos(yaw);
  const Scalar sy = std::sin(pitch);
  const Scalar cy = std::cos(pitch);

  const Scalar wx = ang_vel_W(0);
  const Scalar wy = ang_vel_W(1);
  const Scalar wz = ang_vel_W(2);

  // singular at pitch = ±pi/2 (cy -> 0)
  const Scalar tmp = (cz * wx + sz * wy) / cy;

  return Vec3<Scalar>(
      sy * tmp + wz,
      -sz * wx + cz * wy,
      tmp);
}

template <typename Scalar>
inline Quat<Scalar> extractYawQuaternion(const Quat<Scalar>& q) {
  const Quat<Scalar> qn = q.normalized();

  const Scalar yaw = std::atan2(
      Scalar(2) * (qn.w() * qn.z() + qn.x() * qn.y()),
      Scalar(1) - Scalar(2) * (qn.y() * qn.y() + qn.z() * qn.z()));

  return Quat<Scalar>(Eigen::AngleAxis<Scalar>(yaw, Vec3<Scalar>::UnitZ()));
}

template <typename Derived>
inline typename Derived::Scalar
yawFromR(const Eigen::MatrixBase<Derived>& R) {
  static_assert(Derived::RowsAtCompileTime == 3 &&
                Derived::ColsAtCompileTime == 3,
                "yawFromR: R must be 3x3");

  using Scalar = typename Derived::Scalar;

  // ZYX convention: yaw = atan2(R(1,0), R(0,0))
  return std::atan2(R(1, 0), R(0, 0));
}

template <typename Scalar>
inline Eigen::Matrix<Scalar, 3, 3> Rz(Scalar yaw) {
  const Scalar c = std::cos(yaw);
  const Scalar s = std::sin(yaw);

  Eigen::Matrix<Scalar, 3, 3> R;
  R << c, -s, Scalar(0),
       s,  c, Scalar(0),
       Scalar(0), Scalar(0), Scalar(1);
  return R;
}

} // namespace legged_base
