#pragma once
// Kunzhao Ren, LeggedAI Lab, Inc. 2026

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace legged_base {

/**
 * ======================= BASIC MATH TYPES =======================
 *
 * These aliases define the canonical vector and quaternion types
 * used throughout legged_base.
 *
 * - Scalar is typically float or double
 * - All vectors are column vectors
 * - Quaternion follows Eigen convention:
 *     coeffs() = [x, y, z, w]
 *
 * ================================================================
 */

template <typename Scalar>
using Vec2 = Eigen::Matrix<Scalar, 2, 1>;

template <typename Scalar>
using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

template <typename Scalar>
using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

template <typename Scalar>
using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using Mat3 = Eigen::Matrix<Scalar, 3, 3>;

template <typename Scalar>
using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar>
using Quat = Eigen::Quaternion<Scalar>;

} // namespace legged_base
