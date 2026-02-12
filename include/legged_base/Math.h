#pragma once

#include <Eigen/Dense>
#include <algorithm> // for std::max

namespace legged_base {

template <typename T>
inline int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d s = Eigen::Matrix3d::Zero();
  s(0, 1) = -v.z();
  s(0, 2) = v.y();
  s(1, 0) = v.z();
  s(1, 2) = -v.x();
  s(2, 0) = -v.y();
  s(2, 1) = v.x();
  return s;
}

/// @brief Moore–Penrose pseudoinverse using SVD
/// @param J  Input matrix
/// @param tol  Relative tolerance (default 1e-9)
/// @return J^+  (Moore–Penrose pseudoinverse)
inline Eigen::MatrixXd pseudoInverseSVD(const Eigen::MatrixXd &J, double tol = 1e-9) {
    // Decompose J = U Σ V^T
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::VectorXd sigma = svd.singularValues();

    // Threshold for treating singular values as zero
    double sigma_max = sigma(0); // largest singular value
    double tolerance = tol * std::max(J.cols(), J.rows()) * sigma_max;

    // Build Σ^+
    Eigen::VectorXd sigmaInv = sigma;
    for (long i = 0; i < sigma.size(); i++) {
        sigmaInv(i) = (sigma(i) > tolerance) ? 1.0 / sigma(i) : 0.0;
    }

    // Compute pseudoinverse: J^+ = V Σ^+ U^T
    return svd.matrixV() * sigmaInv.asDiagonal() * svd.matrixU().transpose();
}

/// @brief Damped least squares pseudoinverse
/// @param J  Input matrix
/// @param lambda  Damping factor (default 1e-6)
/// @return J^+_λ  (approximate pseudoinverse)
inline Eigen::MatrixXd pseudoInverseDLS(const Eigen::MatrixXd &J, double lambda = 1e-6) {
    Eigen::MatrixXd JJt = J * J.transpose();
    Eigen::MatrixXd JJt_damped = JJt + lambda * Eigen::MatrixXd::Identity(JJt.rows(), JJt.cols());
    return J.transpose() * JJt_damped.ldlt().solve(Eigen::MatrixXd::Identity(JJt.rows(), JJt.cols()));
}

} // namespace legged_base
