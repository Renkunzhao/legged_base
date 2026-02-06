#pragma once
#include <Eigen/Dense>
#include <vector>

namespace legged_base {

template <typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
stdVecToEigen(const std::vector<Scalar> &v) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> out(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    out[i] = v[i];
  return out;
}

template <typename Derived>
inline std::vector<typename Derived::Scalar>
eigenToStdVec(const Eigen::MatrixBase<Derived> &v) {
  using Scalar = typename Derived::Scalar;
  const auto tmp = v.eval(); // 保证连续
  return std::vector<Scalar>(tmp.data(), tmp.data() + tmp.size());
}

} // namespace legged_base
