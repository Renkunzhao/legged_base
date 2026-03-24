#include "legged_base/math/rotation_euler_zyx.hpp"

#include <pinocchio/math/rpy.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace Eigen;
using namespace legged_base;

namespace {

double wrapAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

Vector3d wrappedDiff(const Vector3d& a, const Vector3d& b) {
    Vector3d diff = a - b;
    for (int i = 0; i < 3; ++i) {
        diff[i] = wrapAngle(diff[i]);
    }
    return diff;
}

void checkNear(const std::string& name, double err, double tol, int sample_id) {
    if (err > tol) {
        throw std::runtime_error(
            "[Rotation_test] " + name + " mismatch at sample " + std::to_string(sample_id) +
            ", err = " + std::to_string(err) + ", tol = " + std::to_string(tol));
    }
}

}  // namespace

int main() {
    constexpr int kNumSamples = 200;
    constexpr double kTolR = 1e-12;
    constexpr double kTolQuat = 1e-12;
    constexpr double kTolEuler = 1e-12;
    constexpr double kTolOmega = 1e-12;

    std::mt19937 rng(0);
    std::uniform_real_distribution<double> yaw_roll_dist(-M_PI, M_PI);
    std::uniform_real_distribution<double> pitch_dist(-1.2, 1.2);
    std::uniform_real_distribution<double> rate_dist(-10.0, 10.0);

    double max_r_err = 0.0;
    double max_quat_err = 0.0;
    double max_euler_err = 0.0;
    double max_omega_w_err = 0.0;
    double max_omega_b_err = 0.0;
    double max_rate_inv_err = 0.0;

    for (int i = 0; i < kNumSamples; ++i) {
        Vector3d eulerZYX;
        eulerZYX << yaw_roll_dist(rng), pitch_dist(rng), yaw_roll_dist(rng);

        const Vector3d rpy = eulerZYX.reverse();
        const Matrix3d R_self = eulerZYX2Quat(eulerZYX).toRotationMatrix();
        const Matrix3d R_pino = pinocchio::rpy::rpyToMatrix(rpy);

        const double r_err = (R_self - R_pino).norm();
        max_r_err = std::max(max_r_err, r_err);
        checkNear("R", r_err, kTolR, i);

        Vector4d quat_self = eulerZYX2QuatVec(eulerZYX);
        Vector4d quat_pino = Quaterniond(R_pino).coeffs();
        if (quat_self.dot(quat_pino) < 0.0) {
            quat_pino = -quat_pino;
        }
        const double quat_err = (quat_self - quat_pino).norm();
        max_quat_err = std::max(max_quat_err, quat_err);
        checkNear("quat", quat_err, kTolQuat, i);

        const Vector3d euler_self = quat2eulerZYX(quat_self);
        const Vector3d euler_pino = pinocchio::rpy::matrixToRpy(R_pino).reverse();
        const double euler_err = wrappedDiff(euler_self, euler_pino).norm();
        max_euler_err = std::max(max_euler_err, euler_err);
        checkNear("euler", euler_err, kTolEuler, i);

        Vector3d eulerZYX_dot;
        eulerZYX_dot << rate_dist(rng), rate_dist(rng), rate_dist(rng);
        const Vector3d rpy_dot = eulerZYX_dot.reverse();

        const Matrix3d J_world =
            pinocchio::rpy::computeRpyJacobian(rpy, pinocchio::LOCAL_WORLD_ALIGNED);
        const Matrix3d J_body =
            pinocchio::rpy::computeRpyJacobian(rpy, pinocchio::LOCAL);

        const Vector3d omega_world_self = eulerZYXDot2AngVelW(eulerZYX, eulerZYX_dot);
        const Vector3d omega_world_pino = J_world * rpy_dot;
        const double omega_w_err = (omega_world_self - omega_world_pino).norm();
        max_omega_w_err = std::max(max_omega_w_err, omega_w_err);
        checkNear("omega_world", omega_w_err, kTolOmega, i);

        const Vector3d omega_body_self = R_self.transpose() * omega_world_self;
        const Vector3d omega_body_pino = J_body * rpy_dot;
        const double omega_b_err = (omega_body_self - omega_body_pino).norm();
        max_omega_b_err = std::max(max_omega_b_err, omega_b_err);
        checkNear("omega_body", omega_b_err, kTolOmega, i);

        const Vector3d eulerZYX_dot_recovered = angVelW2EulerZYXDot(eulerZYX, omega_world_pino);
        const double rate_inv_err = (eulerZYX_dot_recovered - eulerZYX_dot).norm();
        max_rate_inv_err = std::max(max_rate_inv_err, rate_inv_err);
        checkNear("euler_dot_inverse", rate_inv_err, kTolOmega, i);
    }

    std::cout << "[Rotation_test] pinocchio::rpy and legged_base rotation helpers are consistent.\n";
    std::cout << "  max R err                = " << max_r_err << "\n";
    std::cout << "  max quat err             = " << max_quat_err << "\n";
    std::cout << "  max euler err            = " << max_euler_err << "\n";
    std::cout << "  max omega world err      = " << max_omega_w_err << "\n";
    std::cout << "  max omega body err       = " << max_omega_b_err << "\n";
    std::cout << "  max euler_dot inverse err= " << max_rate_inv_err << std::endl;

    return 0;
}
