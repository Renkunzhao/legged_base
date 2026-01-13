#include "legged_model/Lie.h"

#include "iostream"
#include <Eigen/src/Geometry/Quaternion.h>
#include <cstdio>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <manif/impl/se3/SE3.h>
#include <manif/impl/so3/SO3.h>
#include <manif/manif.h>

using namespace Lie;

int main(){
    std::getchar();
    std::srand((unsigned int) std::time(nullptr));

    Eigen::Quaterniond q1 = Eigen::Quaterniond::UnitRandom();
    Eigen::Quaterniond q2 = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector4d q1_, q2_;
    q1_ << q1.w(), q1.x(), q1.y(), q1.z();
    q2_ << q2.w(), q2.x(), q2.y(), q2.z();

    std::cout << "[Lie_test] q1: " << q1.coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] q1_: " << q1_.transpose() << std::endl;
    std::cout << "[Lie_test] q2: " << q2.coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] q2_: " << q2_.transpose() << std::endl;

    std::cout << "[Lie_test] q1*q2 (Eigen):    " << (q1*q2).coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] q1*q2 (quat_productMatL): " << quat_product(q1_, q2_).transpose() << std::endl;
    std::cout << "[Lie_test] q1*q2 (quat_productMatR): " << ( quat_productMatR(q2_) * q1_).transpose() << std::endl;


    std::cout << "[Lie_test] q1  conjugate: " << q1.conjugate().coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] q1_ conjugate: " << quat_conjugate(q1_).transpose() << std::endl;

    Eigen::Vector3d phi1 = quat_Log(q1_);
    std::cout << "[Lie_test] Log(q1_): " << phi1.transpose() << std::endl;
    std::cout << "[Lie_test] Exp(Log(q1_)): (Eigen) " << Eigen::Quaterniond(Eigen::AngleAxisd(phi1.norm(), phi1.normalized())).coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] Exp(Log(q1_)): (quat_Exp) " << quat_Exp(quat_Log(q1_)).transpose() << std::endl;

    Eigen::Matrix3d R1 = q1.toRotationMatrix();
    Eigen::Matrix3d R2 = q2.toRotationMatrix();
    Eigen::Matrix3d R1_ = quat_ToR(q1_);
    Eigen::Matrix3d R2_ = quat_ToR(q2_);
    std::cout << "[Lie_test] R1:\n" << R1 << std::endl;
    std::cout << "[Lie_test] R1_:\n" << R1_ << std::endl;

    Eigen::Vector3d phi1_R = R_Log(R1_);
    std::cout << "[Lie_test] Exp(Log(R1_)): (Eigen)\n" << Eigen::Quaterniond(Eigen::AngleAxisd(phi1_R.norm(), phi1_R.normalized())).toRotationMatrix() << std::endl;
    std::cout << "[Lie_test] Exp(Log(R1_)): (R_Exp)\n" << R_Exp(R_Log(R1_)) << std::endl;
    
    Eigen::Quaterniond q1_R(R1);
    std::cout << "[Lie_test] q1_R: (Eigen)" << q1_R.coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] q1_R: (R_ToQuat)" << R_ToQuat(R1_).transpose() << std::endl;

    std::cout << "[Lie_test] q1_*q2':\n" << quat_ToR(quat_product(q1_, quat_conjugate(q2_))) << std::endl;
    std::cout << "[Lie_test] R1_ * R2_^T:\n" << R1_*R2_.transpose() << std::endl;

    manif::SO3d SO3_1(q1), SO3_2(q2);
    std::cout << "[Lie_test] manif\n";
    std::cout << "[Lie_test] Log(R1*R2^T): " << SO3_1.lminus(SO3_2).coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] Log(R2*R1^T): " << SO3_2.lminus(SO3_1).coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] Log(R2^T*R1): " << SO3_1.rminus(SO3_2).coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] Log(R1^T*R_2): " << SO3_2.rminus(SO3_1).coeffs().transpose() << std::endl;
    
    std::cout << "[Lie_test] self\n";
    std::cout << "[Lie_test] Log(R1*R2^T): " << R_boxminusW(R1_, R2_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(R2*R1^T): " << R_boxminusW(R2_, R1_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(R2^T*R1): " << R_boxminusL(R1_, R2_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(R1^T*R_2): " << R_boxminusL(R2_, R1_).transpose() << std::endl;
    std::cout << "[Lie_test] R2*Log(R2^T*R1): " << (R2_*R_boxminusL(R1_, R2_)).transpose() << std::endl;
    std::cout << "[Lie_test] R1*Log(R1^T*R_2): " << (R1_*R_boxminusL(R2_, R1_)).transpose() << std::endl;

    std::cout << "[Lie_test] Log(q1*q2'): " << quat_boxminusW(q1_,q2_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(q2*q1'): " << quat_boxminusW(q2_,q1_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(q2'*q1): " << quat_boxminusL(q1_,q2_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(q1'*q2): " << quat_boxminusL(q2_,q1_).transpose() << std::endl;
    std::cout << "[Lie_test] q2*Log(q2'*q1)*q2': " << ( quat_rotateVec(q2_, quat_boxminusL(q1_,q2_).transpose()) ).transpose() << std::endl;
    std::cout << "[Lie_test] q1*Log(q1'*q2)*q1': " << ( quat_rotateVec(q1_, quat_boxminusL(q2_,q1_).transpose()) ).transpose() << std::endl;

    manif::SE3d T1 = manif::SE3d::Random();
    manif::SE3d T2 = manif::SE3d::Random();
    T1.quat(q1);
    T2.quat(q2);
    Matrix4d T1_ = T1.transform();
    Matrix4d T2_ = T2.transform();
    std::cout << "[Lie_test] T1:\n" << T1 << std::endl;
    std::cout << "[Lie_test] T2:\n" << T2 << std::endl;
    std::cout << "[Lie_test] T1_:\n" << T1_ << std::endl;
    std::cout << "[Lie_test] T2_:\n" << T2_ << std::endl;

    std::cout << "[Lie_test] Log(T1):" << T1.log() << std::endl;
    std::cout << "[Lie_test] Log(T1_):" << T_Log(T1_).transpose() << std::endl;
    std::cout << "[Lie_test] Exp(Log(T1)):\n" << T1.log().exp().transform() << std::endl;
    std::cout << "[Lie_test] Exp(Log(T1_)):\n" << T_Exp(T_Log(T1_)) << std::endl;
    
    std::cout << "[Lie_test] Log(T2'*T1):" << T1.rminus(T2) << std::endl;
    std::cout << "[Lie_test] Log(T1'*T2):" << T2.rminus(T1) << std::endl;
    std::cout << "[Lie_test] Log(T1*T2'):" << T1.lminus(T2) << std::endl;
    std::cout << "[Lie_test] Log(T2*T1'):" << T2.lminus(T1) << std::endl;

    std::cout << "[Lie_test] Log(T2'*T1):" << T_boxminusL(T1_, T2_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(T1'*T2):" << T_boxminusL(T2_, T1_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(T1*T2'):" << T_boxminusW(T1_, T2_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(T2*T1'):" << T_boxminusW(T2_, T1_).transpose() << std::endl;

    // manif::SE3d X = manif::SE3d::Random();
    // manif::SE3Tangentd w = manif::SE3Tangentd::Random();

    // manif::SE3d::Jacobian J_o_x, J_o_w;

    // auto X_plus_w = X.plus(w, J_o_x, J_o_w);
    // std::cout << "[Lie_test] X:\n" << X << std::endl;
    // std::cout << "[Lie_test] w:\n" << w << std::endl;
    // std::cout << "[Lie_test] X+w:\n" << X_plus_w << std::endl;
    // std::cout << "[Lie_test] J_o_x:\n" << J_o_x << std::endl;
    // std::cout << "[Lie_test] J_o_w:\n" << J_o_w << std::endl;

    // pinocchio::SE3 M1(X.rotation(), X.translation());
    // std::cout << "[Lie_test] M1:\n" << M1 << std::endl;
    // std::cout << "[Lie_test] Jlog6(M1):\n" << pinocchio::Jlog6(M1).transpose() << std::endl;

    // 随机生成两个 SE3
    manif::SE3d X = manif::SE3d::Random();
    manif::SE3d Y = manif::SE3d::Random();

    // 右减：log(Y^{-1} * X)
    manif::SE3d::Jacobian J_x, J_y;
    manif::SE3Tangentd rminus = X.rminus(Y, J_x, J_y);

    std::cout << "manif rminus(X,Y) = " << rminus << std::endl;
    std::cout << "Jacobian wrt X (manif):\n" << J_x << std::endl;
    std::cout << "Jacobian wrt Y (manif):\n" << J_y << std::endl;

    // 同样的运算在 Pinocchio 中
    pinocchio::SE3 Xp(X.rotation(), X.translation());
    pinocchio::SE3 Yp(Y.rotation(), Y.translation());

    pinocchio::SE3 rel = Yp.inverse() * Xp; // Y^{-1} * X
    Eigen::Matrix<double,6,6> Jlog = pinocchio::Jlog6(rel);

    std::cout << "Pinocchio Jlog6(Y^{-1}*X):\n" << Jlog << std::endl;

    // 验证：manif 的 J_x 应该等于 Pinocchio 的 Jlog6
    std::cout << "Difference (manif J_x - pinocchio Jlog6): "
              << (J_x - Jlog).norm() << std::endl;

    // ===== 验证 dIntegrate vs manif::plus =====

    // ===== 验证 dIntegrate vs manif::plusJacobian =====
    std::cout << "\n==== Verify dIntegrate (Pinocchio) vs plusJacobian (manif) ====\n";

    // 随机一个状态 Xp_m ∈ SE3 和扰动 wm_m ∈ se(3)
    manif::SE3d Xp_m = manif::SE3d::Random();
    manif::SE3Tangentd wm_m = manif::SE3Tangentd::Random();

    // manif 计算 plus 和 Jacobians
    manif::SE3d::Jacobian Jx_manif, Jw_manif;
    manif::SE3d Xp_plus_w = Xp_m.plus(wm_m, Jx_manif, Jw_manif);

    std::cout << "[manif] Jx:\n" << Jx_manif << std::endl;
    std::cout << "[manif] Jw:\n" << Jw_manif << std::endl;

    // pinocchio 模型: 一个自由基 (FreeFlyer)
    pinocchio::Model model;
    model.addJoint(0, pinocchio::JointModelFreeFlyer(), pinocchio::SE3::Identity(), "base");
    pinocchio::Data data(model);

    // 转换到 pinocchio 格式
    Eigen::VectorXd q_ff(7), v_ff(6);
    // q = [x y z qw qx qy qz]
    Eigen::Quaterniond quat_ff(Xp_m.rotation());
    q_ff << Xp_m.translation(), quat_ff.w(), quat_ff.x(), quat_ff.y(), quat_ff.z();
    v_ff = wm_m.coeffs(); // [vx vy vz wx wy wz]

    // dIntegrate wrt q
    Eigen::MatrixXd Jq_pin(6,6); // (nv x nq)
    pinocchio::dIntegrate(model, q_ff, v_ff, Jq_pin, pinocchio::ARG0);

    // dIntegrate wrt v
    Eigen::MatrixXd Jv_pin(6,6); // (nv x nv)
    pinocchio::dIntegrate(model, q_ff, v_ff, Jv_pin, pinocchio::ARG1);

    std::cout << "[pinocchio] Jq:\n" << Jq_pin << std::endl;
    std::cout << "[pinocchio] Jv:\n" << Jv_pin << std::endl;

    // 对比差异
    // manif 的 Jx, Jw 都是 6x6，需要从 Pinocchio 的结果里取对应部分比较
    double diff_Jx = (Jx_manif - Jq_pin.leftCols(6)).norm(); // 丢掉 quaternion 的最后一列
    double diff_Jw = (Jw_manif - Jv_pin).norm();

    std::cout << "||Jx_manif - Jq_pin|| = " << diff_Jx << std::endl;
    std::cout << "||Jw_manif - Jv_pin|| = " << diff_Jw << std::endl;

}
