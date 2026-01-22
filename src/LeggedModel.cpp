#include "legged_model/LeggedModel.h"
#include "legged_model/Math.h"
#include "legged_model/Lie.h"
#include "legged_model/Utils.h"
#include "legged_model/Rotation.h"
#include <cstddef>
#include <iostream>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/math/rpy.hpp>

using namespace Lie;
using namespace LeggedAI;

void LeggedModel::loadConfig(const YAML::Node& node){
    this->loadUrdf(node["urdfPath"].as<string>(), "quaternion",
                    node["baseName"].as<string>(), 
                    node["contact3DofNames"].as<vector<string>>(), 
                    node["contact6DofNames"].as<vector<string>>(),
                    node["hipNames"].as<vector<string>>(),
                    node["verbose"].as<bool>());

    contact3DofPossInit_ = splitVectors<Eigen::Vector3d>(yamlToEigenVector(node["contact3DofPossInit"]), 3) ;
    if (contact3DofPossInit_.size() != nContacts3Dof_) {
        std::cout << "[LeggedModel] contact3DofPossInit and contact3DofNames size doesn't match.\n";  
    }

    jointOrder_ = node["jointOrder"].as<vector<string>>();
    // print custom joint order
    cout << "[LeggedModel] Custom joint order: ";
    for (const auto& jointName : jointOrder_) {
        cout << jointName << " ";
    }
    cout << endl;

    qj_min_ = yamlToEigenVector(node["qj_min"]);
    qj_max_ = yamlToEigenVector(node["qj_max"]);
    tau_max_ = yamlToEigenVector(node["tau_max"]);
    this->setJointLimits(qj_max_, qj_min_);
}

void LeggedModel::loadUrdf(string urdfPath, string baseType, string baseName,
                           vector<string> contact3DofNames, 
                           vector<string> contact6DofNames, 
                           vector<string> hipNames, 
                           bool verbose) {
    cout << "[LeggedModel] Load URDF from " << urdfPath << endl;
    baseType_ = baseType;
    if (baseType_ == "quaternion") {
        // 使用 pinocchio::JointModelFreeFlyer 的浮动基机器人模型（基于四元数）
        pinocchio::urdf::buildModel(urdfPath, pinocchio::JointModelFreeFlyer(), model_);
        nqBase_ = 7;
    } else if (baseType_ == "eulerZYX") {
        // 使用 pinocchio::JointModelComposite(Translation + EulerZYX) 的浮动基机器人模型   
        pinocchio::JointModelComposite jointComposite(2);
        jointComposite.addJoint(pinocchio::JointModelTranslation());      // 3 DoF 平移
        jointComposite.addJoint(pinocchio::JointModelSphericalZYX());     // 3 DoF 旋转
        pinocchio::urdf::buildModel(urdfPath, jointComposite, model_);
        nqBase_ = 6;
    } else {
        throw runtime_error("Invalid orientation type specified: " + baseType_);
    }

    cout << "---- Joints ----" << endl;
    for (size_t i = 0; i < model_.joints.size(); ++i) {
        cout << i << ": " << model_.names[i] << endl;
    }

    cout << "---- Links (Frames of type BODY) ----" << endl;
    for (size_t i = 0; i < model_.frames.size(); ++i) {
        if (model_.frames[i].type == pinocchio::BODY) {
            cout << i << ": " << model_.frames[i].name << endl;
        }
    }

    data_ = pinocchio::Data(model_);    

    nJoints_ = model_.nv - 6;

    // set joint order
    // 跳过 universe 和 base
    for (size_t i = 2; i < model_.names.size(); ++i) {
        jointNames_.push_back(model_.names[i]);
    }

    baseName_ = baseName;

    contact3DofNames_ = contact3DofNames;
    nContacts3Dof_ = contact3DofNames_.size();
    for(const auto& ee3Dof_ : contact3DofNames_) contact3DofIds_.push_back(model_.getBodyId(ee3Dof_));

    contact6DofNames_ = contact6DofNames;
    nContacts6Dof_ = contact6DofNames_.size();
    for(const auto& ee6Dof_ : contact6DofNames_) contact6DofIds_.push_back(model_.getBodyId(ee6Dof_));

    hipNames_ = hipNames;
    for(const auto& hipName : hipNames_) hipIds_.push_back(model_.getJointId(hipName));

    // Translation bounds
    model_.lowerPositionLimit.head<3>().setConstant(-10.0);  // x, y, z
    model_.upperPositionLimit.head<3>().setConstant(10.0);

    // Orientation（四元数不设置限制，EulerZYX可以设置为 -pi 到 pi）
    if (baseType_ == "eulerZYX") {
        model_.lowerPositionLimit.segment<3>(3).setConstant(-M_PI);
        model_.upperPositionLimit.segment<3>(3).setConstant(M_PI);
    }

    verbose_ = verbose;
    cout << "[LeggedModel] nDof: " << nDof() << endl; 
}

vector<Eigen::Vector3d> LeggedModel::contact3DofPoss(const Eigen::VectorXd& q_pin){
    pinocchio::forwardKinematics(model_, data_, q_pin);
    pinocchio::updateFramePlacements(model_, data_);

    vector<Eigen::Vector3d> contact3DofPoss;
    for (const auto& Id : contact3DofIds_) contact3DofPoss.push_back(data_.oMf[Id].translation());
    return contact3DofPoss;
}

Eigen::VectorXd LeggedModel::contact3DofPossOrder(const Eigen::VectorXd& jointPos, const Eigen::VectorXd& qBase){
    Eigen::VectorXd q_pin(this->nqPin()), qJoint(nJoints_);
    LeggedAI::reorder(jointOrder_, jointPos, jointNames_, qJoint);
    if (qBase.size() == 0) {
        q_pin << qBase0(), qJoint;
    } else {
        q_pin << qBase, qJoint;
    }
    return concatVectors(contact3DofPoss(q_pin));
}

vector<Eigen::Vector3d> LeggedModel::contact3DofVels(const Eigen::VectorXd& q_pin, const Eigen::VectorXd& v_pin){
    pinocchio::forwardKinematics(model_, data_, q_pin, v_pin);
    pinocchio::updateFramePlacements(model_, data_);

    vector<Eigen::Vector3d> contact3DofVels;
    for (const auto& Id : contact3DofIds_) contact3DofVels.push_back(pinocchio::getFrameVelocity(model_, data_, Id, pinocchio::LOCAL_WORLD_ALIGNED).linear());
    return contact3DofVels;
}

vector<Eigen::Vector3d> LeggedModel::contact6DofPoss(const Eigen::VectorXd& q_pin){
    pinocchio::forwardKinematics(model_, data_, q_pin);
    pinocchio::updateFramePlacements(model_, data_);

    vector<Eigen::Vector3d> contact6DofPoss;
    for (const auto& Id : contact6DofIds_) contact6DofPoss.push_back(data_.oMf[Id].translation());
    return contact6DofPoss;
}

vector<Eigen::Vector3d> LeggedModel::contact6DofVels(const Eigen::VectorXd& q_pin, const Eigen::VectorXd& v_pin){
    pinocchio::forwardKinematics(model_, data_, q_pin, v_pin);
    pinocchio::updateFramePlacements(model_, data_);

    vector<Eigen::Vector3d> contact6DofVels;
    for (const auto& Id : contact6DofIds_) contact6DofVels.push_back(pinocchio::getFrameVelocity(model_, data_, Id, pinocchio::LOCAL_WORLD_ALIGNED).linear());
    return contact6DofVels;
}

vector<Eigen::Vector3d> LeggedModel::hipPoss(const Eigen::VectorXd& qBase){
    Eigen::VectorXd q_pin = Eigen::VectorXd::Zero(model_.nq);
    q_pin.head(nqBase_) = qBase;
    pinocchio::forwardKinematics(model_, data_, q_pin);
    vector<Eigen::Vector3d> hipPoss;
    for (const auto& Id : hipIds_) hipPoss.push_back(data_.oMi[Id].translation());
    return hipPoss;
}

vector<Eigen::Vector3d> LeggedModel::hipPossProjected(const Eigen::VectorXd& qBase){
    auto hipPossProjected = hipPoss(qBase);
    for (auto& Pos: hipPossProjected) Pos[2] = 0;
    return hipPossProjected;
}

Eigen::MatrixXd LeggedModel::jacobian3Dof(Eigen::VectorXd q_pin){
    pinocchio::forwardKinematics(model_, data_, q_pin);
    pinocchio::computeJointJacobians(model_, data_);
    Eigen::MatrixXd jac(3*nContacts3Dof_, model_.nv);
    for (size_t i = 0; i < nContacts3Dof_; ++i) {
        Eigen::MatrixXd jac_temp(6, model_.nv);
        jac_temp.setZero();
        pinocchio::getFrameJacobian(model_, data_, contact3DofIds_[i], pinocchio::LOCAL_WORLD_ALIGNED, jac_temp);
        jac.block(3*i, 0, 3, model_.nv) = jac_temp.topRows<3>();
    }
    return jac;
}

Eigen::MatrixXd LeggedModel::jacobian3DofOrder(const Eigen::VectorXd& jointPos, const Eigen::VectorXd& qBase) {
    Eigen::VectorXd q_pin(this->nqPin()), qJoint(nJoints_);
    LeggedAI::reorder(jointOrder_, jointPos, jointNames_, qJoint);
    if (qBase.size() == 0) {
        q_pin << qBase0(), qJoint;
    } else {
        q_pin << qBase, qJoint;
    }
    Eigen::MatrixXd jac_reordered(3*nContacts3Dof_, 6 + nJoints_);
    jac_reordered = jacobian3Dof(q_pin);
    LeggedAI::reorder_cols(jointNames_, jacobian3Dof(q_pin).rightCols(nJoints_), jointOrder_, jac_reordered.rightCols(nJoints_));
    return jac_reordered;
}

Eigen::MatrixXd LeggedModel::jacobian3DofSimped(const Eigen::VectorXd& jointPos) {
    Eigen::MatrixXd jac(3*nContacts3Dof_, 3);
    jac.setZero();
    auto jac_full = jacobian3DofOrder(jointPos);
    for (size_t i = 0; i < nContacts3Dof_; ++i) {
        // This require that the order of contact3DofNames_ matches the order of joints controlling them!!!
        jac.block(3*i, 0, 3, 3) = jac_full.block(3*i, 6+3*i, 3, 3);
    }

    if (verbose_) {
        cout << "[LeggedModel] Original   3Dof Jacobian:\n" << jac_full << endl;
        cout << "[LeggedModel] Simplified 3Dof Jacobian:\n" << jac << endl;
    }
    return jac;
}

IKStatus LeggedModel::inverseKine3Dof(Eigen::VectorXd qBase, Eigen::VectorXd& q_pin, Eigen::VectorXd qJoints0, vector<Eigen::Vector3d> contact3DofPoss) {
    if (qBase.size() != nqBase_) {
        throw runtime_error("Base pose vector size does not match nqBase_");
    }

    if (qJoints0.size() == 0) {
        // No init qJoints0
        if (verbose_)
            cout << "[LeggedModel] No initial guess provided. Using default guess.\n";
        qJoints0 = (model_.lowerPositionLimit + model_.upperPositionLimit)/2;
        qJoints0 = qJoints0.tail(nJoints_);
    }

    if (contact3DofPoss.empty()) {
        // contact3DofPoss = hipPossProjected(qBase);
        contact3DofPoss = contact3DofPossInit_;
        if (verbose_)
            cout << "[LeggedModel] Auto-generated default foot targets from hip projections." << endl;
    }

    if (contact3DofPoss.size() != contact3DofNames_.size()) {
        throw runtime_error("Mismatch in number of target positions and foot names");
    }

    // TODO don't use 
    Matrix3d R;
    if (baseType_ == "quaternion") {
        // qBase xyzw, quat_ToR require wxyz
        R = quat_ToR(quat_wxyz(qBase.tail(4)));
    }
    else if(baseType_ == "eulerZYX") {
        R = pinocchio::rpy::rpyToMatrix(qBase.tail(3).reverse());
    }

    // contact3DofPoss is feet position in world frame, get foot pos relative to base in base frame using R^t * (contact3DofPoss - base_pos)
    Eigen::VectorXd desEEpos(nContacts3Dof_ * 3);
    for (size_t i = 0; i < contact3DofPoss.size(); i++) {
        desEEpos.segment(3*i, 3) = R.transpose() * (contact3DofPoss[i] - qBase.head(3));
    }

    int max_iters = 1000;
    double tol = 1e-4, dt = 0.1, damping = 1e-6;
    Eigen::VectorXd q(model_.nq); 
    q << qBase0(), qJoints0;
    if (verbose_) cout << "[LeggedModel] IK start from " << q.transpose() << endl;
    // err = [err_foot_1^T, err_foot_2^T, ...]^T
    Eigen::VectorXd err = Eigen::VectorXd::Zero(nContacts3Dof_*3);
    Eigen::VectorXd dqj = Eigen::VectorXd::Zero(model_.nv-6);
    Eigen::MatrixXd Jj = Eigen::MatrixXd::Zero(nContacts3Dof_*3, model_.nv-6);
    for (int i = 0; i < max_iters; i++) {
        pinocchio::forwardKinematics(model_, data_, q);
        pinocchio::updateFramePlacements(model_, data_);

        err = desEEpos;
        for (size_t i = 0; i < contact3DofIds_.size(); i++) {
            err.segment(3*i, 3) -= data_.oMf[contact3DofIds_[i]].translation();
        }

        if (err.norm() < tol) {
            if (verbose_) cout << "[LeggedModel] IK Converged in " << i << " iterations. Final error: " << err.norm() << endl;
            q_pin << qBase, q.tail(nJoints_);
            Eigen::VectorXd qj_max = model_.upperPositionLimit.tail(nJoints_);
            Eigen::VectorXd qj_min = model_.lowerPositionLimit.tail(nJoints_);

            if ( ((q_pin.tail(nJoints_).array() < qj_min.array()) || (q_pin.tail(nJoints_).array() > qj_max.array())).any() ) {
                if (verbose_) {
                    if (baseType_ == "quaternion") {
                        cout << "[LeggedModel] inverseKine3Dof: joint pos out of range." 
                                << "\n qBase: " << qBase.head(3).transpose() << " " << quat2eulerZYX(qBase.tail(4)).transpose() << endl;   
                    } else if (baseType_ == "eulerZYX") {
                        cout << "[LeggedModel] inverseKine3Dof: joint pos out of range." 
                                << "\n qBase: " << qBase.transpose() << endl;  
                    }
                    for (size_t i=0;i<contact3DofNames_.size();++i) {
                        cout << contact3DofNames_[i] << ": " << contact3DofPoss[i].transpose() << endl;
                    }
                    cout << "qJoints: " << q_pin.tail(nJoints_).transpose() << endl;
                    throw runtime_error("[LeggedModel] inverseKine3Dof: joint pos out of range.");
                }
                q_pin.tail(nJoints_) = q_pin.tail(nJoints_).cwiseMax(qj_min).cwiseMin(qj_max);
                std::cout << "[LeggedModel] IK solve successfully, but joint pos out of range, applying clip."
                        << "\norigin:" << q.tail(nJoints_).transpose()
                        << "\ncliped:" << q_pin.tail(nJoints_).transpose() << std::endl;
                return IKStatus::OutOfRange;
            }

            return IKStatus::Success;
        }
        
        Jj = jacobian3Dof(q).rightCols(nJoints_);
        dqj = pseudoInverseDLS(Jj)*err;
        
        q.tail(dqj.size()) += dqj * dt;

        // 将角度包裹到 [-pi, pi]
        for (int j = 0; j < dqj.size(); ++j) {
            double& angle = q[nqBase_ + j];
            angle = atan2(sin(angle), cos(angle));
        }
    }
    return IKStatus::Failure;
}

IKStatus LeggedModel::stanceIK(VectorXd& jointPos, Eigen::Vector3d base_pos, Eigen::Vector3d base_eulerZYX) {
    if (jointPos.size()!=nJoints_) {
        std::cout << "[LeggedModel] stanceIK: joint size doesn't match!\n";
    }

    Eigen::VectorXd qBase(nqBase_), q_pin(this->nqPin());
    if (baseType_ == "quaternion") {
        qBase << base_pos, eulerZYX2QuatVec(base_eulerZYX);
    } else if (baseType_ == "eulerZYX") {
        qBase << base_pos, base_eulerZYX;
    }
    auto status = inverseKine3Dof(qBase, q_pin);
    reorder(jointNames_, q_pin.tail(nJoints_), jointOrder_, jointPos);

    if (true && status!=IKStatus::Success) {
        std::cout << "[LeggedModel] stanceIK: IK solve status: " << (int)status 
        << "\nbase pos: " << base_pos.transpose() << "\n base eulerZYX" << base_eulerZYX.transpose() << std::endl;
    }
    return status;
}

// \dot{q}_j = J_j^+(v - J_b \dot{q}_b)
Eigen::VectorXd LeggedModel::inverseDiffKine3Dof(Eigen::VectorXd q_pin, Eigen::VectorXd vBase, vector<Eigen::Vector3d> contact3DofVels){
    Eigen::VectorXd desEEvel(nContacts3Dof_ * 3);

    if (contact3DofVels.empty()) {
        for (size_t i = 0; i < contact3DofNames_.size(); ++i) {
            contact3DofVels.push_back(Eigen::Vector3d::Zero());
        }
    }

    for (size_t i = 0; i < contact3DofVels.size(); i++) {
        desEEvel.segment(3*i, 3) = contact3DofVels[i];
    }

    auto J = jacobian3Dof(q_pin);
    auto Jb = J.leftCols(6);
    auto Jj = J.rightCols(nJoints_);

    Eigen::VectorXd v_pin(model_.nv);
    v_pin << vBase, pseudoInverseDLS(Jj)*(desEEvel - Jb*vBase);
    return v_pin;
}

