#ifndef LEGGEDMODEL_H
#define LEGGEDMODEL_H

#include "legged_model/LeggedState.h"

#include <Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/rnea.hpp>

using namespace Eigen;
using namespace std;

/**
    * @brief LeggedModel 类，封装了 Pinocchio 模型的基本操作
    * @note baseType_ = "quaternion" 时
                q_pinocchio = [base_pos, base_quaternion(x y z w), q_joint]
                v_pinocchio = [base_linearVel(base), base_angularVel(base), dq_joint]
    * @note baseType_ = "eulerZYX" 时
                q_pinocchio = [base_pos, base_eulerZYX, q_joint]
                v_pinocchio = [base_linearVel(world), base_eulerZYX_dot, dq_joint]
    * @note 使用 pinocchio::rpy 进行旋转变换，需注意 eulerZYX = [yaw pitch roll] = rpy.reverse()
 */
class LeggedModel {
private:
    bool verbose_;

    pinocchio::Model model_;
    pinocchio::Data data_;

    string baseType_;
    size_t nqBase_;
    string baseName_;               // 基座名称

    size_t nJoints_;
    vector<string> jointNames_;     // Joint names in the default Pinocchio model order (alphabetical)
    vector<string> jointOrder_;     // Custom joint ordering used by the controller, different from the Pinocchio model order
    VectorXd qj_min_, qj_max_, tau_max_;    // Joint torque limits

    // 3 Dof end effector
    size_t nContacts3Dof_;
    vector<string> contact3DofNames_;
    vector<size_t> contact3DofIds_;

    // 6 Dof end effector
    size_t nContacts6Dof_;
    vector<string> contact6DofNames_;
    vector<size_t> contact6DofIds_;

    vector<string> hipNames_;
    vector<size_t> hipIds_;

public:
    void setVerbose() {verbose_ = true;}
    void unsetVerbose() {verbose_ = false;}

    const pinocchio::Model& model() const {return model_;}
    pinocchio::Data& data() {return data_;}

    const string& baseType() const {return baseType_;}
    size_t nqBase() const {return  nqBase_;}
    VectorXd qBase0() const {
        if (baseType_ == "quaternion") {
            VectorXd qBase0(nqBase_);
            qBase0 << 0, 0, 0, 0, 0, 0, 1;
            return qBase0;
        }
        else if(baseType_ == "eulerZYX") {
            VectorXd qBase0(nqBase_);
            qBase0 << 0, 0, 0, 0, 0, 0;
            return qBase0;
        }
        else {
            throw runtime_error("[LeggedModel] qBase0: unknown baseType_");
        }
    }

    size_t nDof() const {return  nJoints_ + 6;}
    size_t nJoints() const {return  nJoints_;}
    const vector<string>& jointNames() const {return jointNames_;}
    const vector<string>& jointOrder() const {return jointOrder_;}
    const VectorXd& qjMin() const {return qj_min_;}
    const VectorXd& qjMax() const {return qj_max_;}
    const VectorXd& tauMax() const {return tau_max_;}
    void setJointLimits(VectorXd qj_max, VectorXd qj_min){
        if (qj_max.size() != nJoints_ || qj_min.size() != nJoints_) {
            throw runtime_error("[LeggedModel] setJointLimits: qMax/qMin vector size does not match nJoints_");
        }

        for(size_t i=0;i<nJoints_;++i){
            model_.lowerPositionLimit[nqBase_ + i] = qj_min[i];
            model_.upperPositionLimit[nqBase_ + i] = qj_max[i];
        }
    }

    size_t nContacts3Dof() const {return  nContacts3Dof_;}
    const vector<string>& contact3DofNames() const {return  contact3DofNames_;}
    const vector<size_t>& contact3DofIds() const {return  contact3DofIds_;}
    vector<Vector3d> contact3DofPoss(const VectorXd& q_pin);
    vector<Vector3d> contact3DofVels(const VectorXd& q_pin, const VectorXd& v_pin);
    // jointPos is in custom order, return contact3DofPoss in order of contact3DofNames_ and stacked as [ee1_pos, ee2_pos, ...]
    VectorXd contact3DofPossOrder(const VectorXd& jointPos, const VectorXd& qBase = VectorXd());
    
    size_t nContacts6Dof() const {return  nContacts6Dof_;}
    const vector<string>& contact6DofNames() const {return  contact6DofNames_;}
    const vector<size_t>& contact6DofIds() const {return  contact6DofIds_;}
    vector<Vector3d> contact6DofPoss(const VectorXd& q_pin);
    vector<Vector3d> contact6DofVels(const VectorXd& q_pin, const VectorXd& v_pin);

    vector<Vector3d> hipPoss(const VectorXd& qBase);
    vector<Vector3d> hipPossProjected(const VectorXd& qBase);
    
    Vector3d com(const VectorXd& q_pin) {return pinocchio::centerOfMass(model_, data_, q_pin);}
    Vector3d vcom(const VectorXd& q_pin, const VectorXd& v_pin) {
        pinocchio::centerOfMass(model_, data_, q_pin, v_pin);
        return data_.vcom[0];
    }
    VectorXd hcom(const VectorXd& q_pin, const VectorXd& v_pin) {
        pinocchio::computeCentroidalMomentum(model_, data_, q_pin, v_pin);
        return data_.hg.toVector();
    }

    /*
        stack jacobian of all 3Dof contact point, [3*nContacts3Dof_, nDof]
    */
    MatrixXd jacobian3Dof(VectorXd q_pin);
    MatrixXd jacobian3DofOrder(const VectorXd& jointPos, const VectorXd& qBase = VectorXd());
    MatrixXd jacobian3DofSimped(const VectorXd& jointPos);

    VectorXd inverseKine3Dof(VectorXd qBase, VectorXd qJoints0 = VectorXd(), vector<Vector3d> contact3DofPoss = {}) {
        VectorXd q_pin(nqBase_ + nJoints_), qJoints(nJoints_);
        inverseKine3Dof(qBase, qJoints, qJoints0, contact3DofPoss);
        q_pin << qBase, qJoints;
        return q_pin;
    }
    bool inverseKine3Dof(VectorXd qBase, VectorXd& qJoints, VectorXd qJoints0 = VectorXd(), vector<Vector3d> contact3DofPoss = {});
    VectorXd inverseDiffKine3Dof(VectorXd q_pin, VectorXd vBase, vector<Vector3d> contact3DofVels = {});

    // Dynamics
    VectorXd g(const VectorXd& q_pin) {
        return pinocchio::computeGeneralizedGravity(model_, data_, q_pin);
    };

    VectorXd nle(const VectorXd& q_pin, const VectorXd& v_pin) {
        return pinocchio::nonLinearEffects(model_, data_, q_pin, v_pin);
    };

    void loadConfig(const YAML::Node& node);
    void loadUrdf(string urdfPath, string baseType, string baseName, 
        vector<string> contact3DofNames = {},
        vector<string> contact6DofNames = {},
        vector<string> hipNames = {}, 
        bool verbose = false);

    // This function call createCustomState of leggedState to create a custom state (consistent with q,v used in LeggedModel) in leggedState 
    void creatPinoState(LeggedState& leggedState) const {
        // creat q_pinocchio and v_pinocchio
        if (baseType_ == "quaternion") {
            leggedState.createCustomState("q_pin", {"base_pos", "base_quat", "joint_pos"}, jointNames_);
            leggedState.createCustomState("v_pin", {"base_lin_vel_B", "base_ang_vel_B", "joint_vel"}, jointNames_);
        } else if (baseType_ == "eulerZYX") {
            leggedState.createCustomState("q_pin", {"base_pos", "base_eulerZYX", "joint_pos"}, jointNames_);
            leggedState.createCustomState("v_pin", {"base_lin_vel_W", "base_eulerZYX_dot", "joint_vel"}, jointNames_);
        }
        leggedState.createCustomState("eePos_pin", {"ee3Dof_pos", "ee6Dof_pos"}, {}, contact3DofNames_, contact6DofNames_);
        leggedState.createCustomState("eeVel_pin", {"ee3Dof_vel", "ee6Dof_vel"}, {}, contact3DofNames_, contact6DofNames_);
        leggedState.createCustomState("f_pin", {"ee3Dof_fc", "ee6Dof_fc"}, {}, contact3DofNames_, contact6DofNames_);
    }
};

#endif // LEGGEDMODEL_H
