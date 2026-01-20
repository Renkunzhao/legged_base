#include "legged_model/LeggedModel.h"

#include <Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <yaml-cpp/yaml.h>
#include <pinocchio/algorithm/joint-configuration.hpp>

void solveStanceIKInteractive(LeggedModel& leggedModel)
{
    double x, y, z, yaw, pitch, roll;

    std::cout << "Enter: x y z yaw pitch roll (rad). Example:\n"
              << "  0.1 0.0 0.35 0.0 0.15 -0.2\n"
              << "Ctrl+D to exit.\n";

    while (true) {
        std::cout << "> ";
        if (!(std::cin >> x >> y >> z >> yaw >> pitch >> roll)) {
            std::cout << "\n[LeggedModel] input ended.\n";
            break;
        }

        Eigen::VectorXd base_pos(3), base_eulerZYX(3);
        Eigen::VectorXd jointPos(leggedModel.nJoints());

        base_pos << x, y, z;
        base_eulerZYX << yaw, pitch, roll;   // [yaw, pitch, roll]

        auto status = leggedModel.stanceIK(jointPos, base_pos, base_eulerZYX);

        std::cout << "[LeggedModel] stanceIK solve " << (status==IKStatus::Success ? "SUCCESS" : "FAIL")
                  << "\n  base_pos      : " << base_pos.transpose()
                  << "\n  base_eulerZYX : " << base_eulerZYX.transpose()
                  << "\n";

        if (status==IKStatus::Success) {
            std::cout << "  jointPos      : " << jointPos.transpose() << "\n";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char **argv)
{
    std::string configFile;
    if (argc!=2) throw std::runtime_error("[LeggedModel_test] configFile path required.");
    else configFile = argv[1];
    std::cout << "[LeggedModel_test] Load config from " << configFile << std::endl;

    YAML::Node configNode = YAML::LoadFile(configFile);

    LeggedModel leggedModel;
    leggedModel.loadConfig(configNode);

    std::cout << "[LeggedModel]: " << "nDof " << leggedModel.nDof() << std::endl;
    std::cout << "[LeggedModel]: " << "nJoints " << leggedModel.nJoints() << std::endl;
    std::cout << "[LeggedModel]: " << "nContacts3Dof " << leggedModel.nContacts3Dof() << std::endl;
    std::cout << "[LeggedModel]: " << "nContacts6Dof " << leggedModel.nContacts6Dof() << std::endl;

    for(size_t i=0;i<leggedModel.nContacts3Dof();i++) std::cout << "[LeggedModel]: " << leggedModel.contact3DofNames()[i] << ": " << leggedModel.contact3DofIds()[i] << std::endl;
    for(size_t i=0;i<leggedModel.nContacts6Dof();i++) std::cout << "[LeggedModel]: " << leggedModel.contact6DofNames()[i] << ": " << leggedModel.contact6DofIds()[i] << std::endl;

    std::cout << "--- Position limits ---\n" 
                << leggedModel.model().lowerPositionLimit.transpose() << "\n" 
                << leggedModel.model().upperPositionLimit.transpose() << std::endl;

    Eigen::VectorXd q_rand = pinocchio::randomConfiguration(leggedModel.model());
    auto contact3DofPoss = leggedModel.contact3DofPoss(q_rand);
    std::cout << "[LeggedModel]: " << "q_rand " << q_rand.transpose() << std::endl;
    for(size_t i=0;i<leggedModel.nContacts3Dof();i++) std::cout << "[LeggedModel]: " << leggedModel.contact3DofNames()[i] << ": " << contact3DofPoss[i].transpose() << std::endl;

    Eigen::VectorXd q_pin(leggedModel.nqPin());
    auto flag = leggedModel.inverseKine3Dof(q_rand.head(leggedModel.nqBase()), q_pin, VectorXd(), contact3DofPoss);
    std::cout << "[LeggedModel]: " << "q_pin " << q_pin.transpose() << std::endl;
    std::cout << "[LeggedModel]: " << "err " << (q_rand-q_pin).tail(leggedModel.nJoints()).norm() << std::endl;

    // stance IK range test
    solveStanceIKInteractive(leggedModel);
}
