
#include <iostream>
#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "legged_base/Utils.h"

int main(int argc, char* argv[]){
    // Get urdf path from command line arguments
    std::string urdfPath;
    if(argc > 1) {
        urdfPath = argv[1];
    } else {
        // TODO: 用 LeggedAI::getEnv
        urdfPath = LeggedAI::getEnv("WORKSPACE") + "/src/legged_base/urdf/g1.urdf"; // Default path
    }
    
    // Build model from URDF
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdfPath, pinocchio::JointModelFreeFlyer(), model);
    
    // Print joint names
    std::cout << "Number of joints: " << model.njoints << std::endl;
    for(int i = 0; i < model.njoints; ++i) {
        std::cout << "Joint " << i << ": " << model.names[i] << std::endl;
    }
    
    return 0;
}