#include "legged_model/Rotation.h"
#include <iostream>

using namespace Eigen;
using namespace LeggedAI;

int main(){

    Vector3d eulerZYX;
    eulerZYX << 0.3, -0.2, 0.5;

    std::cout << "eulerZYX (origin)  = " << eulerZYX.transpose() << std::endl;
    std::cout << "eulerZYX (inverse) = " << quat2eulerZYX(eulerZYX2QuatVec(eulerZYX)).transpose() << std::endl;

}
