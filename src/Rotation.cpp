// Kunzhao Ren, LeggedAI Lab, Inc. 2026

#include "legged_model/Rotation.h"
#include <cmath>

namespace LeggedAI {

using namespace std;
using namespace Eigen;

Quaterniond eulerZYX2Quat(const Vector3d& eulerZYX)
{
    // eulerZYX = [yaw, pitch, roll]
    const AngleAxisd yaw  (eulerZYX(0), Vector3d::UnitZ());
    const AngleAxisd pitch(eulerZYX(1), Vector3d::UnitY());
    const AngleAxisd roll (eulerZYX(2), Vector3d::UnitX());

    return yaw * pitch * roll; // Eigen quaternion (wxyz internal), coeffs() is [x y z w]
}

Vector4d eulerZYX2QuatVec(const Vector3d& eulerZYX)
{
    return eulerZYX2Quat(eulerZYX).coeffs(); // [x y z w]
}

Vector3d eulerZYXDot2AngVelW(const Vector3d& eulerZYX,
                            const Vector3d& eulerZYX_dot)
{
    const double yaw   = eulerZYX(0);
    const double pitch = eulerZYX(1);

    const double sz = sin(yaw);
    const double cz = cos(yaw);
    const double sy = sin(pitch);
    const double cy = cos(pitch);

    const double dyaw   = eulerZYX_dot(0);
    const double dpitch = eulerZYX_dot(1);
    const double droll  = eulerZYX_dot(2);

    // ω_W (world frame)
    const double wx = -sz * dpitch + cy * cz * droll;
    const double wy =  cz * dpitch + cy * sz * droll;
    const double wz =  dyaw - sy * droll;

    return Vector3d(wx, wy, wz);
}

Vector3d angVelW2EulerZYXDot(const Vector3d& eulerZYX,
                            const Vector3d& base_ang_vel_W)
{
    const double yaw   = eulerZYX(0);
    const double pitch = eulerZYX(1);

    const double sz = sin(yaw);
    const double cz = cos(yaw);
    const double sy = sin(pitch);
    const double cy = cos(pitch);

    const double wx = base_ang_vel_W(0);
    const double wy = base_ang_vel_W(1);
    const double wz = base_ang_vel_W(2);

    // singular at pitch = ±pi/2  (cy -> 0)
    const double tmp = (cz * wx + sz * wy) / cy;

    const double dyaw   = sy * tmp + wz;
    const double dpitch = -sz * wx + cz * wy;
    const double droll  = tmp;

    return Vector3d(dyaw, dpitch, droll);
}

} // namespace LeggedAI
