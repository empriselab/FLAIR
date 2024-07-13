/**
 * @file getTorques.cpp
 * @author Lorenzo Shaikewitz
 * @brief Version of getTorques that uses extra two joints of end effector
 * @version 0.1
 * @date 2022-07-11
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <urdf/model.h>

/*
Roadmap:
1) use fkin to compute last two columns of Jacobian (for additional 2 joints, see 133a notes)
2) connect this with ROS so it can receive q, dq, transforms
3) Modify to compute trajectory of all joints, assuming coriolis = 0.

*/


class Kinematics {
public:
    Kinematics(std::string urdf_filename) {
        // load the URDF
        urdf::Model model;
        model.initFile(urdf_filename);

        // pull out joints
        std::cout << "Successfully parsed URDF file.";
    }   

private:

};


franka::Torques CartesianImpedanceController::getTorques(const franka::RobotState& state, franka::Duration /*period*/) {
    // static long j = 0;
    Matrix4x4d pose = Matrix4x4d(state.O_T_EE.data());
    Matrix4x4d pose_d = pose_interp_.getNext(state, globals_->pose_command.load());

    // get start / goal variables
    Vector3d position = pose.block<3,1>(0,3);
    Vector3d position_d = pose_d.block<3,1>(0,3);
    Eigen::Quaterniond orientation(pose.block<3,3>(0,0)); //This extracts rotation matrix
    Eigen::Quaterniond orientation_d(pose_d.block<3,3>(0,0));
    orientation.normalize();
    orientation_d.normalize();

    // get state variables
    std::array<double, 7> coriolis_array = model_->coriolis(state);
    std::array<double, 42> jacobian_array =
        model_->zeroJacobian(franka::Frame::kEndEffector, state);
    
    // convert to Eigen
    Eigen::Map<const Vector7d> coriolis(coriolis_array.data());
    Eigen::Map<const Matrix6x7d> jacobian(jacobian_array.data());

    Eigen::Map<const Vector7d> dq(state.dq.data());
    auto ee_vel = jacobian * dq;  // forward kinematics to get "measured" end effector velocitys

    // errors
    Vector3d error_pos = position - position_d;
    
    // get shortest difference
    if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
        orientation.coeffs() << -orientation.coeffs();
    }
    // true "difference" quaternion
    Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d);

    // error
    Vector6d error, desired_acc, desired_wrench;
    error.head(3) << error_pos;
    error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();

    // Transform rotation to base frame
    error.tail(3) << -pose.block<3,3>(0,0) * error.tail(3);

    // xddot = - kp * error - kv * vel
    for (int i = 0; i < 6; ++i) desired_acc[i] = - params_.kp[i] * error[i] - params_.kv[i] * ee_vel[i];

    // wrench = M_op * xddot (TODO decoupling option), i found that including inertia Mop makes things a lot worse...
    desired_wrench = desired_acc;

    // compute control
    Vector7d tau_task, tau_d;
    // Spring damper system with damping ratio=1
    tau_task << jacobian.transpose() * desired_wrench;

    // tau_task = Vector7d::Zero();

    // clip
    for (size_t i = 0; i < tau_task.size(); i++)
    {
        if (std::abs(tau_task(i)) > TAU_TASK_MAX) tau_task(i) = copysign(TAU_TASK_MAX, tau_task(i));
    }

    tau_d << tau_task + coriolis;

    std::array<double, 7> tau_d_array{};
    Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;
    return tau_d_array;
}