/**
 * @file wrist_driver.hpp
 * @author Lorenzo Shaikewitz
 * @brief Main node for receiving and executing joint commands for wrist driver.
 * @version 0.1
 * @date 2022-06-29
 * 
 * @copyright Copyright (c) 2022
 * 
 * Using this class:
 * Navigate to the workspace (robotis_ws on the NUC)
 * 1) source the workspace (. install/local_setup.bash)
 * 2) Plug in the wrist joint. Then add port permissions: sudo chmod a+rw /dev/ttyUSB0
 * 3) Run the driver! Make sure power is connected to the fork, then run:
 *      ros2 run wrist_driver_ros2 wrist_driver
 * 
 * Once the driver is running you can receive the current state and transmit a new state.
 */

#ifndef WRIST_DRIVER_HPP_
#define WRIST_DRIVER_HPP_

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>

// #include "rclcpp/rclcpp.hpp"
// #include "rcutils/cmdline_parser.h"
#include "dynamixel_sdk/dynamixel_sdk.h"
#include "wrist_driver_constants.hpp"

// msg
#include "wrist_driver_interfaces/SimpleJointAngleCommand.h"
#include "wrist_driver_interfaces/WristState.h"
#include "wrist_driver_interfaces/SetWristMode.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Quaternion.h"


class WristDriverNode {
public:
    // message shortcuts
    using SimpleJointAngleCommandMsg = wrist_driver_interfaces::SimpleJointAngleCommand;
    using WristStateMsg = wrist_driver_interfaces::WristState;
    using SetWristModeRequest = wrist_driver_interfaces::SetWristModeRequest;
    using SetWristModeResponse = wrist_driver_interfaces::SetWristModeResponse;
    using PoseMsg = geometry_msgs::PoseStamped;

    WristDriverNode(ros::NodeHandle nh);
    virtual ~WristDriverNode();

private:
    int mode_{0};
    // dynamixel class variables
    dynamixel::GroupSyncRead readPosition_;
    dynamixel::GroupSyncRead readVelocity_;
    dynamixel::GroupSyncRead readCurrent_;

    // writes desired joint angle to dynamixels
    bool writeJt0(double q);
    bool writeJt1(double q);

    // callback for /wrist_state publishing (TODO: UPDATE RATE)
    // rclcpp::TimerBase::SharedPtr timer_wristState_;
    ros::Timer timer_wristState_;
    void cb_wristState(const ros::TimerEvent& event);

    // ros callback for /wrist_pose publishin
    // void wrist_state_callback(const rclcpp::TimerBase::SharedPtr timer);

    // callback for /cmd_wrist_joint_angles subscriber
    // Example: ros2 topic pub -1 /cmd_wrist_joint_angles wrist_driver_interfaces/SimpleJointAngleCommand "{q0: 300, q1: 1000}"
    void cb_jointAngleCmd(const SimpleJointAngleCommandMsg &msg);

    // callback for /set_wrist_mode
    bool cb_setWristMode(SetWristModeRequest &request, SetWristModeResponse &response);


    // publishers
    ros::Publisher pub_wristState_; 
    ros::Publisher pub_wristPose_;
    // rclcpp::Publisher<WristStateMsg>::SharedPtr pub_wristState_;
    // rclcpp::Publisher<PoseMsg>::SharedPtr pub_wristPose_;

    // subscribers
    ros::Subscriber sub_jointAngleCommand_;
    // rclcpp::Subscription<SimpleJointAngleCommandMsg>::SharedPtr sub_jointAngleCommand_;

    // services
    ros::ServiceServer srv_setWristMode_;
    // rclcpp::Service<SetWristModeSrv>::SharedPtr srv_setWristMode_;

};

#endif
