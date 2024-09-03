/**
 * @file wrist_driver.cpp
 * @author Lorenzo Shaikewitz
 * @brief Main node for receiving and executing joint commands for wrist driver.
 * @version 0.1
 * @date 2022-06-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <cstdio>
#include <memory>
#include <string>
#include <array>
#include <functional>
#include <cmath>
#include <algorithm>

#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/mman.h>

#include <stdlib.h>
#include <limits.h>
#include <malloc.h>
#include <sys/resource.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <stdint.h>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <csignal>

// #include "rclcpp/rclcpp.hpp"
// #include "rcutils/cmdline_parser.h"
// #include "dynamixel_sdk/dynamixel_sdk.h"

#include "wrist_driver.hpp"
#include "wrist_driver_constants.hpp"

using std::placeholders::_1;
using std::placeholders::_2;
using namespace std::chrono_literals; // TODO: REMOVE

dynamixel::PortHandler* portHandler;
dynamixel::PacketHandler* packetHandler;

// forward declarations
boost::array<double, 16> fkin(const boost::array<double, 2>& q);
geometry_msgs::Quaternion getForceQuat(boost::array<double, 16>& T);

// Constructor
WristDriverNode::WristDriverNode(ros::NodeHandle nh) :
        readPosition_(portHandler, packetHandler, ADDR_CURR_POSITION, NUM_BYTES),
        readVelocity_(portHandler, packetHandler, ADDR_CURR_VELOCITY, NUM_BYTES),
        readCurrent_ (portHandler, packetHandler, ADDR_CURR_CURRENT , NUM_BYTES_CURRENT) {
            
    // ROS_INFO("wrist_driver_node started!");

    // QOS settings
    // this->declare_parameter("qos_depth", 10);
    // int8_t qos_depth = 0;
    // this->get_parameter("qos_depth", qos_depth);
    // const auto QOS_RKL10V = rclcpp::QoS(rclcpp::KeepLast(qos_depth)).reliable().durability_volatile();

    // set up sync read parameters
    readPosition_.addParam(JT0_ID); readPosition_.addParam(JT1_ID);
    readVelocity_.addParam(JT0_ID); readVelocity_.addParam(JT1_ID);
    readCurrent_.addParam(JT0_ID);  readCurrent_.addParam(JT1_ID);

    // change wrist mode to velocity control
    // auto proxy_wrist_mode_request = SetWristModeRequest();
    // proxy_wrist_mode_request.mode = 1;
    // auto proxy_wrist_mode_response = SetWristModeResponse();
    // cb_setWristMode(proxy_wrist_mode_request, proxy_wrist_mode_response);

    // publishers
    pub_wristState_ = nh.advertise<sensor_msgs::JointState>("wrist_joint_states", 10);
    // pub_wristState_ = this->create_publisher<WristStateMsg>(
    //     "wrist_state", QOS_RKL10V);
    pub_wristPose_ = nh.advertise<PoseMsg>("wrist_cartesian_states", 10);
    // pub_wristPose_  = this->create_publisher<PoseMsg>(
        // "franka_panda/wrist_pose", QOS_RKL10V);
    // timer_wristState_ = this->create_wall_timer(100ms, std::bind(&WristDriverNode::cb_wristState, this));
    timer_wristState_ = nh.createTimer(ros::Duration(0.1),  &WristDriverNode::cb_wristState, this);

    // subscribers
    sub_jointAngleCommand_ = nh.subscribe(
        "cmd_wrist_joint_angles", 10, &WristDriverNode::cb_jointAngleCmd, this);
    // sub_jointAngleCommand_ = 
        // this->create_subscription<SimpleJointAngleCommandMsg>(
            // "cmd_wrist_joint_angles", QOS_RKL10V, 
            // std::bind(&WristDriverNode::cb_jointAngleCmd, this, _1));

    // services
    srv_setWristMode_ = nh.advertiseService("set_wrist_mode", &WristDriverNode::cb_setWristMode, this);
    // srv_setWristMode_ = this->create_service<SetWristModeSrv>(
    //     "set_wrist_mode", std::bind(&WristDriverNode::cb_setWristMode, this, _1, _2));
}

WristDriverNode::~WristDriverNode() {/*does nothing*/}

// publishes to /wrist_state. 1 kHz timer callback
 void WristDriverNode::cb_wristState(const ros::TimerEvent& event) {
    // lambda to accelerate setup
    auto setupRead = [this](dynamixel::GroupSyncRead& reader, uint8_t addr, int num_bytes) {
    // read from dynamixels
    int comm_result = reader.txRxPacket();
    if (comm_result != COMM_SUCCESS)
        ROS_ERROR("%s", packetHandler->getTxRxResult(comm_result));
    // make sure both motors have data available
    if (!reader.isAvailable(JT0_ID, addr, num_bytes))
        ROS_ERROR("Could not read from joint 0.");
    if (!reader.isAvailable(JT1_ID, addr, num_bytes))
        ROS_ERROR("Could not read from joint 1.");
    if (!reader.isAvailable(JT0_ID, addr, num_bytes) || !reader.isAvailable(JT1_ID, addr, num_bytes))
    {
        ROS_ERROR("Could not read from both joints -- shutting down node.");
        ros::shutdown();
    }
    };

    setupRead(readPosition_, ADDR_CURR_POSITION, NUM_BYTES);
    setupRead(readVelocity_, ADDR_CURR_VELOCITY, NUM_BYTES);
    setupRead(readCurrent_ , ADDR_CURR_CURRENT, NUM_BYTES_CURRENT);

    int raw_q0{ readPosition_.getData(JT0_ID, ADDR_CURR_POSITION, NUM_BYTES) };
    int raw_q1{ readPosition_.getData(JT1_ID, ADDR_CURR_POSITION, NUM_BYTES) };

    // actually get the data
    // auto msg = WristStateMsg();
    // msg.q[0]  = (raw_q0 - JT0_CENTER_POSITION) * PULSE_TO_RAD;
    // msg.q[1]  = (raw_q1 - JT1_CENTER_POSITION) * PULSE_TO_RAD;
    // msg.dq[0] = readPosition_.getData(JT0_ID, ADDR_CURR_VELOCITY, NUM_BYTES) * RAWVEL_TO_RAD;
    // msg.dq[1] = readPosition_.getData(JT1_ID, ADDR_CURR_VELOCITY, NUM_BYTES) * RAWVEL_TO_RAD;
    // msg.current[0] = readPosition_.getData(JT0_ID, ADDR_CURR_CURRENT, NUM_BYTES_CURRENT);
    // msg.current[1] = readPosition_.getData(JT1_ID, ADDR_CURR_CURRENT, NUM_BYTES_CURRENT);

    // msg.relpose = fkin(msg.q);

    // pub_wristState_.publish(msg);

    auto joint_state = sensor_msgs::JointState();
    joint_state.header.stamp = ros::Time::now();
    
    joint_state.name.push_back("forkpitch_joint");
    joint_state.position.push_back((raw_q0 - JT0_CENTER_POSITION) * PULSE_TO_RAD);
    joint_state.velocity.push_back(readPosition_.getData(JT0_ID, ADDR_CURR_VELOCITY, NUM_BYTES) * RAWVEL_TO_RAD);
    joint_state.effort.push_back(0);

    joint_state.name.push_back("forkroll_joint");
    joint_state.position.push_back((raw_q1 - JT1_CENTER_POSITION) * PULSE_TO_RAD);
    joint_state.velocity.push_back(readPosition_.getData(JT1_ID, ADDR_CURR_VELOCITY, NUM_BYTES) * RAWVEL_TO_RAD);
    joint_state.effort.push_back(0);

    pub_wristState_.publish(joint_state);

    // auto posemsg = PoseMsg();
    // posemsg.header.stamp = ros::Time::now();
    // posemsg.pose.position.x = msg.relpose[3];
    // posemsg.pose.position.y = msg.relpose[7];
    // posemsg.pose.position.z = msg.relpose[11];
    // posemsg.pose.orientation = getForceQuat(msg.relpose);

    // pub_wristPose_.publish(posemsg);
 }


// Callback for /cmd_wrist_joint_angles
void WristDriverNode::cb_jointAngleCmd(const SimpleJointAngleCommandMsg &msg) {
    // NOTE: SYNC WRITE DOES NOT WORK WHEN USING ONE IN EXTENDED POSITION CONTROL MODE
    double q0 = msg.q0; // msg->q_desired[0];
    double q1 = msg.q1; // msg->q_desired[1];
    
    // ROS_INFO("Writing joint 0 to %.2f rad(/s) and joint 1 to %0.2f rad(/s).", q0, q1);
    
    writeJt0(q0);
    writeJt1(q1);
}


// Writes joint angles for joint 0
bool WristDriverNode::writeJt0(double q) {
    switch (mode_) {
        case 0: // joint control mode 
        {
            // convert to pulse
            uint32_t pulse = lround(q / PULSE_TO_RAD) + JT0_CENTER_POSITION;

            // // check range of pulse
            // if ((pulse > JT0_MAX_POSITION) || (pulse < JT0_MIN_POSITION)) {
            //     ROS_INFO("Joint 0 goal position %d is out of range.", pulse);
            //     return false;
            // }

            // Write!
            uint8_t dxl_error = 0;
            // write position for joint 0
            int comm_result = packetHandler->write4ByteTxRx(
                portHandler,
                JT0_ID,
                ADDR_GOAL_POSITION,
                pulse,
                &dxl_error
            );
            if (comm_result != COMM_SUCCESS) {
                ROS_ERROR("Error writing to joint 0: %s", packetHandler->getTxRxResult(comm_result));
                return false;
            }
            else if (dxl_error != 0) {
                ROS_ERROR("Error reading joint 0: %s", packetHandler->getRxPacketError(dxl_error));
                return false;
            }
	    break;
        }
        case 1: // velocity control mode
        {
            // convert to pulse
            uint32_t pulse = lround(q / RAWVEL_TO_RAD);
            // ROS_INFO("%d", pulse);

            // Write!
            uint8_t dxl_error = 0;
            // write position for joint 0
            int comm_result = packetHandler->write4ByteTxRx(
                portHandler,
                JT0_ID,
                ADDR_GOAL_VELOCITY,
                pulse,
                &dxl_error
            );
            if (comm_result != COMM_SUCCESS) {
                ROS_ERROR("Error writing to joint 0: %s", packetHandler->getTxRxResult(comm_result));
                return false;
            }
            else if (dxl_error != 0) {
                ROS_ERROR("Error reading joint 0: %s", packetHandler->getRxPacketError(dxl_error));
                return false;
            }
            break;
        }
        
    }
    return true;
}


// Writes joint angles for joint 1
bool WristDriverNode::writeJt1(double q) {
    switch (mode_) {
        case 0: // joint control mode 
        {
            // convert to pulse
            uint32_t pulse = lround(q / PULSE_TO_RAD) + JT1_CENTER_POSITION;

            // check range of pulse (TODO)
            // if ((pulse > JT1_MAX_POSITION) || (pulse < JT1_MIN_POSITION)) {
            //     ROS_INFO("Joint 1 goal position %d is out of range.", pulse);
            //     return false;
            // }

            // Write!
            uint8_t dxl_error = 0;
            // write position for joint 1
            int comm_result = packetHandler->write4ByteTxRx(
                portHandler,
                JT1_ID,
                ADDR_GOAL_POSITION,
                pulse,
                &dxl_error
            );
            if (comm_result != COMM_SUCCESS) {
                ROS_ERROR("Error writing to joint 1: %s", packetHandler->getTxRxResult(comm_result));
                return false;
            }
            else if (dxl_error != 0) {
                ROS_ERROR("Error reading joint 1: %s", packetHandler->getRxPacketError(dxl_error));
                return false;
            }
            break;
        }
        case 1: // velocity control mode
        {
            // convert to pulse
            uint32_t pulse = lround(q / RAWVEL_TO_RAD);
            // ROS_INFO("%d", pulse);

            // Write!
            uint8_t dxl_error = 0;
            // write position for joint 1
            int comm_result = packetHandler->write4ByteTxRx(
                portHandler,
                JT1_ID,
                ADDR_GOAL_VELOCITY,
                pulse,
                &dxl_error
            );
            if (comm_result != COMM_SUCCESS) {
                ROS_ERROR("Error writing to joint 1: %s", packetHandler->getTxRxResult(comm_result));
                return false;
            }
            else if (dxl_error != 0) {
                ROS_ERROR("Error reading joint 1: %s", packetHandler->getRxPacketError(dxl_error));
                return false;
            }
            break;
        }
        
    }
    return true;
}


/* Begin non-class functions (TODO: Move to other file) */

// Use control_mode to change between position, extended position, current, etc.
bool setupDynamixel(uint8_t dxl_id, uint8_t control_mode = 3)
{
    uint8_t dxl_error = 0;
    int dxl_comm_result = COMM_TX_FAIL;

    // Use Position Control Mode
    dxl_comm_result = packetHandler->write1ByteTxRx(
        portHandler,
        dxl_id,
        ADDR_OPERATING_MODE,
        control_mode,
        &dxl_error
    );

    if (dxl_comm_result != COMM_SUCCESS) {
        ROS_ERROR("Failed to set mode.");
        return false;
    } 
    // else {
        // ROS_INFO("Succeeded to set mode.");
    // }

    // Enable Torque of DYNAMIXEL
    dxl_comm_result = packetHandler->write1ByteTxRx(
        portHandler,
        dxl_id,
        ADDR_TORQUE_ENABLE,
        1,
        &dxl_error
    );

    if (dxl_comm_result != COMM_SUCCESS) {
        ROS_ERROR("Failed to enable torque.");
        return false;
    } 
    // else {
        // ROS_INFO("Succeeded to enable torque.");
    // }

    return true;
}


boost::array<double, 16> fkin(const boost::array<double, 2>& q) {
    double q1 = q[0];
    double q2 = q[1];

    double s1 = std::sin(q1);
    double s2 = std::sin(q2);
    double c1 = std::cos(q1);
    double c2 = std::cos(q2);

    boost::array<double, 16> transform{
         c1*c2, -c1*s2, s1,  (DIST_JT0_JT1 + DIST_JT1_EFF)*s1 + DIST_OOP_FORK*c2,
            s2,     c2,  0,  DIST_OOP_FORK*s2,
        -s1*c2,  s1*s2, c1,  DIST_TOP_JT0 + (DIST_JT0_JT1 + DIST_JT1_EFF)*c1,
        0, 0, 0, 1
    };

    return transform;
}

geometry_msgs::Quaternion getForceQuat(boost::array<double, 16>& T) {
    boost::array<double, 4> A{
        1. + T[0] + T[5] + T[10],
        1. + T[0] - T[5] - T[10],
        1. - T[0] + T[5] - T[10],
        1. - T[0] - T[5] + T[10]
    };

    double* maxapointer = std::max_element(A.begin(), A.end());
    int i = std::distance(A.begin(),maxapointer);
    double maxa = *maxapointer;

    double c = 0.5/std::sqrt(maxa);
    auto q = geometry_msgs::Quaternion();
    if (i == 0) {
        q.w = c*maxa;
        q.x = c*(T[9] - T[6]);
        q.y = c*(T[2] - T[8]);
        q.z = c*(T[4] - T[1]);
    } else if (i == 1) {
        q.w = c*(T[9] - T[6]);
        q.x = c*maxa;
        q.y = c*(T[4] + T[1]);
        q.z = c*(T[2] + T[8]);
    } else if (i == 2) {
        q.w = c*(T[2] - T[8]);
        q.x = c*(T[4] + T[1]);
        q.y = c*maxa;
        q.z = c*(T[9] + T[6]);
    } else {
        q.w = c*(T[4] - T[1]);
        q.x = c*(T[2] + T[8]);
        q.y = c*(T[9] + T[6]);
        q.z = c*maxa;
    }

    // CONVERT quaternion into F/T coordinates by rotating it by 90 degrees about z
    boost::array<double, 4> qrot{     std::sqrt(2)/2, 0, 0, std::sqrt(2)/2 };
    boost::array<double, 4> qprod1{ qrot[0]*q.w - qrot[1]*q.x - qrot[2]*q.y - qrot[3]*q.z,
                                  qrot[0]*q.x + qrot[1]*q.w + qrot[2]*q.z - qrot[3]*q.y,
                                  qrot[0]*q.y - qrot[1]*q.z + qrot[2]*q.w + qrot[3]*q.x,
                                  qrot[0]*q.z + qrot[1]*q.y - qrot[2]*q.x + qrot[3]*q.w };
    boost::array<double, 4> qrotconj{ qrot[0], -qrot[1], -qrot[2], -qrot[3] };
    auto q2 = geometry_msgs::Quaternion();
    q2.w = qrotconj[0]*qprod1[0] - qrotconj[1]*qprod1[1] - qrotconj[2]*qprod1[2] - qrotconj[3]*qprod1[3];
    q2.x = qrotconj[0]*qprod1[1] + qrotconj[1]*qprod1[0] + qrotconj[2]*qprod1[3] - qrotconj[3]*qprod1[2];
    q2.y = qrotconj[0]*qprod1[2] - qrotconj[1]*qprod1[3] + qrotconj[2]*qprod1[0] + qrotconj[3]*qprod1[1];
    q2.z = qrotconj[0]*qprod1[3] + qrotconj[1]*qprod1[2] - qrotconj[2]*qprod1[1] + qrotconj[3]*qprod1[0];

    return q2;
}

/* End non-class functions */


bool WristDriverNode::cb_setWristMode(SetWristModeRequest &request, SetWristModeResponse &response) {
    // ONLY TURNS OFF TORQUE FOR JOINT 1 (this is the only joint that actually changes)
    uint8_t dxl_error{};
    switch (request.mode) {
        case 0: {
            ROS_INFO("Switching to joint control mode.");
            // disable torque
            packetHandler->write1ByteTxRx(
                portHandler,
                BROADCAST_ID,
                ADDR_TORQUE_ENABLE,
                0,
                &dxl_error
            );

            // re-enables torque and sets mode
            setupDynamixel(JT0_ID);
            setupDynamixel(JT1_ID, 4);
            mode_ = 0;
            response.success = true;
            }
            break;
        case 1: {
            ROS_INFO("Switching both joints to velocity control.");
            // disable torque
            packetHandler->write1ByteTxRx(
                portHandler,
                BROADCAST_ID,
                ADDR_TORQUE_ENABLE,
                0,
                &dxl_error
            );

            // re-enables torque and sets mode
            setupDynamixel(JT0_ID, 1);
            setupDynamixel(JT1_ID, 1);
            mode_ = 1;
            response.success = true;
            }
            break;
        default:
            ROS_INFO("Mode not recognized.");
            response.success = false;
            break;
    }
    return true;    
}

// Global variable to keep track of the running state
volatile sig_atomic_t keep_running = 1;

// Signal handler function
void handle_sigquit(int signum) {
    keep_running = 0;
    std::cout << "Ctrl+\ detected! Gracefully exiting..." << std::endl;
    ros::shutdown();
}

int main(int argc, char* argv[]) {

    // Register signal handler
    signal(SIGQUIT, handle_sigquit); // Handle Ctrl+\ for graceful shutdown

    while (keep_running)  // Use keep_running to control the loop
    {
        // std::cout<< "Keep running state: " << keep_running << std::endl;
        // sleep for 1 second
        sleep(1);

        int threadPriority{ 97 };
        char* portName;
        bool portOverriden{false};
        // thread handling
        if (argc >= 2) {
            // expect the first argument to be the port
            portName = argv[1];
            portOverriden = true;
            if (argc >= 3) {
                // expect the first argument to be the priority
                threadPriority = atoi(argv[2]);
            }
        }

        // pid_t pid = getpid();
        // struct sched_param param;
        // memset(&param, 0, sizeof(param));
        // param.sched_priority = threadPriority;
        // if(sched_setscheduler(pid, SCHED_FIFO, &param)) {
        //     throw std::runtime_error("Couldn't set scheduling priority and policy");
        // }
        // std::cout << "Thread priority set to " << threadPriority << std::endl;

        if (portOverriden)
            portHandler = dynamixel::PortHandler::getPortHandler(portName);
        else
            portHandler = dynamixel::PortHandler::getPortHandler(PORT_NAME);
        packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

        uint8_t dxl_error = 0;
        int dxl_comm_result = COMM_TX_FAIL;

        // Open Serial Port
        dxl_comm_result = portHandler->openPort();
        if (dxl_comm_result == false) {
            ROS_ERROR("Failed to open the port! Do I have the right permissions?");
            return -1;
        } 
        // else {
        //     ROS_INFO("Succeeded to open the port.");
        // }

        // Set the baudrate of the serial port (use DYNAMIXEL Baudrate)
        dxl_comm_result = portHandler->setBaudRate(BAUDRATE);
        if (dxl_comm_result == false) {
            ROS_ERROR("Failed to set the baudrate!");
            return -1;
        } 
        // else {
        //     ROS_INFO("Succeeded to set the baudrate.");
        // }

        bool connected_0 = setupDynamixel(JT0_ID);
        bool connected_1 = setupDynamixel(JT1_ID, 4);

        if (!connected_0 || !connected_1) {
            // ROS_ERROR("Failed to connect to dynamixels.");
            continue;
        }

        ROS_INFO("Wrist Driver starting");
        ros::init(argc, argv, "wrist_driver_ros");
        ros::NodeHandle nh;

        std::cout << "Connected to dynamixels - starting wrist driver node." << std::endl;
        WristDriverNode wrist_driver(nh);
        ros::spin();

        std::cout << "Shutting down connection to dynamixels." << std::endl;

        // rclcpp::init(argc, argv);

        // auto readwritenode = std::make_shared<WristDriverNode>();
        // rclcpp::spin(readwritenode);
        // rclcpp::shutdown();

        // Disable Torque of DYNAMIXELs
        packetHandler->write1ByteTxRx(
            portHandler,
            BROADCAST_ID,
            ADDR_TORQUE_ENABLE,
            0,
            &dxl_error
        );

    }

    std::cout << "Shut down wrist driver." << std::endl;
}
