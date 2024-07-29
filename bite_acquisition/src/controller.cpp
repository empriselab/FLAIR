#include "controller.hpp"
#include <cmath>        // std::abs
#include <typeinfo>

using namespace std;

// Create an event listener that will set the promise action event to the exit value
// Will set promise to either END or ABORT
// Use finish_promise.get_future.get() to wait and get the value
std::function<void(k_api::Base::ActionNotification)> 
    create_event_listener_by_promise(std::promise<k_api::Base::ActionEvent>& finish_promise)
{
    return [&finish_promise] (k_api::Base::ActionNotification notification)
    {
        const auto action_event = notification.action_event();
        switch(action_event)
        {
        case k_api::Base::ActionEvent::ACTION_END:
        case k_api::Base::ActionEvent::ACTION_ABORT:
            finish_promise.set_value(action_event);
            break;
        default:
            break;
        }
    };
}

// Create an event listener that will set the sent reference to the exit value
// Will set to either END or ABORT
// Read the value of returnAction until it is set
std::function<void(k_api::Base::ActionNotification)>
    create_event_listener_by_ref(k_api::Base::ActionEvent& returnAction)
{
    return [&returnAction](k_api::Base::ActionNotification notification)
    {
        const auto action_event = notification.action_event();
        switch(action_event)
        {
        case k_api::Base::ActionEvent::ACTION_END:
        case k_api::Base::ActionEvent::ACTION_ABORT:
            returnAction = action_event;
            break;
        default:
            break;
        }
    };
}

Controller::Controller(ros::NodeHandle nh)
{
    ROS_INFO("Retreiving ROS parameters");
    ros::param::get("~username", m_username);
    ros::param::get("~password", m_password);
    ros::param::get("~ip_address", m_ip_address);
    ros::param::get("~api_session_inactivity_timeout_ms", m_api_session_inactivity_timeout_ms);
    ros::param::get("~api_connection_inactivity_timeout_ms", m_api_connection_inactivity_timeout_ms);

    ROS_INFO("Starting to initialize controller");
    m_tcp_transport = new k_api::TransportClientTcp();
    m_tcp_transport->connect(m_ip_address, TCP_PORT);
    m_tcp_router = new k_api::RouterClient(m_tcp_transport, [](k_api::KError err) { ROS_ERROR("Kortex API error was encountered with the TCP router: %s", err.toString().c_str()); });

    // Set session data connection information
    auto createSessionInfo = Kinova::Api::Session::CreateSessionInfo();
	createSessionInfo.set_username(m_username);
	createSessionInfo.set_password(m_password);
	createSessionInfo.set_session_inactivity_timeout(m_api_session_inactivity_timeout_ms);
    createSessionInfo.set_connection_inactivity_timeout(m_api_connection_inactivity_timeout_ms);

    // Session manager service wrapper
    ROS_INFO("Creating session for communication");
    m_tcp_session_manager = new k_api::SessionManager(m_tcp_router);
    // Create the sessions so we can start using the robot
    try 
    {
        m_tcp_session_manager->CreateSession(createSessionInfo);
        ROS_INFO("Session created successfully for TCP services");
    }
    catch(std::runtime_error& ex_runtime)
    {
        std::string error_string = "The node could not connect to the arm. Did you specify the right IP address and is the arm powered on?";
        ROS_ERROR("%s", error_string.c_str());
        throw ex_runtime;
    }

    // Create services
    mBase = new k_api::Base::BaseClient(m_tcp_router);
    mBaseCyclic = new k_api::BaseCyclic::BaseCyclicClient(m_tcp_router);

    // Do we require these?
    mActuatorConfig = new k_api::ActuatorConfig::ActuatorConfigClient(m_tcp_router);
    mServoingMode = k_api::Base::ServoingModeInformation();
    mControlModeMessage = k_api::ActuatorConfig::ControlModeInformation();

    // Clearing faults
    try
    {
        mBase->ClearFaults();
    }
    catch(...)
    {
        std::cout << "Unable to clear robot faults" << std::endl;
        return;
    }

    mJointStatePub = nh.advertise<sensor_msgs::JointState>("robot_joint_states", 10);
    mCartesianStatePub = nh.advertise<bite_acquisition::CartesianState>("robot_cartesian_state", 10);
    mRobotStateTimer = nh.createTimer(ros::Duration(0.01),  &Controller::publishState, this);

    mSetJointPositionService = nh.advertiseService("set_joint_position", &Controller::setJointPosition, this);
    mSetJointVelocityService = nh.advertiseService("set_joint_velocity", &Controller::setJointVelocity, this);
    mSetPoseService = nh.advertiseService("set_pose", &Controller::setPose, this);
    mSetPoseWaypointsService = nh.advertiseService("set_pose_waypoints", &Controller::setPoseWaypoints, this);
    mSetTwistService = nh.advertiseService("set_twist", &Controller::setTwist, this);

    mTareFTSensorSub = nh.subscribe("tare_ft_sensor", 10, &Controller::tareFTSensorCallback, this);
    mZeroFTSensorValues = std::vector<double>(6, 0.0);
    mFTSensorValues = std::vector<double>(6, 0.0);
    mForceThreshold = std::vector<double>(6, 1000.0);

    mWatchdogActive = true;
    ROS_INFO("Controller initialized");
}

Controller::~Controller()
{
    try
    {
        mBase->Stop();
    }
    catch (k_api::KDetailedException& ex)
    {
        std::cout << "Kortex exception: " << ex.what() << std::endl;

        std::cout << "Error sub-code: " << k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))) << std::endl;
    }
    
    m_tcp_session_manager->CloseSession();
    m_tcp_router->SetActivationStatus(false);
    m_tcp_transport->disconnect();

    delete mBase;
    delete mBaseCyclic;

    delete m_tcp_session_manager;
    delete m_tcp_router;
    delete m_tcp_transport;

    ros::Duration(0.10).sleep();
}

inline double Controller::degreesToRadians(double degrees)
{
    return (M_PI / 180.0) * degrees;
}

inline double Controller::radiansToDegrees(double radians)
{
    return (180.0 / M_PI) * radians;
}

void Controller::tareFTSensorCallback(const std_msgs::Bool& msg)
{
    mTareFTSensor.store(true);
    // ROS_INFO("Trying to stop action... ");
    // auto start_time = ros::Time::now();
    // try
    // {
    //     mBase->StopAction();
    // }
    // catch (k_api::KDetailedException& ex)
    // {
    //     std::cout << "Kortex exception: " << ex.what() << std::endl;

    //     std::cout << "Error sub-code: " << k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))) << std::endl;
    // }
    // auto end_time = ros::Time::now();
    // ROS_INFO_STREAM("Stopping action took " << (end_time - start_time).toSec() << " seconds");
}

void Controller::publishState(const ros::TimerEvent& event)
{
    auto start_time = ros::Time::now();
    mLastFeedback = mBaseCyclic->RefreshFeedback();
    auto mid_time = ros::Time::now();

    // std::cout << "RefreshFeedback took " << (mid_time - start_time).toSec() << " seconds" << std::endl;

    int actuator_count = mBase->GetActuatorCount().count();
    
    auto joint_state = sensor_msgs::JointState();
    joint_state.header.stamp = ros::Time::now();
    for (std::size_t i = 0; i < actuator_count; ++i)
    {
        joint_state.name.push_back("joint_" + std::to_string(i + 1));
        joint_state.position.push_back(degreesToRadians(double(mLastFeedback.actuators(i).position())));
        joint_state.velocity.push_back(degreesToRadians(double(mLastFeedback.actuators(i).velocity())));
        joint_state.effort.push_back(double(mLastFeedback.actuators(i).torque()));
    }

    // Read finger state. Note: position and velocity are percentage values
    // (0-100). Effort is set as current consumed by gripper motor (mA).
    joint_state.name.push_back("finger_joint");
    joint_state.position.push_back(0.8*mLastFeedback.interconnect().gripper_feedback().motor()[0].position() / 100.0);
    joint_state.velocity.push_back(0.8*mLastFeedback.interconnect().gripper_feedback().motor()[0].velocity() / 100.0);
    joint_state.effort.push_back(mLastFeedback.interconnect().gripper_feedback().motor()[0].current_motor());
    // joint_state.position.push_back(0.5);
    // joint_state.velocity.push_back(0.0);
    // joint_state.effort.push_back(0.0);

    mJointStatePub.publish(joint_state);

    auto cartesian_state = bite_acquisition::CartesianState();
    cartesian_state.header.stamp = ros::Time::now();

    cartesian_state.pose.position.x = mLastFeedback.base().tool_pose_x();
    cartesian_state.pose.position.y = mLastFeedback.base().tool_pose_y();
    cartesian_state.pose.position.z = mLastFeedback.base().tool_pose_z();

    tf2::Quaternion quat;
    quat.setRPY(degreesToRadians(mLastFeedback.base().tool_pose_theta_x()), 
                degreesToRadians(mLastFeedback.base().tool_pose_theta_y()), 
                degreesToRadians(mLastFeedback.base().tool_pose_theta_z()));
    cartesian_state.pose.orientation = tf2::toMsg(quat);


    // tf2::Matrix3x3 m(quat);
    // double roll, pitch, yaw;
    // m.getRPY(roll, pitch, yaw);

    // std::cout << "Roll: " << roll << " " <<degreesToRadians(mLastFeedback.base().tool_pose_theta_x()) << std::endl;
    // std::cout << "Pitch: " << pitch << " " <<degreesToRadians(mLastFeedback.base().tool_pose_theta_y()) << std::endl;
    // std::cout << "Yaw: " << yaw << " " <<degreesToRadians(mLastFeedback.base().tool_pose_theta_z()) << std::endl;

    cartesian_state.twist.linear.x = mLastFeedback.base().tool_twist_linear_x();
    cartesian_state.twist.linear.y = mLastFeedback.base().tool_twist_linear_y();
    cartesian_state.twist.linear.z = mLastFeedback.base().tool_twist_linear_z();
    cartesian_state.twist.angular.x = degreesToRadians(mLastFeedback.base().tool_twist_angular_x());
    cartesian_state.twist.angular.y = degreesToRadians(mLastFeedback.base().tool_twist_angular_y());
    cartesian_state.twist.angular.z = degreesToRadians(mLastFeedback.base().tool_twist_angular_z());

    mFTSensorValues[0] = mLastFeedback.base().tool_external_wrench_force_x();
    mFTSensorValues[1] = mLastFeedback.base().tool_external_wrench_force_y();
    mFTSensorValues[2] = mLastFeedback.base().tool_external_wrench_force_z();
    mFTSensorValues[3] = mLastFeedback.base().tool_external_wrench_torque_x();
    mFTSensorValues[4] = mLastFeedback.base().tool_external_wrench_torque_y();
    mFTSensorValues[5] = mLastFeedback.base().tool_external_wrench_torque_z();

    if (mTareFTSensor.load())
    {
        mZeroFTSensorValues = mFTSensorValues;
        mTareFTSensor.store(false);
    }

    cartesian_state.wrench.force.x = mFTSensorValues[0] - mZeroFTSensorValues[0];
    cartesian_state.wrench.force.y = mFTSensorValues[1] - mZeroFTSensorValues[1];
    cartesian_state.wrench.force.z = mFTSensorValues[2] - mZeroFTSensorValues[2];
    cartesian_state.wrench.torque.x = mFTSensorValues[3] - mZeroFTSensorValues[3];
    cartesian_state.wrench.torque.y = mFTSensorValues[4] - mZeroFTSensorValues[4];
    cartesian_state.wrench.torque.z = mFTSensorValues[5] - mZeroFTSensorValues[5];
   
    mCartesianStatePub.publish(cartesian_state);

    if(mUpdateForceThreshold.load())
    {
        mForceThreshold = mNewForceThreshold;
        mUpdateForceThreshold.store(false);
    }

    if( std::abs(mFTSensorValues[0] - mZeroFTSensorValues[0]) > std::abs(mForceThreshold[0])
        or std::abs(mFTSensorValues[1] - mZeroFTSensorValues[1]) > std::abs(mForceThreshold[1])
        or std::abs(mFTSensorValues[2] - mZeroFTSensorValues[2]) > std::abs(mForceThreshold[2])
        or std::abs(mFTSensorValues[3] - mZeroFTSensorValues[3]) > std::abs(mForceThreshold[3])
        or std::abs(mFTSensorValues[4] - mZeroFTSensorValues[4]) > std::abs(mForceThreshold[4])
        or std::abs(mFTSensorValues[5] - mZeroFTSensorValues[5]) > std::abs(mForceThreshold[5]))
    {   
        ROS_INFO("Force threshold exceeded");
        std::cout<<"Measured force: "<<std::endl;
        std::cout<<"Fx: "<<mFTSensorValues[0] - mZeroFTSensorValues[0]<<std::endl;
        std::cout<<"Fy: "<<mFTSensorValues[1] - mZeroFTSensorValues[1]<<std::endl;
        std::cout<<"Fz: "<<mFTSensorValues[2] - mZeroFTSensorValues[2]<<std::endl;
        std::cout<<"Tx: "<<mFTSensorValues[3] - mZeroFTSensorValues[3]<<std::endl;
        std::cout<<"Ty: "<<mFTSensorValues[4] - mZeroFTSensorValues[4]<<std::endl;
        std::cout<<"Tz: "<<mFTSensorValues[5] - mZeroFTSensorValues[5]<<std::endl;
        
        // try
        // {
        //     mBase->StopAction();
        // }
        // catch (k_api::KDetailedException& ex)
        // {
        //     std::cout << "Kortex exception: " << ex.what() << std::endl;

        //     std::cout << "Error sub-code: " << k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))) << std::endl;
        // }

        // mForceThreshold = std::vector<double>(6, 1000.0); // stop if you tried stopping once
    }
    
    auto end_time = ros::Time::now();
    // ROS_INFO_STREAM("Publishing state took " << (end_time - start_time).toSec() << " seconds");
}

bool Controller::setJointVelocity(bite_acquisition::JointCommandRequest &request, bite_acquisition::JointCommandResponse &response)
{
    ROS_INFO("Got set joint velocity command");

    if(request.mode != std::string("VELOCITY"))
    {
        ROS_ERROR("Wrong mode for set joint velocity command");
        response.success = false;
        response.error_msg = "Wrong mode for set joint velocity command";
        return false;
    }

    try
    {
        mBase->StopAction();
    }
    catch (k_api::KDetailedException& ex)
    {
        std::cout << "Kortex exception: " << ex.what() << std::endl;

        std::cout << "Error sub-code: " << k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))) << std::endl;
    }

    k_api::Base::JointSpeeds joint_speeds;
    int actuator_count = mBase->GetActuatorCount().count();
    for (size_t i = 0 ; i < actuator_count; ++i)
    {
        auto joint_speed = joint_speeds.add_joint_speeds();
        joint_speed->set_joint_identifier(i);
        joint_speed->set_value(radiansToDegrees(request.command[i]));
        joint_speed->set_duration(1);
    }
    mBase->SendJointSpeedsCommand(joint_speeds);

    int timeout = request.timeout * 1000;
    ROS_INFO("Will stop robot after %d ms", timeout);

    // Wait for timeout seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
    
    // Stop the robot
    std::cout << "Stopping the robot" << std::endl;
    mBase->Stop();

    response.success = true;

    return true;
}

bool Controller::setJointPosition(bite_acquisition::JointCommandRequest &request, bite_acquisition::JointCommandResponse &response)
{
    ROS_INFO("Got joint position command");

    try
    {
        mBase->StopAction();
    }
    catch (k_api::KDetailedException& ex)
    {
        std::cout << "Kortex exception: " << ex.what() << std::endl;

        std::cout << "Error sub-code: " << k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))) << std::endl;
    }

    auto action = k_api::Base::Action();
    action.set_name("Example angular action movement");
    action.set_application_data("");

    auto reach_joint_angles = action.mutable_reach_joint_angles();
    auto joint_angles = reach_joint_angles->mutable_joint_angles();

    auto actuator_count = mBase->GetActuatorCount();

    std::cout << "Actuator count: " << actuator_count.count() << std::endl;
    // Arm straight up
    for (size_t i = 0; i < actuator_count.count(); ++i) 
    {
        auto joint_angle = joint_angles->add_joint_angles();
        joint_angle->set_joint_identifier(i);
        joint_angle->set_value(radiansToDegrees(request.command[i]));
    }

    // Connect to notification action topic
    // (Promise alternative)
    // See cartesian examples for Reference alternative
    std::promise<k_api::Base::ActionEvent> finish_promise;
    auto finish_future = finish_promise.get_future();
    auto promise_notification_handle = mBase->OnNotificationActionTopic(
        create_event_listener_by_promise(finish_promise),
        k_api::Common::NotificationOptions()
    );

    std::cout << "Executing action" << std::endl;
    mBase->ExecuteAction(action);

    std::cout << "Waiting for movement to finish ..." << std::endl;

    // Wait for future value from promise
    // (Promise alternative)
    // See cartesian examples for Reference alternative
    const auto status = finish_future.wait_for(TIMEOUT_DURATION);
    mBase->Unsubscribe(promise_notification_handle);

    if(status != std::future_status::ready)
    {
        std::cout << "Timeout on action notification wait" << std::endl;
        return false;
    }
    const auto promise_event = finish_future.get();

    std::cout << "Angular movement completed" << std::endl;
    std::cout << "Promise value : " << k_api::Base::ActionEvent_Name(promise_event) << std::endl; 

    return true;
}

bool Controller::setPose(bite_acquisition::PoseCommandRequest &request, bite_acquisition::PoseCommandResponse &response)
{
    ROS_INFO("Received pose command");

    try
    {
        mBase->StopAction();
    }
    catch (k_api::KDetailedException& ex)
    {
        std::cout << "Kortex exception: " << ex.what() << std::endl;

        std::cout << "Error sub-code: " << k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))) << std::endl;
    }

    mNewForceThreshold = std::vector<double>(request.force_threshold.begin(), request.force_threshold.end());
    mUpdateForceThreshold.store(true);

    auto action = k_api::Base::Action();
    action.set_name("Example Cartesian action movement");
    action.set_application_data("");

    auto constrained_pose = action.mutable_reach_pose();
    auto pose = constrained_pose->mutable_target_pose();
    pose->set_x(request.target.position.x);                 // x (meters)
    pose->set_y(request.target.position.y);                 // y (meters)
    pose->set_z(request.target.position.z);                 // z (meters)
    tf2::Quaternion quat;
    tf2::fromMsg(request.target.orientation, quat);
    tf2::Matrix3x3 m(quat);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    pose->set_theta_x(radiansToDegrees(roll));    // theta x (degrees)
    pose->set_theta_y(radiansToDegrees(pitch));    // theta y (degrees)
    pose->set_theta_z(radiansToDegrees(yaw));    // theta z (degrees)

    std::cout << "Setting cartesian pose" << std::endl;
    std::cout << "X: " << pose->x() << std::endl;
    std::cout << "Y: " << pose->y() << std::endl;
    std::cout << "Z: " << pose->z() << std::endl;
    std::cout << "Theta X: " << pose->theta_x() << std::endl;
    std::cout << "Theta Y: " << pose->theta_y() << std::endl;
    std::cout << "Theta Z: " << pose->theta_z() << std::endl;

    // Connect to notification action topic
    // (Reference alternative)
    // See angular examples for Promise alternative
    k_api::Base::ActionEvent event = k_api::Base::ActionEvent::UNSPECIFIED_ACTION_EVENT;
    auto reference_notification_handle = mBase->OnNotificationActionTopic(
        create_event_listener_by_ref(event),
        k_api::Common::NotificationOptions()
    );

    std::cout << "Executing action" << std::endl;
    mBase->ExecuteAction(action);

    std::cout << "Waiting for movement to finish ..." << std::endl;

    // Wait for reference value to be set
    // (Reference alternative)
    // See angular examples for Promise alternative
    // Set a timeout after 20s of wait
    const auto timeout = std::chrono::system_clock::now() + TIMEOUT_DURATION;
    while(event == k_api::Base::ActionEvent::UNSPECIFIED_ACTION_EVENT &&
        std::chrono::system_clock::now() < timeout)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    mBase->Unsubscribe(reference_notification_handle);

    if(event == k_api::Base::ActionEvent::UNSPECIFIED_ACTION_EVENT)
    {
        std::cout << "Timeout on action notification wait" << std::endl;
        response.success = false;
        return true;
    }

    std::cout << "Cartesian movement completed" << std::endl;
    std::cout << "Reference value : " << k_api::Base::ActionEvent_Name(event) << std::endl;

    response.success = true;
    return true;
}

bool Controller::setPoseWaypoints(bite_acquisition::PoseWaypointsCommandRequest &request, bite_acquisition::PoseWaypointsCommandResponse &response)
{
    ROS_INFO("Received pose waypoints command");
    return true;
}

bool Controller::setTwist(bite_acquisition::TwistCommandRequest &request, bite_acquisition::TwistCommandResponse &response)
{
    ROS_INFO("Received twist command");
    return true;
}

int main(int argc, char * argv[]) {
    ROS_INFO_STREAM("Kinova controller starting");
    ros::init(argc, argv, "controller");
    ros::NodeHandle nh;

    Controller controller(nh);
    // ros::spin();
    ros::MultiThreadedSpinner spinner(4); // Use 4 threads
    spinner.spin(); // spin() will not return until the node has been shutdown

}
