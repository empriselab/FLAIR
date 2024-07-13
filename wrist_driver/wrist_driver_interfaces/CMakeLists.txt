cmake_minimum_required(VERSION 3.5)
project(wrist_driver_interfaces)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(catkin REQUIRED COMPONENTS
    std_msgs
    message_generation
)

add_message_files(
  FILES
  SimpleJointAngleCommand.msg
  WristState.msg
)

add_service_files(
    FILES
    SetWristMode.srv
)

generate_messages(
    DEPENDENCIES
    std_msgs
)

catkin_package(CATKIN_DEPENDS
    message_runtime
)
