cmake_minimum_required(VERSION 3.0.2)
project(bite_acquisition)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
add_definitions(-D_OS_UNIX)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

add_message_files(
  FILES
  CartesianState.msg
)

add_service_files(
    FILES
    JointCommand.srv
    PoseCommand.srv
    PoseWaypointsCommand.srv
    TwistCommand.srv
    Segmentation.srv
    VQA.srv
)

generate_messages(
    DEPENDENCIES
    geometry_msgs
    sensor_msgs
)

find_package(Boost REQUIRED COMPONENTS system)
link_directories(${PROJECT_SOURCE_DIR}/lib/release)

catkin_package(
  INCLUDE_DIRS
    include
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    geometry_msgs
    sensor_msgs
    message_runtime
)

include_directories(
  include
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${libkindrv_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include/client
  ${PROJECT_SOURCE_DIR}/include/client_stubs
  ${PROJECT_SOURCE_DIR}/include/messages
  ${PROJECT_SOURCE_DIR}/include/common
)

add_executable(controller src/controller.cpp)

set(KINOVA_ARM_LIBS ${catkin_LIBRARIES})
set(KINOVA_ARM_LIBS
    ${KINOVA_ARM_LIBS}
    -l:libKortexApiCpp_without_protobuf.a
    -l:libKortexApiCpp.a)

## Specify libraries to link a library or executable target against
target_link_libraries(controller
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${KINOVA_ARM_LIBS}
)