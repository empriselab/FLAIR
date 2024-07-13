cmake_minimum_required(VERSION 3.5)
project(wrist_driver_ros)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  wrist_driver_interfaces
  dynamixel_sdk
)

catkin_package(CATKIN_DEPENDS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  wrist_driver_interfaces
  dynamixel_sdk
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(wrist_driver src/wrist_driver.cpp)

target_link_libraries(wrist_driver
  ${catkin_LIBRARIES}
)

install(TARGETS wrist_driver
  DESTINATION lib/${PROJECT_NAME}
)