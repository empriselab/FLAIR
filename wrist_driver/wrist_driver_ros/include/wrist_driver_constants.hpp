/**
 * @file constants.hpp
 * @author Lorenzo
 * @brief Contains key constants for wrist driver and dynamixels
 * @version 0.1
 * @date 2022-06-30
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef WRIST_DRIVER_CONSTANTS_H
#define WRIST_DRIVER_CONSTANTS_H

// Dynamixel settings
constexpr int BAUDRATE{1000000};
constexpr char PORT_NAME[]{"/dev/ttyUSB0"};
// Other settings
constexpr int ROBOT_STATE_PUBLISH_FREQUENCY{ 120 }; // Hz
constexpr double REAL_TIME_TOL{ 0.05 }; // %

// Dynamixel register addresses (https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/#control-table-description)
constexpr uint8_t ADDR_OPERATING_MODE{11};
constexpr uint8_t ADDR_TORQUE_ENABLE{64};
constexpr uint8_t ADDR_GOAL_POSITION{116};
constexpr uint8_t ADDR_GOAL_VELOCITY{104};
constexpr uint8_t ADDR_CURR_POSITION{132};
constexpr uint8_t ADDR_CURR_VELOCITY{128};
constexpr uint8_t ADDR_CURR_CURRENT{126};

constexpr int NUM_BYTES{4};
constexpr int NUM_BYTES_CURRENT{2};
constexpr int MODE_POSITION{3};
constexpr int MODE_EXTENDED_POSITION{4};
constexpr double PROTOCOL_VERSION{2.0};
// Dynamixel IDs (configure with wizard)
constexpr uint8_t JT0_ID{ 50 };
constexpr uint8_t JT1_ID{ 100 };

// conversion constants
constexpr double PULSE_TO_RAD{ 3.14159265 / 2048. };
constexpr double RAWVEL_TO_RAD{ 0.229 * (6.2831853) / 60 };

// position constraints
constexpr int JT0_MIN_POSITION{ 0 };
constexpr int JT0_CENTER_POSITION{ 1024 };  // range is +/- 1024
constexpr int JT0_MAX_POSITION{ 2048 };
constexpr int JT1_MIN_POSITION{ -32768 };
constexpr int JT1_CENTER_POSITION{ 0 };  // range arbitrarily set at 8 rotations (4096 pulse per rotation)
constexpr int JT1_MAX_POSITION{ 32768 };

// wrist length constants (in meters)
constexpr double DIST_TOP_JT0{0.1429}; // measured in CAD
constexpr double DIST_JT0_JT1{0.042}; // measured in CAD
constexpr double DIST_JT1_EFF{0.11025}; // estimated (with CAD measurements)
constexpr double DIST_OOP_FORK{0.009};



#endif
