# FLAIR: Feeding via Long-horizon AcquIsition of Realistic dishes

![FLAIR deployed on Kinova 6DoF, 7DoF and Franka Emika Panda 7DoF](assets/6.gif)

## Hardware Requirements

The following robot plaforms are supported:
- Kinova 6-DoF or 7-DoF robot arm
- Franka Emika Panda 7-DoF robot arm

FLAIR's uses a custom feeding utensil. You can access a list of components and detailed specifications [here](https://drive.google.com/drive/u/1/folders/1WjtiHdZtLfJFWJ-NYM1NFTaGlcZTuRHH).

## Setting up FLAIR

### Dependencies

Ensure you have the following dependencies installed:
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything)

Further, download weights for the spaghetti segmentation model from [this link](https://drive.google.com/file/d/1MCgjYcFv6nTxO-e3mlMtdZIzmOgu3rNb/view?usp=sharing) and place it in `FLAIR/bite_acquisition`.

For Kinova arms, use the kortex API included in this repository. For Franka arms, use Polymetis combined with Robot-Lighting. Setup instructions for Franka can be found [here](https://github.com/jhejna/robot-lightning).

### ROS Workspace Setup

Create and setup the feeding workspace:
```
mkdir -p flair_ws/src
cd flair_ws/src
git clone https://github.com/empriselab/FLAIR.git
```

Build the workspace after ensuring all dependencies are met:
```
catkin build
cd ..
source devel/setup.sh
```

### Feeding Utensil Setup

Follow these steps to set up and control the feeding utensil using the Dynamixel SDK:

1. Install Dynamixel SDK: [Dynamixel SDK Installation Guide](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/download/#repository)

2. Install Dynamixel Wizard 2.0: [Dynamixel Wizard 2.0 Installation Guide](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/)

3. Follow the Quick Start Guide: [Dynamixel Quick Start Guide for Basic Setup and Assembly](https://emanual.robotis.com/docs/en/dxl/dxl-quick-start-guide/)

4. Configure Dynamixel
   - Set the IDs to 50 and 100 and scan Dynamixel. Refer to Section 6 in the Dynamixel Wizard 2.0 link.
   - Manually change the baud rate to 1Mbps if scanning shows a different value.

5. Start the feeding utensil: Source the workspace, plug in the wrist joint, add port permissions, and run the driver:
   ```
   source devel/setup.bash
   sudo chmod a+rw /dev/ttyUSB0
   rosrun wrist_driver_ros wrist_driver
   ```

6. Test the feeding utensil:
   ```
   rostopic pub -1 /cmd_wrist_joint_angles wrist_driver_interfaces/SimpleJointAngleCommand '{q0: 0.0, q1: 0.0}'
   ```

## Running FLAIR

TO run FLAIR, follow these steps, running each command in a separate terminal window. Remember to source your environment in each terminal to ensure all dependencies are properly loaded.

1. Start the ROS core:
   ```
   roscore
   ```

2. Open Rviz for visualizing robot actions:
   ```
   rviz
   ```

3. Start the wrist driver:
   ```
   rosrun wrist_driver_ros wrist_driver
   ```

4. Launch the bite acquisition process for Kinova or Franka setups:
   ```
   roslaunch bite_acquisition kinova/franka.launch
   ```

5. In a new terminal, set the OpenAI API key and run the main feeding bot script (these should be run in the same terminal):
   ```
   export OPENAI_API_KEY=<your_api_key_here>
   python feeding_bot.py
   ```
