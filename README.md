# FLAIR: Feeding via Long-horizon AcquIsition of Realistic dishes

![FLAIR deployed on Kinova 6DoF, 7DoF and Franka Emika Panda 7DoF](assets/6.gif)

## Hardware Requirements

The following robot plaforms are supported:
- Kinova 6-DoF or 7-DoF robot arm
- Franka Emika Panda 7-DoF robot arm

FLAIR uses a custom feeding utensil. You can access a list of components and detailed specifications [here](https://drive.google.com/drive/u/1/folders/1WjtiHdZtLfJFWJ-NYM1NFTaGlcZTuRHH). 

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

To run FLAIR, follow these steps, running each command in a separate terminal window. Remember to source your environment in each terminal to ensure all dependencies are properly loaded. If you're setting up FLAIR for a new real robot setup, you may need to reconfigure certain variables and run some intermediate tests. For more information, refer to the [Configuring FLAIR for your robot setup](#configuring-flair-for-your-robot-setup) section.

1. Start the ROS core:
   ```
   roscore
   ```

2. Start the wrist driver:
   ```
   rosrun wrist_driver_ros wrist_driver
   ```

3. Launch the bite acquisition process for Kinova or Franka setups with appropriate launch file params:
   ```
   roslaunch bite_acquisition kinova/franka.launch
   ```

4. In a new terminal, set the OpenAI API key and run the main feeding bot script (these should be run in the same terminal):
   ```
   export OPENAI_API_KEY=<your_api_key_here>
   python feeding_bot.py
   ```


## Configuring FLAIR for your robot setup

### Updating Above Plate and Before Transfer Configs

To configure FLAIR for your real robot setup, you'll need to configure the following:

1. Above Plate Config ([Kinova 6DoF](https://github.com/empriselab/FLAIR/blob/0d52bce1081e7ca358b71edd0e9655e905bbc7d0/bite_acquisition/scripts/robot_controller/kinova_controller.py#L18), [Kinova 7DoF](https://github.com/empriselab/FLAIR/blob/0d52bce1081e7ca358b71edd0e9655e905bbc7d0/bite_acquisition/scripts/robot_controller/kinova_controller.py#L21), [Franka](https://github.com/empriselab/FLAIR/blob/0d52bce1081e7ca358b71edd0e9655e905bbc7d0/bite_acquisition/scripts/robot_controller/franka_controller.py#L41)) - This is a joint space (Kinova) or task space (Franka) configuration that the robot goes to before initiating bite acquisition. Ensure that the entire plate is in view of the in-hand camera at this configuration.

2. Before Transfer Config ([Kinova 6DoF](https://github.com/empriselab/FLAIR/blob/0d52bce1081e7ca358b71edd0e9655e905bbc7d0/bite_acquisition/scripts/robot_controller/kinova_controller.py#L19), [Kinova 7DoF](https://github.com/empriselab/FLAIR/blob/0d52bce1081e7ca358b71edd0e9655e905bbc7d0/bite_acquisition/scripts/robot_controller/kinova_controller.py#L22), [Franka](https://github.com/empriselab/FLAIR/blob/0d52bce1081e7ca358b71edd0e9655e905bbc7d0/bite_acquisition/scripts/robot_controller/franka_controller.py#L45)) - This is a joint space (Kinova) or task space (Franka) configuration that the robot goes to before initiating bite transfer. Ensure that the user's face is in view of the in-hand camera at this configuration.  

**Testing:** You can use the [test_robot_configs.py](https://github.com/empriselab/FLAIR/blob/main/bite_acquisition/scripts/test/test_robot_configs.py) script to test the Above Plate and Before Transfer robot configs, and the motion between them.

### Calibrating the Camera to Fork Tip Transform

FLAIR uses the camera to fork tip transform to move the fork tip to a key point identified in the camera image. While we provide a default value for this transform that roughly matches the camera mount + feeding utensil setup provided in the hardware files, each Intel RealSense camera has different intrinsics and each feeding utensil might have slight build differences that require recalibration.

You can change the transform between the end effector link and camera link ([Kinova](https://github.com/empriselab/FLAIR/blob/0d52bce1081e7ca358b71edd0e9655e905bbc7d0/bite_acquisition/launch/kinova.launch#L42), [Franka](https://github.com/empriselab/FLAIR/blob/0d52bce1081e7ca358b71edd0e9655e905bbc7d0/bite_acquisition/launch/franka.launch#L27)) and the transform between the fork roll link and fork tip ([Kinova](https://github.com/empriselab/FLAIR/blob/0d52bce1081e7ca358b71edd0e9655e905bbc7d0/kortex_description/tools/feeding_utensil/feeding_utensil.xacro#L175), [Franka](https://github.com/empriselab/FLAIR/blob/0d52bce1081e7ca358b71edd0e9655e905bbc7d0/franka_description/panda_arm_hand.urdf#L394)).

**Testing:** You can use [test_calibration.py](https://github.com/empriselab/FLAIR/blob/main/bite_acquisition/scripts/test/test_calibration.py) to verify that the camera to fork tip transform is correctly calibrated. In the pop-up window, ensure that the yellow dot is at the center of the fork tip at various feeding utensil pitch values.

### Testing All Robot-Assisted Feeding Skills in Isolation

Before running FLAIR, we recommend testing the correct functioning of all robot-assisted feeding skills that FLAIR uses, i.e., skewering, scooping, twirling, grouping (pushing), dipping, cutting for bite acquisition, and outside-mouth bite transfer.

For all bite acquisition skills, you may set a common plate height parameter ([Kinova](https://github.com/empriselab/FLAIR/blob/0d52bce1081e7ca358b71edd0e9655e905bbc7d0/bite_acquisition/launch/kinova.launch#L13), [Franka](https://github.com/empriselab/FLAIR/blob/0d52bce1081e7ca358b71edd0e9655e905bbc7d0/bite_acquisition/launch/franka.launch#L9)) representing the height of the plate with respect to the robot base. While FLAIR uses the RealSense's depth stream to estimate food item depth, the plate height parameter is used to ensure safety by verifying that the estimated food item depth is between [plate height, plate height + max_food_height], where max_food_height is also a launch file parameter.

**Testing wrist configs:** You can use [test_wrist_configs.py](https://github.com/empriselab/FLAIR/blob/main/bite_acquisition/scripts/test/test_wrist_configs.py) to test that all the wrist configurations required by FLAIR are functional.

**Testing bite acquisition skills with manual key point selection:** You can use [test_acq_skills_pixel_selection.py](https://github.com/empriselab/FLAIR/blob/main/bite_acquisition/scripts/test/test_acq_skills_pixel_selection.py) to test the correct functioning of all bite acquisition skills with manual keypoint selection.

**Testing skewering:** You can use [test_skewering.py](https://github.com/empriselab/FLAIR/blob/main/bite_acquisition/scripts/test/test_skewering.py) to test that skewering with autonomous keypoint selection is functional. You can set the food item type [here](https://github.com/empriselab/FLAIR/blob/0d52bce1081e7ca358b71edd0e9655e905bbc7d0/bite_acquisition/scripts/test/test_skewering.py#L22). We will add tests for other bite acquisition skills with autonomous keypoint selection soon.

**Testing outside-mouth bite transfer:** You can use [test_transfer.py](https://github.com/empriselab/FLAIR/blob/main/bite_acquisition/scripts/test/test_transfer.py) to test outside-mouth bite transfer.

## Citing

Please consider citing our paper if you use code from this repository:

```bibtex
@INPROCEEDINGS{Jenamani-RSS-24, 
    AUTHOR    = {Rajat Kumar Jenamani AND Priya Sundaresan AND Maram Sakr AND Tapomayukh Bhattacharjee AND Dorsa Sadigh}, 
    TITLE     = {{FLAIR: Feeding via Long-Horizon AcquIsition of Realistic dishes}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2024}, 
    ADDRESS   = {Delft, Netherlands}, 
    MONTH     = {July}, 
    DOI       = {10.15607/RSS.2024.XX.031} 
}
```

## Acknowledgements

The wrist controller scripts were adapted from [Lorenzo Shaikewitz](https://lorenzos.io/)'s [work](https://arxiv.org/abs/2211.12705). Special thanks to [Mahanthesh R](https://www.linkedin.com/in/mahanthesh-r/) and [Jake Miller](https://www.linkedin.com/in/jakmilller/) from Cleveland State University for testing this repository on their own robot setup and providing valuable feedback.
