import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import math
import os
import time
import os
import pickle

# ros imports
import rospy
import tf2_ros
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float64, Bool

import threading
import flair_utils
import cmath
import yaml
import argparse

ROBOT = 'kinova-deployment' # 'kinova' or 'franka' or 'kinova-deployment' (used for controller on the NUC)
# ROBOT = 'kinova'    

from rs_ros import RealSenseROS
from pixel_selector import PixelSelector

if ROBOT == 'franka':
    from robot_controller.franka_controller import FrankaRobotController
elif ROBOT == 'kinova':
    from robot_controller.kinova_controller import KinovaRobotController
elif ROBOT == 'kinova-deployment':
    from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient
    from feeding_deployment.robot_controller.command_interface import JointCommand, CartesianCommand
else:
    raise ValueError("Invalid robot type")

from wrist_controller import WristController
from visualizer import Visualizer

# PLATE_HEIGHT = 0.16 # 0.192 for scooping, 0.2 for skewering, 0.198 for pushing, twirling
PLATE_HEIGHT = 0.16 # 0.192 for scooping, 0.2 for skewering, 0.198 for pushing, twirling

class SkillLibrary:
    def __init__(self, robot_controller, wrist_controller):
        self.robot_controller = robot_controller
        self.wrist_controller = wrist_controller

        self.pixel_selector = PixelSelector()
        self.tf_utils = flair_utils.TFUtils()
        self.visualizer = Visualizer()

        self.beep_publisher = rospy.Publisher('/beep', String, queue_size=10)

        print("Skill library initialized")

    def reset(self):
        if ROBOT == 'kinova-deployment':
            above_plate_pos = [-2.86495014, -1.61460533, -2.6115943, -1.37673391, 1.11842806, -1.17904586, -2.6957422]
            self.robot_controller.execute_command(JointCommand(above_plate_pos))
        else:
            self.robot_controller.reset()
        self.wrist_controller.reset()

    def move_utensil_to_pose(self, tip_pose, tip_to_wrist = None):

        self.tf_utils.publishTransformationToTF('base_link', 'fork_tip_target', tip_pose)

        if tip_to_wrist is None:
            tip_to_wrist = self.tf_utils.getTransformationFromTF('fork_tip', 'tool_frame')
        tool_frame_target = tip_pose @ tip_to_wrist

        self.visualizer.visualize_fork(tip_pose)
        self.tf_utils.publishTransformationToTF('base_link', 'tool_frame_target', tool_frame_target)
        
        input("Press enter to actually move utensil.")
        if ROBOT == 'franka':
            self.robot_controller.move_to_pose(tool_frame_target)
        elif ROBOT == 'kinova':
            self.robot_controller.move_to_pose(self.tf_utils.get_pose_msg_from_transform(tool_frame_target))
        elif ROBOT == 'kinova-deployment':
            tool_frame_pos = tool_frame_target[:3,3].reshape(1,3).tolist()[0] # one dimensional list
            tool_frame_quat = Rotation.from_matrix(tool_frame_target[:3,:3]).as_quat()
            self.robot_controller.execute_command(CartesianCommand(tool_frame_pos, tool_frame_quat))

    def scooping_skill(self, color_image, depth_image, camera_info, keypoints = None):

        if keypoints is not None:
            start, end = keypoints
        else:
            clicks = self.pixel_selector.run(color_image, num_clicks=2)
            start = clicks[0]
            end = clicks[1]

        fork_rotation = self.tf_utils.getTransformationFromTF('camera_color_optical_frame', 'fork_tip')[:3,:3]

        # action 1: angle the wrist to scoop angle
        self.wrist_controller.set_to_scoop_pos()

        push_angle = flair_utils.angle_between_pixels(np.array(start), np.array(end), color_image.shape[1], color_image.shape[0], orientation_symmetry = False)
        # push_angle = push_angle - 180
        print("Push angle: ", push_angle)
        
        validity, lowest_point = flair_utils.pixel2World(camera_info, start[0], start[1], depth_image)
        if not validity:
            print("Invalid lowest point")
            return
        
        validity, center_point = flair_utils.pixel2World(camera_info, end[0], end[1], depth_image)
        if not validity:
            print("Invalid center point")
            return
        
        fork_rotation_scoop = self.tf_utils.getTransformationFromTF('camera_color_optical_frame', 'fork_tip')[:3,:3]
        print("Fork rotation: ", fork_rotation)

        wrist_rotation = np.linalg.inv(fork_rotation) @ fork_rotation_scoop

        # print roll yaw and pitch of wrist rotation
        print("Roll: ", Rotation.from_matrix(wrist_rotation).as_euler('xyz', degrees=True)[0])
        print("Yaw: ", Rotation.from_matrix(wrist_rotation).as_euler('xyz', degrees=True)[1])
        print("Pitch: ", Rotation.from_matrix(wrist_rotation).as_euler('xyz', degrees=True)[2])

        # estimate roation of fork along x axis

        scooping_start_pose = np.zeros((4,4))
        scooping_start_pose[:3,:3] = Rotation.from_euler('xyz', [0,0,push_angle], degrees=True).as_matrix() @ wrist_rotation
        scooping_start_pose[:3,3] = lowest_point.reshape(1,3)
        scooping_start_pose[3,3] = 1

        scooping_start_pose = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame") @ scooping_start_pose

        scooping_start_pose[2,3] = PLATE_HEIGHT

        scooping_end_pose = np.zeros((4,4))
        scooping_end_pose[:3,:3] = Rotation.from_euler('xyz', [0,0,push_angle], degrees=True).as_matrix() @ wrist_rotation
        scooping_end_pose[:3,3] = center_point.reshape(1,3)
        scooping_end_pose[3,3] = 1

        scooping_end_pose = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame") @ scooping_end_pose

        scooping_end_pose[2,3] = PLATE_HEIGHT

        # visualize 
        self.visualizer.visualize_food(scooping_start_pose, id = 0)
        self.visualizer.visualize_food(scooping_end_pose, id = 1)

        # action 2: move to above start position
        waypoint_1_tip = np.copy(scooping_start_pose)
        waypoint_1_tip[2,3] += 0.07
        self.move_utensil_to_pose(waypoint_1_tip)

        # input("Press enter to continue")

        # action 3: move down until you are at start position
        waypoint_2_tip = np.copy(scooping_start_pose)
        # waypoint_2_tip[2,3] += 0.02
        self.move_utensil_to_pose(waypoint_2_tip)

        # action 4: skim the surface to reach end position
        waypoint_3_tip = np.copy(scooping_end_pose)
        # waypoint_3_tip[2,3] += 0.02
        self.move_utensil_to_pose(waypoint_3_tip)

        # action 4.5: use the wrist to scoop
        tip_to_wrist = self.tf_utils.getTransformationFromTF('fork_tip', 'tool_frame')
        self.wrist_controller.scooping_scoop()

        # action 5: move up
        waypoint_4_tip = np.copy(scooping_end_pose)
        waypoint_4_tip[2,3] += 0.07
        self.move_utensil_to_pose(waypoint_4_tip, tip_to_wrist)

    def dipping_skill(self, color_image, depth_image, camera_info, keypoint = None):

        print("Executing dipping skill.")
        if keypoint is not None:
            (center_x, center_y) = keypoint
        else:
            clicks = self.pixel_selector.run(color_image)
            (center_x, center_y) = clicks[0]

        cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.imshow('vis', color_image)

        # get 3D point from depth image
        validity, point = flair_utils.pixel2World(camera_info, center_x, center_y, depth_image)

        if not validity:
            print("Invalid point")
            return

        # action 1: Rotate scooping DoF to dip angle
        self.wrist_controller.set_to_dip_pos()
        
        fork_rotation = self.tf_utils.getTransformationFromTF('camera_color_optical_frame', 'fork_tip')[:3,:3]

        point_transform = np.zeros((4,4))
        point_transform[:3,:3] = fork_rotation
        point_transform[:3,3] = point.reshape(1,3)
        point_transform[3,3] = 1

        point_base = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame") @ point_transform

        point_base[2,3] = PLATE_HEIGHT + 0.03 #0.045

        self.visualizer.visualize_food(point_base)

        # action 2: Move to above position
        waypoint_1_tip = np.copy(point_base)
        waypoint_1_tip[2,3] += 0.05
        self.move_utensil_to_pose(waypoint_1_tip)

        # action 3: Move down until tip touches the food item
        waypoint_2_tip = np.copy(point_base)
        self.move_utensil_to_pose(waypoint_2_tip)

        # action 4: Move up
        waypoint_3_tip = np.copy(point_base)
        waypoint_3_tip[2,3] += 0.05
        self.move_utensil_to_pose(waypoint_3_tip)       

    def flipping_skill(self, food_item):
        raise NotImplementedError
    
    def cutting_skill(self, color_image, depth_image, camera_info, keypoint = None, cutting_angle = None):

        if keypoint is not None:
            (center_x, center_y) = keypoint

            # shift cutting point perpendicular to the cutting angle in the direction of the fork tines (less y value)
            pt = cmath.rect(23, np.pi/2-cutting_angle)
            center_x = center_x + int(pt.real)
            center_y = center_y - int(pt.imag)
            # cv2.line(color_image_vis, (center_x-x2,center_y+y2), (cut_point[0]+x2,cut_point[1]-y2), (255,0,0), 2)

            cutting_angle = math.degrees(cutting_angle)
            cutting_angle = cutting_angle + 180 # Rajat ToDo - remove this hack bruh
        else:
            clicks = self.pixel_selector.run(color_image, num_clicks=2)
            (left_x, left_y) = clicks[0]
            (right_x, right_y) = clicks[1]
            print("Left: ", left_x, left_y)
            print("Right: ", right_x, right_y)
            if left_y < right_y:
                center_x, center_y = left_x, left_y
                clicks[0], clicks[1] = clicks[1], clicks[0]
            else:
                center_x, center_y = right_x, right_y
            cutting_angle = flair_utils.angle_between_pixels(np.array(clicks[0]), np.array(clicks[1]), color_image.shape[1], color_image.shape[0], orientation_symmetry = False)

        # visualize cutting point and line between left and right points
        cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)
        # cv2.line(color_image, (left_x, left_y), (right_x, right_y), (0, 0, 255), 2)

        cv2.imshow('vis', color_image)
        cv2.waitKey(0)

        # get 3D point from depth image
        validity, point = flair_utils.pixel2World(camera_info, center_x, center_y, depth_image)

        if not validity:
            print("Invalid point")
            return
        
        fork_rotation = self.tf_utils.getTransformationFromTF('camera_color_optical_frame', 'fork_tip')[:3,:3]
        
        # action 1: Set wrist state to cutting angle
        self.wrist_controller.set_to_cut_pos()
        self.wrist_controller.set_to_cut_pos()
        
        fork_rotation_cut = self.tf_utils.getTransformationFromTF('camera_color_optical_frame', 'fork_tip')[:3,:3]
        wrist_rotation = np.linalg.inv(fork_rotation) @ fork_rotation_cut

        print('Cutting angle: ', cutting_angle)
        # update cutting angle to take into account incline of fork tines
        cutting_angle = cutting_angle + 25

        cutting_pose = np.zeros((4,4))
        cutting_pose[:3,:3] = Rotation.from_euler('xyz', [0,0,cutting_angle], degrees=True).as_matrix() @ wrist_rotation
        cutting_pose[:3,3] = point.reshape(1,3)
        cutting_pose[3,3] = 1

        cutting_pose = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame") @ cutting_pose

        cutting_pose[2,3] = max(cutting_pose[2,3], PLATE_HEIGHT)

        self.visualizer.visualize_food(cutting_pose)

        waypoint_1_tip = np.copy(cutting_pose)
        waypoint_1_tip[2,3] += 0.03 

        self.move_utensil_to_pose(waypoint_1_tip)

        # action 2: Move down until tip touches the plate

        waypoint_2_tip = np.copy(cutting_pose)
        waypoint_2_tip[2,3] = PLATE_HEIGHT - 0.009
        self.move_utensil_to_pose(waypoint_2_tip)

        tip_to_wrist = self.tf_utils.getTransformationFromTF('fork_tip', 'tool_frame')

        # action 2.5: slightly turn the fork tines so that the food item flips over / separates
        self.wrist_controller.cutting_tilt()

        # action 3: Push orthogonal to the cutting angle, in direction of towards the robot (+y relative to the fork)
        waypoint_3_tip = np.copy(cutting_pose)
        waypoint_3_tip[2,3] = PLATE_HEIGHT - 0.009
        y_displacement = np.eye(4)
        y_displacement[1,3] = 0.02
        waypoint_3_tip = waypoint_3_tip @ y_displacement
        self.move_utensil_to_pose(waypoint_3_tip, tip_to_wrist)

        ## action 3: Move up
        
        waypoint_4_tip = np.copy(waypoint_3_tip)
        waypoint_4_tip[2,3] += 0.035 
        
        self.move_utensil_to_pose(waypoint_4_tip, tip_to_wrist)

    def skewering_skill(self, color_image, depth_image, camera_info, keypoint=None, major_axis=None, action_index=0):
        if keypoint is not None:
            (center_x, center_y) = keypoint
            major_axis = np.degrees(major_axis)
        else:
            clicks = self.pixel_selector.run(color_image)
            (center_x, center_y) = clicks[0]
            major_axis = 0
        
        print(f"Center x {center_x}, Center y {center_y}, Action index {action_index}")

        # get 3D point from depth image
        validity, point = flair_utils.pixel2World(camera_info, center_x, center_y, depth_image)

        if not validity:
            print("Invalid point")
            return

        print("Getting transformation from base_link to camera_color_optical_frame")
        food_transform = np.eye(4)
        food_transform[:3,3] = point.reshape(1,3)
        food_base = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame") @ food_transform
        print("---- Height of skewer point: ", food_base[2,3])

        print("Food detection height: ", food_base[2,3])
        input("Press enter to continue")
        food_base[2,3] = max(food_base[2,3] - 0.01, PLATE_HEIGHT) 
        print("---- Height of skewer point (after max): ", food_base[2,3]) 

        food_base[:3,:3] = Rotation.from_euler('xyz', [0,0,0], degrees=True).as_matrix()

        skewer_axis = -major_axis - 90
        if action_index%2 == 0: # 0s are along major axis
            print("-------- Rotating by 90 degrees")
            skewer_axis = skewer_axis + 90
        if skewer_axis > 90:
            skewer_axis = skewer_axis - 180

        food_base[:3,:3] = Rotation.from_euler('xyz', [0,0,skewer_axis], degrees=True).as_matrix()

        self.tf_utils.publishTransformationToTF('base_link', 'food_frame', food_base)
        self.visualizer.visualize_food(food_base)

        base_to_tip = self.tf_utils.getTransformationFromTF('base_link', 'fork_tip')
        food_base[:3,:3] = food_base[:3,:3] @ base_to_tip[:3,:3]

        # Action 1: Move to action start position
        waypoint_1_tip = np.copy(food_base)
        waypoint_1_tip[2,3] += 0.05
        self.move_utensil_to_pose(waypoint_1_tip)

        # Action 2: Move inside food item
        waypoint_2_tip = np.copy(food_base)
        self.move_utensil_to_pose(waypoint_2_tip)

        # Rajat ToDo: Switch to scooping pick up
        self.scooping_pickup()
        # self.move_utensil_to_pose(waypoint_1_tip)

    def joint_state_callback(self, joint_name, msg):
        if joint_name in msg.name:
            index = msg.name.index(joint_name)
            joint_position = msg.position[index]
            return joint_position
        return None

    def scooping_pickup(self, hack = True):

        forkpitch_to_tip = self.tf_utils.getTransformationFromTF('forkpitch', 'fork_tip')
        print("Forkpitch to tip: ", forkpitch_to_tip)
        distance = forkpitch_to_tip[0,3]

        print("Distance: ", distance)

        tool_frame = self.tf_utils.getTransformationFromTF('base_link', 'tool_frame')

        tool_frame_displacement = np.eye(4)
        tool_frame_displacement[0,3] = distance/8 # move down
        tool_frame_displacement[1,3] = -distance*3/4 # move back

        tool_frame_target = tool_frame @ tool_frame_displacement

        self.tf_utils.publishTransformationToTF('base_link', 'tool_frame_target', tool_frame_target)
        
        input("Press enter to start scooping pickup")

        scoop_thread = threading.Thread(target=self.wrist_controller.scoop_wrist)
        scoop_thread.start()

        # input("Press enter to also move the robot...")
        if ROBOT == 'franka' or ROBOT == 'kinova':
            raise NotImplementedError
        elif ROBOT == 'kinova-deployment':
            self.robot_controller.execute_command(CartesianCommand(tool_frame_target[:3,3].tolist(), Rotation.from_matrix(tool_frame_target[:3,:3]).as_quat()))

        # wait for scoop thread to finish
        scoop_thread.join()

    def pushing_skill(self, color_image, depth_image, camera_info, keypoints = None):
        
        if keypoints is not None:
            start, end = keypoints
        else:
            clicks = self.pixel_selector.run(color_image, num_clicks=2)
            start = clicks[0]
            end = clicks[1]
        
        validity, end_vec_3d = flair_utils.pixel2World(camera_info, end[0], end[1], depth_image)
        if not validity:
            print("Invalid depth detected")
            return

        push_angle = flair_utils.angle_between_pixels(np.array(start), np.array(end), color_image.shape[1], color_image.shape[0], orientation_symmetry = False)

        validity, start_vec_3d = flair_utils.pixel2World(camera_info, start[0], start[1], depth_image)
        if not validity:
            print("Invalid depth detected")
            return
        
        print("Executing pushing action.")

        grouping_start_pose = np.zeros((4,4))
        grouping_start_pose[:3,:3] = Rotation.from_euler('xyz', [0,0,push_angle], degrees=True).as_matrix()
        grouping_start_pose[:3,3] = start_vec_3d.reshape(1,3)
        grouping_start_pose[3,3] = 1

        grouping_start_pose = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame") @ grouping_start_pose

        #print('here', grouping_start_pose[2], 'z', grouping_start_pose[2,3])
        print("Pushing Depth: ", grouping_start_pose[2,3])
        grouping_start_pose[2,3] = max(PLATE_HEIGHT, grouping_start_pose[2,3])

        grouping_end_pose = np.zeros((4,4))
        grouping_end_pose[:3,:3] = Rotation.from_euler('xyz', [0,0,push_angle], degrees=True).as_matrix()
        grouping_end_pose[:3,3] = end_vec_3d.reshape(1,3)
        grouping_end_pose[3,3] = 1

        grouping_end_pose = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame") @ grouping_end_pose

        grouping_end_pose[2,3] = max(PLATE_HEIGHT, grouping_start_pose[2,3])

        # action 1: Move to above start position
        waypoint_1 = np.copy(grouping_start_pose)
        waypoint_1[2,3] += 0.05
        self.move_utensil_to_pose(waypoint_1)

        # action 2: Move down until tip touches plate
        waypoint_2 = np.copy(grouping_start_pose)
        self.move_utensil_to_pose(waypoint_2)

        # action 3: Move to end position
        waypoint_3 = np.copy(grouping_end_pose)
        self.move_utensil_to_pose(waypoint_3)

        # action 4: Move a bite up
        waypoint_4 = self.tf_utils.getTransformationFromTF('base_link', 'fork_tip')
        waypoint_4[2,3] += 0.05
        self.move_utensil_to_pose(waypoint_4) 

        return 

    def twirling_skill(self, color_image, depth_image, camera_info, keypoint = None, twirl_angle = None):
        
        if keypoint is not None:
            (center_x, center_y) = keypoint
        else:
            clicks = self.pixel_selector.run(color_image)
            (center_x, center_y) = clicks[0]
            twirl_angle = 90

        validity, center_point = flair_utils.pixel2World(camera_info, center_x, center_y, depth_image)
        if not validity:
            print("Invalid center pixel")
            return
        twirl_angle = 90 + twirl_angle

        twirl_camera_frame = np.zeros((4,4))
        twirl_camera_frame[:3,:3] = Rotation.from_euler('xyz', [0,0,twirl_angle], degrees=True).as_matrix()
        twirl_camera_frame[:3,3] = center_point.reshape(1,3)
        twirl_camera_frame[3,3] = 1

        base_to_camera = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame")
        twirl_base_frame = base_to_camera @ twirl_camera_frame

        twirl_base_frame[2,3] = PLATE_HEIGHT

        self.visualizer.visualize_fork(twirl_base_frame)
        self.tf_utils.publishTransformationToTF("base_link", "target_fork_tip", twirl_base_frame)

        # action 1: Move to above position
        waypoint_1_tip = np.copy(twirl_base_frame)
        waypoint_1_tip[2,3] += 0.05

        self.move_utensil_to_pose(waypoint_1_tip)

        ## action 2: Move down until tip touches plate
        waypoint_2_tip = np.copy(twirl_base_frame)
        self.move_utensil_to_pose(waypoint_2_tip)

        ## action 3: Twirl
        self.wrist_controller.twirl_wrist()

        ## action 4: Scooping pick up
        self.scooping_pickup(hack=False)

        return None

    def move_to_mouth(self, OFFSET = 0.1):

        # # ask to open mouth
        # beep_msg = String()
        # beep_msg.data = "O pen your mouth."
        # self.beep_publisher.publish(beep_msg)

        input('Detect mouth center?')

        # check if mouth is open
        while True:
            mouth_open = rospy.wait_for_message('/mouth_open', Bool)
            if mouth_open.data:
                break
            else:
                print('Mouth is closed.')
    
        mouth_center_3d_msg = rospy.wait_for_message('/mouth_center', Point)
        mouth_center_3d = np.array([mouth_center_3d_msg.x, mouth_center_3d_msg.y, mouth_center_3d_msg.z])

        input('Execute transfer?')

        base_to_fork_tip = self.tf_utils.getTransformationFromTF('base_link', 'tool_frame')

        # create frame at mouth center
        mouth_center_transform = np.eye(4)
        mouth_center_transform[:3,3] = mouth_center_3d

        mouth_center_transform = self.tf_utils.getTransformationFromTF('base_link', 'camera_color_optical_frame') @ mouth_center_transform

        mouth_offset = np.eye(4)
        mouth_offset[2,3] = -OFFSET
        
        transfer_target = mouth_center_transform @ mouth_offset

        fork_tip_transform = self.tf_utils.getTransformationFromTF('base_link', 'fork_tip')

        transfer_target[:3,:3] = fork_tip_transform[:3,:3]

        # visualize on rviz
        self.tf_utils.publishTransformationToTF('base_link', 'mouth_center_transform', mouth_center_transform)
        self.tf_utils.publishTransformationToTF('base_link', 'transfer_target', transfer_target)

        self.move_utensil_to_pose(transfer_target)
    
if __name__ == "__main__":
    rospy.init_node('SkillLibrary')

    if ROBOT == 'franka':
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="/home/limbrepos/feeding_ws/src/franka_feeding/configs/feeding.yaml")
        args = parser.parse_args()

        config_path = args.config
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        robot_controller = FrankaRobotController(config)
    elif ROBOT == 'kinova':
        robot_controller = KinovaRobotController()
    elif ROBOT == 'kinova-deployment':
        robot_controller = ArmInterfaceClient()

        # Rajat Just for testing
        above_plate_pos = [-2.86495014, -1.61460533, -2.6115943, -1.37673391, 1.11842806, -1.17904586, -2.6957422]
        robot_controller.execute_command(JointCommand(above_plate_pos))
    
    wrist_controller = WristController()
    wrist_controller.set_velocity_mode()
    skill_library = SkillLibrary(robot_controller, wrist_controller)
    skill_library.reset()
    
    camera = RealSenseROS()
    camera_header, camera_color_data, camera_info_data, camera_depth_data = camera.get_camera_data()

    # skill_library.scooping_pickup()

    skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data)

    # skill_library.scooping_skill(camera_color_data, camera_depth_data, camera_info_data)

    # skill_library.dipping_skill(camera_color_data, camera_depth_data, camera_info_data)

    # skill_library.pushing_skill(camera_color_data, camera_depth_data, camera_info_data)

    # skill_library.twirling_skill(camera_color_data, camera_depth_data, camera_info_data)

    # skill_library.cutting_skill(camera_color_data, camera_depth_data, camera_info_data)

    skill_library.reset()