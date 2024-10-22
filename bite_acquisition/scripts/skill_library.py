import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import math
import threading
import utils
import cmath
import yaml

# ros imports
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from rs_ros import RealSenseROS
from pixel_selector import PixelSelector
from wrist_controller import WristController
from visualizer import Visualizer

class SkillLibrary:
    def __init__(self):

        self.robot_type = rospy.get_param('/robot_type')
        self.plate_height = rospy.get_param('/plate_height')
        self.max_food_height = rospy.get_param('/max_food_height')

        self.pixel_selector = PixelSelector()
        self.tf_utils = utils.TFUtils()
        self.visualizer = Visualizer()
        if self.robot_type == 'franka':
            from robot_controller.franka_controller import FrankaRobotController
            config_path = rospy.get_param('/robot_config')
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            self.robot_controller = FrankaRobotController(config)
        elif self.robot_type == 'kinova_6dof' or self.robot_type == 'kinova_7dof':
            from robot_controller.kinova_controller import KinovaRobotController
            self.robot_controller = KinovaRobotController()
        self.wrist_controller = WristController()

        print("Skill library initialized")

    def reset(self):
        self.pixel_selector.cleanup()
        self.visualizer.clear_visualizations()
        self.robot_controller.reset()
        self.wrist_controller.reset()

    def move_utensil_to_pose(self, tip_pose, tip_to_wrist = None):

        self.tf_utils.publishTransformationToTF('base_link', 'fork_tip_target', tip_pose)

        if tip_to_wrist is None:
            tip_to_wrist = self.tf_utils.getTransformationFromTF('fork_tip', 'tool_frame')
        tool_frame_target = tip_pose @ tip_to_wrist

        self.visualizer.visualize_fork(tip_pose)
        self.tf_utils.publishTransformationToTF('base_link', 'tool_frame_target', tool_frame_target)
    
        self.robot_controller.move_to_pose(tool_frame_target)

    def scooping_skill(self, color_image, depth_image, camera_info, keypoints = None):

        if keypoints is not None:
            start, end = keypoints
        else:
            print("Click on start and end points for scooping in the pixel selector window.")
            clicks = self.pixel_selector.run(color_image, num_clicks=2)
            start = clicks[0]
            end = clicks[1]

        fork_rotation = self.tf_utils.getTransformationFromTF('camera_color_optical_frame', 'fork_tip')[:3,:3]

        # action 1: angle the wrist to scoop angle
        self.wrist_controller.set_to_scoop_pos()

        push_angle = utils.angle_between_pixels(np.array(start), np.array(end), color_image.shape[1], color_image.shape[0], orientation_symmetry = False)
        # push_angle = push_angle - 180
        
        validity, lowest_point = utils.pixel2World(camera_info, start[0], start[1], depth_image)
        if not validity:
            print("ERROR: Scooping start point has invalid depth")
            return
        
        validity, center_point = utils.pixel2World(camera_info, end[0], end[1], depth_image)
        if not validity:
            print("ERROR: Scooping end point has invalid depth")
            return
        
        fork_rotation_scoop = self.tf_utils.getTransformationFromTF('camera_color_optical_frame', 'fork_tip')[:3,:3]

        wrist_rotation = np.linalg.inv(fork_rotation) @ fork_rotation_scoop

        scooping_start_pose = np.zeros((4,4))
        scooping_start_pose[:3,:3] = Rotation.from_euler('xyz', [0,0,push_angle], degrees=True).as_matrix() @ wrist_rotation
        scooping_start_pose[:3,3] = lowest_point.reshape(1,3)
        scooping_start_pose[3,3] = 1
        scooping_start_pose = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame") @ scooping_start_pose
        scooping_start_pose[2,3] = self.plate_height

        scooping_end_pose = np.zeros((4,4))
        scooping_end_pose[:3,:3] = Rotation.from_euler('xyz', [0,0,push_angle], degrees=True).as_matrix() @ wrist_rotation
        scooping_end_pose[:3,3] = center_point.reshape(1,3)
        scooping_end_pose[3,3] = 1
        scooping_end_pose = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame") @ scooping_end_pose
        scooping_end_pose[2,3] = self.plate_height

        print("Scooping points height: ", scooping_start_pose[2,3], scooping_end_pose[2,3])

        self.visualizer.visualize_food(scooping_start_pose, id = 0)
        self.visualizer.visualize_food(scooping_end_pose, id = 1)

        input("Check visualized fork tip poses (red cubes) on rviz. If correct, press ENTER to execute action. Otherwise, press CTRL+C to exit.")

        # action 2: move to above start position
        waypoint_1_tip = np.copy(scooping_start_pose)
        waypoint_1_tip[2,3] += 0.07
        self.move_utensil_to_pose(waypoint_1_tip)

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

        # move to home position
        self.robot_controller.reset()

    def dipping_skill(self, color_image, depth_image, camera_info, keypoint = None):

        print("Executing dipping skill.")
        if keypoint is not None:
            (center_x, center_y) = keypoint
        else:
            print("Click on the point to dip in the pixel selector window.")
            clicks = self.pixel_selector.run(color_image)
            (center_x, center_y) = clicks[0]

        # get 3D point from depth image
        validity, point = utils.pixel2World(camera_info, center_x, center_y, depth_image)

        if not validity:
            print("ERROR: Point of dipping has invalid depth")
            return

        # action 1: Rotate scooping DoF to dip angle
        self.wrist_controller.set_to_dip_pos()
        
        fork_rotation = self.tf_utils.getTransformationFromTF('camera_color_optical_frame', 'fork_tip')[:3,:3]

        point_transform = np.zeros((4,4))
        point_transform[:3,:3] = fork_rotation
        point_transform[:3,3] = point.reshape(1,3)
        point_transform[3,3] = 1

        point_base = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame") @ point_transform

        point_base[2,3] = self.plate_height + 0.03 #0.045

        self.visualizer.visualize_food(point_base)

        input("Check visualized fork tip poses (red cubes) on rviz. If correct, press ENTER to execute action. Otherwise, press CTRL+C to exit.")

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

        # move to home position
        self.robot_controller.reset()

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
            cutting_angle = cutting_angle + 180
        else:
            print("Click on the left and right points for cutting in the pixel selector window.")
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
            cutting_angle = utils.angle_between_pixels(np.array(clicks[0]), np.array(clicks[1]), color_image.shape[1], color_image.shape[0], orientation_symmetry = False)

        # # visualize cutting point and line between left and right points
        # cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)
        # # cv2.line(color_image, (left_x, left_y), (right_x, right_y), (0, 0, 255), 2)

        # cv2.imshow('vis', color_image)
        # cv2.waitKey(0)

        # get 3D point from depth image
        validity, point = utils.pixel2World(camera_info, center_x, center_y, depth_image)

        if not validity:
            print("ERROR: Point of cutting has invalid depth")
            return
        
        fork_rotation = self.tf_utils.getTransformationFromTF('camera_color_optical_frame', 'fork_tip')[:3,:3]
        
        # action 1: Set wrist state to cutting angle
        self.wrist_controller.set_to_cut_pos()
        self.wrist_controller.set_to_cut_pos()
        
        fork_rotation_cut = self.tf_utils.getTransformationFromTF('camera_color_optical_frame', 'fork_tip')[:3,:3]
        wrist_rotation = np.linalg.inv(fork_rotation) @ fork_rotation_cut

        # update cutting angle to take into account incline of fork tines
        cutting_angle = cutting_angle + 25

        cutting_pose = np.zeros((4,4))
        cutting_pose[:3,:3] = Rotation.from_euler('xyz', [0,0,cutting_angle], degrees=True).as_matrix() @ wrist_rotation
        cutting_pose[:3,3] = point.reshape(1,3)
        cutting_pose[3,3] = 1

        cutting_pose = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame") @ cutting_pose

        cutting_pose[2,3] = max(cutting_pose[2,3], self.plate_height)

        self.visualizer.visualize_food(cutting_pose)

        waypoint_1_tip = np.copy(cutting_pose)
        waypoint_1_tip[2,3] += 0.03 

        input("Check visualized fork tip poses (red cubes) on rviz. If correct, press ENTER to execute action. Otherwise, press CTRL+C to exit.")

        self.move_utensil_to_pose(waypoint_1_tip)

        # action 2: Move down until tip touches the plate

        waypoint_2_tip = np.copy(cutting_pose)
        waypoint_2_tip[2,3] = self.plate_height - 0.009
        self.move_utensil_to_pose(waypoint_2_tip)

        tip_to_wrist = self.tf_utils.getTransformationFromTF('fork_tip', 'tool_frame')

        # action 2.5: slightly turn the fork tines so that the food item flips over / separates
        self.wrist_controller.cutting_tilt()

        # action 3: Push orthogonal to the cutting angle, in direction of towards the robot (+y relative to the fork)
        waypoint_3_tip = np.copy(cutting_pose)
        waypoint_3_tip[2,3] = self.plate_height - 0.009
        y_displacement = np.eye(4)
        y_displacement[1,3] = 0.02
        waypoint_3_tip = waypoint_3_tip @ y_displacement
        self.move_utensil_to_pose(waypoint_3_tip, tip_to_wrist)

        ## action 3: Move up
        
        waypoint_4_tip = np.copy(waypoint_3_tip)
        waypoint_4_tip[2,3] += 0.035 
        
        self.move_utensil_to_pose(waypoint_4_tip, tip_to_wrist)

    def skewering_skill(self, color_image, depth_image, camera_info, keypoint=None, major_axis=None):
        if keypoint is not None:
            (center_x, center_y) = keypoint
            major_axis = np.degrees(major_axis)
        else:
            print("Click on the point to skewer in the pixel selector window.")
            clicks = self.pixel_selector.run(color_image)
            (center_x, center_y) = clicks[0]
            major_axis = -90
        
        # get 3D point from depth image
        validity, point = utils.pixel2World(camera_info, center_x, center_y, depth_image)

        if not validity:
            print("ERROR: Point of skewer has invalid depth")
            return

        food_transform = np.eye(4)
        food_transform[:3,3] = point.reshape(1,3)
        food_base = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame") @ food_transform
        if food_base[2,3] > self.plate_height + self.max_food_height:
            print("ERROR: Food detection height is outside max food height (probably because of faulty depth detection). Please try again.")
            return

        print("Skewering point height: ", food_base[2,3])
        food_base[2,3] = max(food_base[2,3] - 0.01, self.plate_height) 

        food_base[:3,:3] = Rotation.from_euler('xyz', [0,0,0], degrees=True).as_matrix()

        skewer_axis = -major_axis - 90
        food_base[:3,:3] = Rotation.from_euler('xyz', [0,0,skewer_axis], degrees=True).as_matrix()

        self.visualizer.visualize_food(food_base)

        base_to_tip = self.tf_utils.getTransformationFromTF('base_link', 'fork_tip')
        food_base[:3,:3] = food_base[:3,:3] @ base_to_tip[:3,:3]

        input("Check visualized fork tip poses (red cubes) on rviz. If correct, press ENTER to execute action. Otherwise, press CTRL+C to exit.")

        # Action 1: Move to action start position
        waypoint_1_tip = np.copy(food_base)
        waypoint_1_tip[2,3] += 0.05
        self.move_utensil_to_pose(waypoint_1_tip)

        # Action 2: Move inside food item
        waypoint_2_tip = np.copy(food_base)
        self.move_utensil_to_pose(waypoint_2_tip)

        self.scooping_pickup(hack=False)
        # self.move_utensil_to_pose(waypoint_1_tip)

        # move to home position
        self.robot_controller.reset()

        # reset wrist
        self.wrist_controller.reset()

    def scooping_pickup(self, hack = True):

        forkpitch_to_tip = self.tf_utils.getTransformationFromTF('forkpitch', 'fork_tip')
        distance = forkpitch_to_tip[0,3]

        print("Distance: ", distance)

        fork_base = self.tf_utils.getTransformationFromTF('base_link', 'forkbase')

        fork_base_displacement = np.eye(4)
        fork_base_displacement[0,3] = distance/8
        fork_base_displacement[2,3] = -distance*3/4

        fork_base_target = fork_base @ fork_base_displacement

        forkbase_to_tool_frame = self.tf_utils.getTransformationFromTF('forkbase', 'tool_frame')
        tool_frame_target = fork_base_target @ forkbase_to_tool_frame

        self.tf_utils.publishTransformationToTF('base_link', 'tool_frame_target', tool_frame_target)

        if hack:
            scoop_thread = threading.Thread(target=self.wrist_controller.scoop_wrist_hack)
        else:
            scoop_thread = threading.Thread(target=self.wrist_controller.scoop_wrist)
        scoop_thread.start()

        # input("Press enter to also move the robot...")
        self.robot_controller.move_to_pose(tool_frame_target)

        # wait for scoop thread to finish
        scoop_thread.join()

    def pushing_skill(self, color_image, depth_image, camera_info, keypoints = None):
        
        if keypoints is not None:
            start, end = keypoints
        else:
            print("Click on the start and end points for pushing in the pixel selector window.")
            clicks = self.pixel_selector.run(color_image, num_clicks=2)
            start = clicks[0]
            end = clicks[1]

        push_angle = utils.angle_between_pixels(np.array(start), np.array(end), color_image.shape[1], color_image.shape[0], orientation_symmetry = False)

        validity, start_vec_3d = utils.pixel2World(camera_info, start[0], start[1], depth_image)
        if not validity:
            print("ERROR: Pushing start point has invalid depth")
            return
        
        validity, end_vec_3d = utils.pixel2World(camera_info, end[0], end[1], depth_image)
        if not validity:
            print("ERROR: Pushing end point has invalid depth")
            return
        
        pushing_start_pose = np.zeros((4,4))
        pushing_start_pose[:3,:3] = Rotation.from_euler('xyz', [0,0,push_angle], degrees=True).as_matrix()
        pushing_start_pose[:3,3] = start_vec_3d.reshape(1,3)
        pushing_start_pose[3,3] = 1
        pushing_start_pose = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame") @ pushing_start_pose
        pushing_start_pose[2,3] = self.plate_height

        pushing_end_pose = np.zeros((4,4))
        pushing_end_pose[:3,:3] = Rotation.from_euler('xyz', [0,0,push_angle], degrees=True).as_matrix()
        pushing_end_pose[:3,3] = end_vec_3d.reshape(1,3)
        pushing_end_pose[3,3] = 1
        pushing_end_pose = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame") @ pushing_end_pose
        pushing_end_pose[2,3] = self.plate_height

        print("Pushing points height: ", pushing_start_pose[2,3], pushing_end_pose[2,3])

        # visualize 
        self.visualizer.visualize_food(pushing_start_pose, id = 0)
        self.visualizer.visualize_food(pushing_end_pose, id = 1)

        input("Check visualized fork tip poses (red cubes) on rviz. If correct, press ENTER to execute action. Otherwise, press CTRL+C to exit.")

        # action 1: Move to above start position
        waypoint_1 = np.copy(pushing_start_pose)
        waypoint_1[2,3] += 0.05
        self.move_utensil_to_pose(waypoint_1)

        # action 2: Move down until tip touches plate
        waypoint_2 = np.copy(pushing_start_pose)
        self.move_utensil_to_pose(waypoint_2)

        # action 3: Move to end position
        waypoint_3 = np.copy(pushing_end_pose)
        self.move_utensil_to_pose(waypoint_3)

        # action 4: Move a bite up
        waypoint_4 = self.tf_utils.getTransformationFromTF('base_link', 'fork_tip')
        waypoint_4[2,3] += 0.05
        self.move_utensil_to_pose(waypoint_4) 

        # action 5: Move to above start position
        self.robot_controller.reset()

    def twirling_skill(self, color_image, depth_image, camera_info, keypoint = None, twirl_angle = None):
        
        if keypoint is not None:
            (center_x, center_y) = keypoint
        else:
            print("Click on the point to twirl in the pixel selector window.")
            clicks = self.pixel_selector.run(color_image)
            (center_x, center_y) = clicks[0]
            twirl_angle = 90

        validity, center_point = utils.pixel2World(camera_info, center_x, center_y, depth_image)
        if not validity:
            print("ERROR: Twirl point has invalid depth")  
            return
        twirl_angle = 90 + twirl_angle

        twirl_camera_frame = np.zeros((4,4))
        twirl_camera_frame[:3,:3] = Rotation.from_euler('xyz', [0,0,twirl_angle], degrees=True).as_matrix()
        twirl_camera_frame[:3,3] = center_point.reshape(1,3)
        twirl_camera_frame[3,3] = 1

        base_to_camera = self.tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame")
        twirl_base_frame = base_to_camera @ twirl_camera_frame
        twirl_base_frame[2,3] = self.plate_height

        self.visualizer.visualize_food(twirl_base_frame)
        self.tf_utils.publishTransformationToTF("base_link", "target_fork_tip", twirl_base_frame)

        print("Twirl point height: ", twirl_base_frame[2,3])

        input("Check visualized fork tip poses (red cubes) on rviz. If correct, press ENTER to execute action. Otherwise, press CTRL+C to exit.")

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

        # action 5: Move to above position
        self.robot_controller.reset()

    def transfer_to_mouth(self, OFFSET = 0.1):

        self.robot_controller.move_to_transfer_pose()
        inp = input('Detect mouth center? (y/n): ')
        while inp != 'y':
            inp = input('Detect mouth center? (y/n): ')

        # check if mouth is open
        while True:
            mouth_open = rospy.wait_for_message('/mouth_open', Bool)
            if mouth_open.data:
                break
            else:
                print('Mouth is closed.')
    
        mouth_center_3d_msg = rospy.wait_for_message('/mouth_center', Point)
        mouth_center_3d = np.array([mouth_center_3d_msg.x, mouth_center_3d_msg.y, mouth_center_3d_msg.z])

        input("Press ENTER to move in front of mouth.")

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

        input("Press ENTER to move back to before transfer pose.")
        self.robot_controller.move_to_transfer_pose()