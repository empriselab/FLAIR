import cv2
import ast
import numpy as np
from scipy.spatial.transform import Rotation
import math
import os

import rospy
from sensor_msgs.msg import Image, JointState

from rs_ros import RealSenseROS
import pickle
import yaml
import threading
import time

# ROBOT = 'kinova-deployment' # 'kinova' or 'franka' or 'kinova-deployment' (used for controller on the NUC)
ROBOT = 'kinova-deployment'    

from rs_ros import RealSenseROS
from pixel_selector import PixelSelector

if ROBOT == 'franka':
    from robot_controller.franka_controller import FrankaRobotController
elif ROBOT == 'kinova':
    from robot_controller.kinova_controller import KinovaRobotController
elif ROBOT == 'kinova-deployment':
    from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient
else:
    raise ValueError("Invalid robot type")

from skill_library import SkillLibrary

# package imports
import flair_utils

from visualization_msgs.msg import Marker, MarkerArray

import sys

from inference_class import BiteAcquisitionInference

from vision_utils import visualize_push, visualize_keypoints, visualize_skewer

HOME_ORIENTATION = Rotation.from_quat([1/math.sqrt(2), 1/math.sqrt(2), 0, 0]).as_matrix()
DEFAULT_FORCE_THRESHOLD = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]

class FeedingBot:
    def __init__(self):

        rospy.init_node('FeedingBot')
        
        self.camera = RealSenseROS()

        if ROBOT == 'franka':
            config_path = "/home/limbrepos/feeding_ws/src/franka_feeding/configs/feeding.yaml"
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            robot_controller = FrankaRobotController(config)
        elif ROBOT == 'kinova':
            robot_controller = KinovaRobotController()
        elif ROBOT == 'kinova-deployment':
            robot_controller = ArmInterfaceClient()

            # Rajat Just for testing
            above_plate_pos = [4.119619921793763, 5.927367810785151, 4.797271913808785, 4.641709217686205, 4.980350922946283, 5.268199221999715, 4.814377930122582]
            robot_controller.set_joint_position(above_plate_pos)

            def publish_joint_states(arm):

                # publish joint states
                joint_states_pub = rospy.Publisher("/robot_joint_states", JointState, queue_size=10)

                while not rospy.is_shutdown():
                    arm_pos, ee_pose, gripper_pos = arm.get_state()
                    joint_state_msg = JointState()
                    joint_state_msg.header.stamp = rospy.Time.now()
                    joint_state_msg.name = [
                        "joint_1",
                        "joint_2",
                        "joint_3",
                        "joint_4",
                        "joint_5",
                        "joint_6",
                        "joint_7",
                        "finger_joint",
                    ]
                    joint_state_msg.position = arm_pos.tolist() + [gripper_pos]
                    joint_state_msg.velocity = [0.0] * 8
                    joint_state_msg.effort = [0.0] * 8
                    joint_states_pub.publish(joint_state_msg)
                    time.sleep(0.01)

            joint_state_thread = threading.Thread(target=publish_joint_states, args=(robot_controller,))
            joint_state_thread.start()

        self.skill_library = SkillLibrary(robot_controller)

        print("Feeding Bot initialized")

        self.inference_server = BiteAcquisitionInference(mode='ours')

        if not os.path.exists("log"):
            os.mkdir("log")
        self.log_file = "log/"
        files = os.listdir(self.log_file)

        if not os.path.exists('history.txt'):
            self.log_count = 1
            self.bite_history = []
        else:
            with open('history.txt', 'r') as f:
                self.bite_history = ast.literal_eval(f.read().strip())
                self.log_count = len(self.bite_history)+1

        print("Log count", self.log_count)
        print("History", self.bite_history)
                
        # self.log_count should be the maximum numbered file in the log folder + 1
        self.log_count = max([int(x.split('_')[0]) for x in files]) + 1 if len(files) > 0 else 1
        #self.log_count = len(os.listdir('log')) + 1
        

    def clear_plate(self):

        # Identify the items on the plate
        # camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
        # items = self.inference_server.recognize_items(camera_color_data)
        # print("Food Items recognized:", items)

        # input("Did the robot recognize the food items correctly?")
        
        #items = ['mashed potatoes', 'sausage']
        # items = ['noodles', 'meatball']
        # items = ['brownie', 'chocolate sauce']
        # self.inference_server.FOOD_CLASSES = [f.replace('banana', 'small piece of sliced banana') for f in items]
        # items = ['banana', 'chocolate sauce']
        items = ['oatmeal', 'strawberry']
        # items = ['red strawberry', 'chocolate sauce', 'ranch dressing', 'blue plate']
        # items = ['mashed potatoes']
        # items =  ['strawberry', 'ranch dressing', 'blue plate']
        # self.inference_server.FOOD_CLASSES = [f.replace('banana', 'small piece of sliced banana') for f in items]
        self.inference_server.FOOD_CLASSES = items

        # User preference
        #user_preference = "I want to eat all the mashed potatoes first, and the sausages after."
        #user_preference = "Alternating bites of spaghetti and meatballs."
        # user_preference = "No preference."
        user_preference = "I want to eat alternating bites of strawberries and oatmeal."

        # Bite history
        bite_history = self.bite_history

        # Continue food
        continue_food_label = None
        continue_dip_label = None

        visualize = True

        actions_remaining = 10
        success = True
        while actions_remaining:
        # if True:

            self.skill_library.reset()

            print('History', bite_history)
            print('Actions remaining', actions_remaining)
            input('Ready?')
            camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
            vis = camera_color_data.copy()

            log_path = self.log_file + str(self.log_count)
            self.log_count += 1

            annotated_image, detections, item_masks, item_portions, item_labels = self.inference_server.detect_items(camera_color_data, log_path)

            item_labels = [l.replace('strawberry piece', 'strawberry') for l in item_labels]   
            item_labels = [l.replace('orange baby carrot', 'baby carrot') for l in item_labels]   

            cv2.imshow('vis', annotated_image)
            cv2.waitKey(0)

            input("Visualzing the detected items. Press Enter to continue.")

            k = input('Are detected items correct?')
            while k not in ['y', 'n']:
                k = input('Are detected items correct?')
                if k == 'e':
                    exit(1)
            while k == 'n':
                exit(1)
                # print("Please manually give the correct labels")
                # print("Detected items:", item_labels)
                # label_id = int(input("What label to correct?"))
                # item_labels[label_id] = input("Correct label:")

                # annotated_image = self.inference_server.get_annotated_image(camera_color_data, detections, item_labels)

                # cv2.imshow('vis', annotated_image)
                # cv2.waitKey(0)

                # input("Visualzing the detected items. Press Enter to continue.")

                # k = input('Are detected items correct now?')
                # while k not in ['y', 'n']:
                #     k = input('Are detected items correct now?')

            cv2.destroyAllWindows()
            
            
            clean_item_labels, _ = self.inference_server.clean_labels(item_labels)

            # remove detections of blue plate
            if 'blue plate' in clean_item_labels:
                idx = clean_item_labels.index('blue plate')
                clean_item_labels.pop(idx)
                item_labels.pop(idx)
                item_masks.pop(idx)
                item_portions.pop(idx)

            print("----- Clean Item Labels:", clean_item_labels)
                
            

            cv2.imwrite(log_path + "_annotated.png", annotated_image)
            cv2.imwrite(log_path + "_color.png", camera_color_data)
            cv2.imwrite(log_path + "_depth.png", camera_depth_data)

            categories = self.inference_server.categorize_items(item_labels) 

            print("--------------------")
            print("Labels:", item_labels)
            print("Categories:", categories)
            print("Portions:", item_portions)
            print("--------------------")

            category_list = []
            labels_list = []
            per_food_masks = [] # For multiple items per food, ordered by prediction confidence
            per_food_portions = []

            # for i in range(len(categories)):
            #     if categories[i] not in category_list:
            #         category_list.append(categories[i])
            #         labels_list.append(clean_item_labels[i])
            #         per_food_masks.append([item_masks[i]])
            #         per_food_portions.append(item_portions[i])
            #     else:
            #         index = category_list.index(categories[i])
            #         per_food_masks[index].append(item_masks[i])
            #         per_food_portions[index] += item_portions[i] 

            for i in range(len(categories)):
                if labels_list.count(clean_item_labels[i]) == 0:
                    category_list.append(categories[i])
                    labels_list.append(clean_item_labels[i])
                    per_food_masks.append([item_masks[i]])
                    per_food_portions.append(item_portions[i])
                else:
                    index = labels_list.index(clean_item_labels[i])
                    per_food_masks[index].append(item_masks[i])
                    per_food_portions[index] += item_portions[i]
            
            print("Bite History", bite_history)
            print("Category List:", category_list)
            print("Labels List:", labels_list)
            print("Per Food Masks Len:", [len(x) for x in per_food_masks])
            print("Per Food Portions:", per_food_portions)

            #food_id, action_type, metadata = self.inference_server.get_manual_action(annotated_image, camera_color_data, per_food_masks, category_list, per_food_portions, user_preference, bite_history, continue_food_id, log_path)
            # food, dip = self.inference_server.get_manual_action(annotated_image, camera_color_data, per_food_masks, category_list, labels_list, per_food_portions, user_preference, bite_history, continue_food_label, log_path)
            food, dip = self.inference_server.get_autonomous_action(annotated_image, camera_color_data, per_food_masks, category_list, labels_list, per_food_portions, user_preference, bite_history, continue_food_label, continue_dip_label, log_path)
            if food is None:
                exit(1)
            food_id, action_type, metadata = food
            dip_id = None
            if dip is not None:
                dip_id, dip_action_type, dip_metadata = dip


            if action_type == 'Twirl':
                densest_point = metadata['point']
                twirl_angle = metadata['twirl_angle']
                if visualize:
                    vis = visualize_keypoints(vis, [densest_point])
                    cv2.imshow('vis', vis)
                    cv2.waitKey(0)
                input('Continue twirling skill?')
                action = self.skill_library.twirling_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = densest_point, twirl_angle = twirl_angle)
            elif action_type == 'Skewer':
                center = metadata['point']
                skewer_angle = metadata['skewer_angle']
                if visualize:
                    vis = visualize_skewer(vis, center, skewer_angle)
                    cv2.imshow('vis', vis)
                    cv2.waitKey(0)
                input('Continue skewering skill?')
                action = self.skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = center, skewer_angle = skewer_angle)
                # keep skewering until the food is on the fork
                food_on_fork = self.inference_server.food_on_fork(self.camera.get_camera_data()[1], visualize=False, log_path=log_path)
                print('Food on fork?', food_on_fork)
                #    action = self.skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = center, skewer_angle = skewer_angle)
            elif action_type == 'Scoop':
                start, end = metadata['start'], metadata['end']
                action = self.skill_library.scooping_skill(camera_color_data, camera_depth_data, camera_info_data, keypoints = [start, end])
            elif action_type == 'Push':
                continue_food_label = labels_list[food_id]
                start, end = metadata['start'], metadata['end']
                if visualize:
                    vis = visualize_push(vis, start, end)
                    cv2.imshow('vis', vis)
                    cv2.waitKey(0)
                input('Continue pushing skill?')
                action = self.skill_library.pushing_skill(camera_color_data, camera_depth_data, camera_info_data, keypoints = [start, end])
            elif action_type == 'Cut':
                continue_food_label = labels_list[food_id]
                if dip_id is not None and dip_action_type == 'Dip':
                    continue_dip_label = labels_list[dip_id]
                cut_point = metadata['point']
                cut_angle = metadata['cut_angle']
                action = self.skill_library.cutting_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = cut_point, cutting_angle = cut_angle)            

            if action_type == 'Twirl' or action_type == 'Scoop': # Terminal actions
                continue_food_label = None
                bite_history.append(labels_list[food_id])
            elif action_type == 'Skewer':
                if food_on_fork: # Terminal actions
                    # Dip the food
                    if dip_id is not None and dip_action_type == 'Dip':
                        dip_point = dip_metadata['point']
                        action = self.skill_library.dipping_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = dip_point)
                        bite_history.append(labels_list[food_id])
                        bite_history.append(labels_list[dip_id])
                        continue_dip_label = None
                    else:
                        bite_history.append(labels_list[food_id])
                    continue_food_label = None
                    
                else:
                    continue_food_label = labels_list[food_id]
                    success = False

            if success:
                actions_remaining -= 1

            with open('history.txt', 'w') as f:
                f.write(str(bite_history))

            k = input('Continue to transfer? Remember to start horizontal spoon.')
            while k not in ['y', 'n']:
                k = input('Continue to transfer? Remember to start horizontal spoon.')
            if k == 'y':
                self.skill_library.transfer_to_mouth()
                k = input('Continue to acquisition? Remember to shutdown horizontal spoon.')
                while k not in ['y', 'n']:
                    k = input('Continue to acquisition? Remember to shutdown horizontal spoon.')
                if k == 'n':
                    exit(1)

if __name__ == "__main__":

    args = None
    feeding_bot = FeedingBot()
    # feeding_bot.skill_library.reset()
    feeding_bot.clear_plate()
    # feeding_bot.skill_library.reset()
    # feeding_bot.skill_library.twirl_wrist()
    # feeding_bot.skill_library.scoop_wrist_hack()
    # feeding_bot.skill_library.set_wrist_state(0.4*math.pi, 0)
    # food_on_fork = feeding_bot.inference_server.food_on_fork(feeding_bot.camera.get_camera_data()[1], visualize=True)
