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
    from feeding_deployment.robot_controller.command_interface import JointCommand, CartesianCommand
else:
    raise ValueError("Invalid robot type")
from wrist_controller import WristController

from skill_library import SkillLibrary

# package imports
import flair_utils

from visualization_msgs.msg import Marker, MarkerArray

import sys

from inference_class import BiteAcquisitionInference

from vision_utils import visualize_push, visualize_keypoints, visualize_skewer

HOME_ORIENTATION = Rotation.from_quat([1/math.sqrt(2), 1/math.sqrt(2), 0, 0]).as_matrix()
DEFAULT_FORCE_THRESHOLD = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]

class FLAIR:
    def __init__(self, robot_controller, wrist_controller, no_waits=False):

        print("Feeding Bot initialized")

        self.skill_library = SkillLibrary(robot_controller, wrist_controller, no_waits)
        self.no_waits = no_waits
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

        # User preference
        self.user_preference = "No preference."

        # Continue food
        self.continue_food_label = None
        self.continue_dip_label = None

        self.visualize = True

        # itermediate variables
        self.items_detection = None
        self.next_action_prediction = None
        
    def identify_plate(self, camera_color_data):

        items = self.inference_server.recognize_items(camera_color_data)
        print("Food Items recognized:", items)

        # k = input("Did the robot recognize the food items correctly?")
        k = 'n'
        if k == 'n':
            # Rajat ToDo: Implement manual input of food items        
            items = ['yellow banana', 'baby carrot']
            
        self.inference_server.FOOD_CLASSES = items
        return items

    def set_food_items(self, items):
        self.inference_server.FOOD_CLASSES = items

    def set_preferences(self, user_preference):
        self.user_preference = user_preference

    def detect_items(self, camera_color_data, camera_depth_data, camera_info_data, log_path):

        vis = camera_color_data.copy()

        log_path = self.log_file + str(self.log_count)
        self.log_count += 1

        annotated_image, detections, item_masks, item_portions, item_labels, plate_bounds = self.inference_server.detect_items(camera_color_data, log_path)

        item_bounding_boxes = []
        item_bounding_boxes_plate = []
        # add bounding boxes corresponding to the detected item masks
        for mask in item_masks:
            non_zero_points = cv2.findNonZero(mask)
            x, y, w, h = cv2.boundingRect(non_zero_points)
            item_bounding_boxes.append([x, y, w, h]) # original image coordinates
            item_bounding_boxes_plate.append([x-plate_bounds[0], y-plate_bounds[1], w, h]) # plate image coordinates

        if not self.no_waits:
            cv2.imshow('vis', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # input("Visualzing the detected items. Press Enter to continue.")

        # k = input('Are detected items correct?')
        # while k not in ['y', 'n']:
        #     k = input('Are detected items correct?')
        #     if k == 'e':
        #         return None
        # while k == 'n':
        #     return None
            # print("Please manually give the correct labels")
            # print("Detected items:", item_labels)
            # label_id = int(input("What label to correct?"))
            # item_labels[label_id] = input("Correct label:")

            # annotated_image = self.inference_server.get_annotated_image(camera_color_data, detections, item_labels)

            # cv2.imshow('vis', annotated_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # input("Visualzing the detected items. Press Enter to continue.")

            # k = input('Are detected items correct now?')
            # while k not in ['y', 'n']:
            #     k = input('Are detected items correct now?')
        
        clean_item_labels, _ = self.inference_server.clean_labels(item_labels)

        # remove detections of blue plate
        if 'blue plate' in clean_item_labels:
            idx = clean_item_labels.index('blue plate')
            clean_item_labels.pop(idx)
            item_labels.pop(idx)
            item_masks.pop(idx)
            item_bounding_boxes.pop(idx)
            item_bounding_boxes_plate.pop(idx)
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
        per_food_bounding_boxes = []
        per_food_bounding_boxes_plate = []

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
                per_food_bounding_boxes.append([item_bounding_boxes[i]])
                per_food_bounding_boxes_plate.append([item_bounding_boxes_plate[i]])
                per_food_portions.append(item_portions[i])
            else:
                index = labels_list.index(clean_item_labels[i])
                per_food_masks[index].append(item_masks[i])
                per_food_bounding_boxes[index].append(item_bounding_boxes[i])
                per_food_bounding_boxes_plate[index].append(item_bounding_boxes_plate[i])
                per_food_portions[index] += item_portions[i]
        
        print("Bite History", self.bite_history)
        print("Category List:", category_list)
        print("Labels List:", labels_list)
        print("Per Food Masks Len:", [len(x) for x in per_food_masks])
        print("Per Food Portions:", per_food_portions)

        plate_image = camera_color_data.copy()[plate_bounds[1]:plate_bounds[1]+plate_bounds[3], plate_bounds[0]:plate_bounds[0]+plate_bounds[2]]

        food_type_to_bounding_boxes = {label: [] for label in labels_list}
        food_type_to_bounding_boxes_plate = {label: [] for label in labels_list}
        food_type_to_masks = {label: [] for label in labels_list}
        food_type_to_skill = {label: None for label in labels_list}

        for i in range(len(labels_list)):
            food_type_to_bounding_boxes[labels_list[i]] = per_food_bounding_boxes[i]
            food_type_to_bounding_boxes_plate[labels_list[i]] = per_food_bounding_boxes_plate[i]
            food_type_to_masks[labels_list[i]] = per_food_masks[i]
            if categories[i] == 'noodles':
                food_type_to_skill[labels_list[i]] = 'Twirl'
            elif categories[i] == 'semisolid':
                food_type_to_skill[labels_list[i]] = 'Scoop'
            else:
                food_type_to_skill[labels_list[i]] = 'Skewer'

        # Rajat ToDo: Remove repeated code
        items_detection = {
            'annotated_image': annotated_image,
            'plate_image': plate_image,
            'plate_bounds': plate_bounds,
            'per_food_masks': per_food_masks, 
            'category_list': category_list, 
            'labels_list': labels_list, 
            'per_food_portions': per_food_portions,
            'food_type_to_bounding_boxes': food_type_to_bounding_boxes,
            'food_type_to_bounding_boxes_plate': food_type_to_bounding_boxes_plate,
            'food_type_to_masks': food_type_to_masks,
            'food_type_to_skill': food_type_to_skill
        }

        self.items_detection = items_detection
        return items_detection
    
    def get_items_detection(self):
        return self.items_detection
    
    def update_items_detection(self, items_detection):
        self.items_detection = items_detection

    def predict_next_action(self, camera_color_data, items_detection, log_path):

        if items_detection is None:
            items_detection = self.items_detection

        annotated_image = items_detection['annotated_image']
        per_food_masks = items_detection['per_food_masks']
        category_list = items_detection['category_list']
        per_food_masks = items_detection['per_food_masks']
        labels_list = items_detection['labels_list']
        per_food_portions = items_detection['per_food_portions']
        
        food, dip, bite_mask_idx = self.inference_server.get_autonomous_action(annotated_image, camera_color_data, per_food_masks, category_list, labels_list, per_food_portions, self.user_preference, self.bite_history, self.continue_food_label, self.continue_dip_label, log_path)
        if food is None:
            return None
        
        food_id, action_type, metadata = food
        if dip is not None:
            dip_id, dip_action_type, dip_metadata = dip
        else:
            dip_id, dip_action_type, dip_metadata = None, None, None
        
        # next bite food item
        next_action_prediction = {
            'food_id': food_id,
            'action_type': action_type,
            'metadata': metadata,
            'dip_id': dip_id,
            'dip_action_type': dip_action_type,
            'dip_metadata': dip_metadata,
            'labels_list': labels_list,
            'bite_mask_idx': bite_mask_idx
        }

        # Rajat ToDo: Update the detections with bite_mask_idx as the first mask for the food item
        
        self.next_action_prediction = next_action_prediction
        return next_action_prediction
    
    def execute_action(self, camera_color_data, camera_depth_data, camera_info_data, next_action_prediction, log_path):

        if next_action_prediction is None:
            next_action_prediction = self.next_action_prediction

        food_id = next_action_prediction['food_id']
        action_type = next_action_prediction['action_type']
        metadata = next_action_prediction['metadata']
        dip_id = next_action_prediction['dip_id']
        dip_action_type = next_action_prediction['dip_action_type']
        dip_metadata = next_action_prediction['dip_metadata']
        labels_list = next_action_prediction['labels_list']

        vis = camera_color_data.copy()
        if action_type == 'Twirl':
            densest_point = metadata['point']
            twirl_angle = metadata['twirl_angle']
            if self.visualize:
                vis = visualize_keypoints(vis, [densest_point])
                cv2.imshow('vis', vis)
                cv2.waitKey(0)
            input('Continue twirling skill?')
            action = self.skill_library.twirling_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = densest_point, twirl_angle = twirl_angle)
        elif action_type == 'Skewer':
            center = metadata['point']
            skewer_angle = metadata['skewer_angle']
            if self.visualize:
                vis = visualize_skewer(vis, center, skewer_angle)
                cv2.imshow('vis', vis)
                cv2.waitKey(0)
            input('Continue skewering skill?')
            action = self.skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = center, major_axis = skewer_angle)
            # keep skewering until the food is on the fork
            # food_on_fork = self.inference_server.food_on_fork(self.camera.get_camera_data()[1], self.visualize=False, log_path=log_path)
            # print('Food on fork?', food_on_fork)
            food_on_fork = True
            #    action = self.skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = center, skewer_angle = skewer_angle)
            
        elif action_type == 'Scoop':
            start, end = metadata['start'], metadata['end']
            action = self.skill_library.scooping_skill(camera_color_data, camera_depth_data, camera_info_data, keypoints = [start, end])
        elif action_type == 'Push':
            self.continue_food_label = labels_list[food_id]
            start, end = metadata['start'], metadata['end']
            if self.visualize:
                vis = visualize_push(vis, start, end)
                cv2.imshow('vis', vis)
                cv2.waitKey(0)
            input('Continue pushing skill?')
            action = self.skill_library.pushing_skill(camera_color_data, camera_depth_data, camera_info_data, keypoints = [start, end])
        elif action_type == 'Cut':
            self.continue_food_label = labels_list[food_id]
            if dip_id is not None and dip_action_type == 'Dip':
                self.continue_dip_label = labels_list[dip_id]
            cut_point = metadata['point']
            cut_angle = metadata['cut_angle']
            action = self.skill_library.cutting_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = cut_point, cutting_angle = cut_angle)            

        if action_type == 'Twirl' or action_type == 'Scoop': # Terminal actions
            self.continue_food_label = None
            self.bite_history.append(labels_list[food_id])
        elif action_type == 'Skewer':
            if food_on_fork: # Terminal actions
                # Dip the food
                if dip_id is not None and dip_action_type == 'Dip':
                    dip_point = dip_metadata['point']
                    action = self.skill_library.dipping_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = dip_point)
                    self.bite_history.append(labels_list[food_id])
                    self.bite_history.append(labels_list[dip_id])
                    self.continue_dip_label = None
                else:
                    self.bite_history.append(labels_list[food_id])
                self.continue_food_label = None
                
            else:
                self.continue_food_label = labels_list[food_id]
                success = False

        with open('history.txt', 'w') as f:
            f.write(str(self.bite_history))

    def execute_manual_action(self, action_type, camera_color_data, camera_depth_data, camera_info_data):

        if action_type == 'Skewer':
            self.skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data)
        elif action_type == 'Scoop':
            self.skill_library.scooping_skill(camera_color_data, camera_depth_data, camera_info_data)
        elif action_type == 'Twirl':
            self.skill_library.twirling_skill(camera_color_data, camera_depth_data, camera_info_data)

if __name__ == "__main__":

    rospy.init_node('FLAIR')
    camera = RealSenseROS()

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
        above_plate_pos = [-2.86495014, -1.61460533, -2.6115943, -1.37673391, 1.11842806, -1.17904586, -2.6957422]
        robot_controller.execute_command(JointCommand(above_plate_pos))
    
    wrist_controller = WristController()
    wrist_controller.set_velocity_mode()
    wrist_controller.reset()

    flair = FLAIR(robot_controller, wrist_controller)

    camera_header, camera_color_data, camera_info_data, camera_depth_data = camera.get_camera_data()
    items = flair.identify_plate(camera_color_data)
    # flair.set_food_items(items)
    flair.set_food_items(['banana slice'])
    items_detection = flair.detect_items(camera_color_data, camera_depth_data, camera_info_data, log_path=None)
    print(" --- Food items detected:", items_detection['clean_item_labels'])
    next_action_prediction = flair.predict_next_action(camera_color_data, items_detection=None, log_path=None)
    print(" --- Next Food Item Prediction:", next_action_prediction['labels_list'][next_action_prediction['food_id']])
    print(" --- Next Action Prediction:", next_action_prediction['action_type'])
    flair.execute_action(camera_color_data, camera_depth_data, camera_info_data, next_action_prediction=None, log_path=None)