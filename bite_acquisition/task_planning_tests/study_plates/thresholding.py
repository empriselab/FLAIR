import cv2
import time
import os
import numpy as np
import supervision as sv


import torch
import torchvision
from torchvision.transforms import ToTensor

from groundingdino.util.inference import Model

import sys
sys.path.append('../..')

from scripts.vision_utils import resize_to_square
from scripts.inference_class import BiteAcquisitionInference

import os
from openai import OpenAI
import ast

import base64
import requests
import cmath
import math

from scripts.src.food_pos_ori_net.model.minispanet import MiniSPANet
from scripts.src.spaghetti_segmentation.model import SegModel
import torchvision.transforms as transforms

print('imports done')

# PLATE_NUMBER = 1 # 1, 2, 3, 4, 6
TRAIN = False

def run_tests(plate_number):
    
    inference_server = BiteAcquisitionInference('ours')

    SOURCE_DIR = '/home/isacc/bite_acquisition/task_planning_tests/study_plates/'

    if plate_number == 1:
        PLATE_NAME = 'spaghetti_meatballs'
        items = ['spaghetti', 'meatball']
        CATEGORY = 'noodle'
    elif plate_number == 2:
        PLATE_NAME = 'fettuccine_chicken_broccoli'
        items = ['fettuccine', 'chicken piece', 'broccoli']
        CATEGORY = 'noodle'
    elif plate_number == 3:
        PLATE_NAME = 'mashed_potato_sausage'
        items = ['mashed potato', 'sausage']
        CATEGORY = 'semisolid'
    elif plate_number == 4:
        PLATE_NAME = 'oatmeal_strawberry'
        items = ['strawberry', 'oatmeal']
        CATEGORY = 'semisolid'
    elif plate_number == 6:
        PLATE_NAME = 'dessert'
        items = ['brownie', 'banana piece', 'banana slice', 'cut banana', 'chocolate sauce']
        CATEGORY = 'cuttable'

    if CATEGORY == 'noodle':
        CANDIDATE_ACTIONS = ['twirl', 'group', 'push']
    elif CATEGORY == 'semisolid':
        CANDIDATE_ACTIONS = ['scoop', 'push']
    elif CATEGORY == 'cuttable':
        CANDIDATE_ACTIONS = ['cut', 'acquire']

    if TRAIN:
        INPUT_DIR = SOURCE_DIR + 'log/' + PLATE_NAME + '/classification_format/train'
        OUTPUT_DIR = SOURCE_DIR + 'param_tuning/thresholding/' + PLATE_NAME
        PREDICTION_DIR = OUTPUT_DIR + '/predictions.txt'
    else:
        INPUT_DIR = SOURCE_DIR + 'log/' + PLATE_NAME + '/classification_format/test'
        OUTPUT_DIR = SOURCE_DIR + 'outputs/thresholding/' + PLATE_NAME
        PREDICTION_DIR = OUTPUT_DIR + '/predictions.txt'
        
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    inference_server.FOOD_CLASSES = items

    correct_count = 0
    total_valid_count = 0
    total_count = 0
    
    prediction_file = open(PREDICTION_DIR, 'w')
    for action_label in CANDIDATE_ACTIONS:
        if not os.path.exists(OUTPUT_DIR + '/' + action_label):
            os.makedirs(OUTPUT_DIR + '/' + action_label)
        TEST_IMGS = os.listdir(os.path.join(INPUT_DIR, action_label))
        # sort images by number
        TEST_IMGS.sort(key=lambda x: int(x.split('_')[0]))
        for img in TEST_IMGS:
            # if img != '386_color.png':
            # if img != '1170_color.png':
                # continue
            SOURCE_IMAGE_PATH = os.path.join(INPUT_DIR, action_label, img)
            print(SOURCE_IMAGE_PATH)
            camera_color_data = cv2.imread(SOURCE_IMAGE_PATH)
            print("Camera color data shape:", camera_color_data.shape)
            # if CATEGORY == 'noodle':
            camera_color_data = resize_to_square(camera_color_data, 480)

            print("Camera color data shape:", camera_color_data.shape)

            total_count += 1
            # try: 
            annotated_image, detections, item_masks, item_portions, item_labels = inference_server.detect_items(camera_color_data, log_path = None)

            clean_item_labels, _ = inference_server.clean_labels(item_labels)

            # remove detections of blue plate
            if 'blue plate' in clean_item_labels:
                idx = clean_item_labels.index('blue plate')
                clean_item_labels.pop(idx)
                item_labels.pop(idx)
                item_masks.pop(idx)
                item_portions.pop(idx)

            print("----- Clean Item Labels:", clean_item_labels)

            categories = inference_server.categorize_items(item_labels) 

            print("--------------------")
            print("Labels:", item_labels)
            print("Categories:", categories)
            print("Portions:", item_portions)
            print("--------------------")

            category_list = []
            labels_list = []
            per_food_masks = [] # For multiple items per food, ordered by prediction confidence
            per_food_portions = []

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

            if CATEGORY == 'noodle':
                if 'noodles' not in categories:
                    print("No noodles detected")
                    continue
                densest, sparsest, twirl_angle, filling_push_start, filling_push_end, valid_actions, valid_actions_vis, heatmap, action, noodle_mask, furthest_vis = inference_server.get_noodle_action(camera_color_data, per_food_masks, category_list)
            
                total_valid_count += 1
                if (action == 'Acquire' and action_label == 'twirl') or (action == 'Group' and action_label == 'group') or (action == 'Push Filling' and action_label == 'push'):
                    correct_count += 1

                # convert to BGR, 3 channel
                noodle_mask = cv2.cvtColor(noodle_mask, cv2.COLOR_GRAY2BGR)

                # print('Valid actions size: ', valid_actions_vis.shape)
                # print('Noodle mask size: ', noodle_mask.shape)

                # kk = input("Press any key to continue")
                # valid_actions_vis = cv2.cvtColor(valid_actions_vis, cv2.COLOR_GRAY2BGR) # Doesn't work

                print("Writing at path: ", os.path.join(OUTPUT_DIR + '/' + action_label + '/', img))
                cv2.imwrite(os.path.join(OUTPUT_DIR + '/' + action_label + '/', img), np.hstack([annotated_image, noodle_mask, furthest_vis, valid_actions_vis, heatmap]))
            
            elif CATEGORY == 'semisolid':
                if 'semisolid' not in categories:
                    print("No semisolid detected")
                    continue
                densest, sparsest, filling_push_start, filling_push_end, valid_actions, valid_actions_vis, heatmap, action, start_px, end_px, color_image_vis, semisolid_mask, furthest_vis  = inference_server.get_scoop_action(camera_color_data, per_food_masks, category_list, log_path=None)

                total_valid_count += 1
                if (action == 'Acquire' and action_label == 'scoop') or (action == 'Push Filling' and action_label == 'push'):
                    correct_count += 1

                semisolid_mask = cv2.cvtColor(semisolid_mask, cv2.COLOR_GRAY2BGR)

                print("Writing at path: ", os.path.join(OUTPUT_DIR + '/' + action_label + '/', img))
                cv2.imwrite(os.path.join(OUTPUT_DIR + '/' + action_label + '/', img), np.hstack([annotated_image, semisolid_mask, furthest_vis, valid_actions_vis, heatmap]))

            elif CATEGORY == 'cuttable':
                # extract banana mask
                banana_masks = None
                for i in range(len(labels_list)):
                    if 'banana' in labels_list[i]:
                        banana_masks = per_food_masks[i]
                        break

                requires_cut, cut_point, cut_angle, color_image_vis = inference_server.get_cut_action(banana_masks, camera_color_data)

                if requires_cut:
                    action = 'cut'
                else:
                    action = 'acquire'

                total_valid_count += 1
                if (action == 'cut' and action_label == 'cut') or (action == 'acquire' and action_label == 'acquire'):
                    correct_count += 1

                banana_mask = np.zeros_like(banana_masks[0])
                for i in range(len(banana_masks)):
                    banana_mask = cv2.bitwise_or(banana_mask, banana_masks[i])
                banana_mask = cv2.cvtColor(banana_mask, cv2.COLOR_GRAY2BGR)
                
                print("Writing at path: ", os.path.join(OUTPUT_DIR + '/' + action_label + '/', img))
                cv2.imwrite(os.path.join(OUTPUT_DIR + '/' + action_label + '/', img), np.hstack([annotated_image, banana_mask, color_image_vis]))

            prediction_file.write(img + ' Label: ' + action_label + ' Prediction: ' + action +'\n')

            print(f"--- Correct Count: {correct_count}, Total Count: {total_count}\n")

            # except Exception as e:
            #     print("Error:", e)
            #     continue

    # print("Correct Count:", correct_count)
    # print("Total Valid Count:", total_valid_count)
    # print("Total Count:", total_count)
    
    return correct_count, total_valid_count, total_count

if __name__ == '__main__':
    plate_stats = {}
    for plate_number in [1, 2, 3, 4, 6]:
        correct_count, total_valid_count, total_count = run_tests(plate_number)
        plate_stats[plate_number] = (correct_count, total_valid_count, total_count)
    
    accuracies = []
    for key in plate_stats.keys():
        correct_count, total_valid_count, total_count = plate_stats[key]
        print("Total count: ", total_count)
        print(f'Plate {key} Correct: {correct_count} Total Valid: {total_valid_count} Total: {total_count} Accuracy: {correct_count/total_count}')
        accuracies.append(correct_count/total_count)
    print(f'Average Accuracy: {sum(accuracies)/len(accuracies)}')