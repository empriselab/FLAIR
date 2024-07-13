import cv2
import numpy as np
import math
import os

import base64
import requests
import json
import time
import os

from gpt_prompts.spaghetti_meatballs.prompt import GPTPredictor as SpaghettiPredictor
from gpt_prompts.fettuccine_chicken_broccoli.prompt import GPTPredictor as FettuccinePredictor
from gpt_prompts.mashed_potato_sausage.prompt import GPTPredictor as MashedPotatoPredictor
from gpt_prompts.oatmeal_strawberry.prompt import GPTPredictor as OatmealPredictor
from gpt_prompts.dessert.prompt import GPTPredictor as DessertPredictor

# PLATE_NUMBER = 2 # 1, 2, 3, 4, 6

def run_tests(plate_number):

    SOURCE_DIR = '/home/isacc/bite_acquisition/task_planning_tests/study_plates/'

    if plate_number == 1:
        PLATE_NAME = 'spaghetti_meatballs'
        CATEGORY = 'noodle'
        predictor = SpaghettiPredictor()
    elif plate_number == 2:
        PLATE_NAME = 'fettuccine_chicken_broccoli'
        CATEGORY = 'noodle'
        predictor = FettuccinePredictor()
    elif plate_number == 3:
        PLATE_NAME = 'mashed_potato_sausage'
        CATEGORY = 'semisolid'
        predictor = MashedPotatoPredictor()
    elif plate_number == 4:
        PLATE_NAME = 'oatmeal_strawberry'
        CATEGORY = 'semisolid'
        predictor = OatmealPredictor()
    elif plate_number == 6:
        PLATE_NAME = 'dessert'
        CATEGORY = 'cuttable'
        predictor = DessertPredictor()

    if CATEGORY == 'noodle':
        CANDIDATE_ACTIONS = ['twirl', 'group', 'push']
    elif CATEGORY == 'semisolid':
        CANDIDATE_ACTIONS = ['scoop', 'push']
    elif CATEGORY == 'cuttable':
        CANDIDATE_ACTIONS = ['cut', 'acquire']

    INPUT_DIR = SOURCE_DIR + 'log/' + PLATE_NAME + '/classification_format/test'
    OUTPUT_DIR = SOURCE_DIR + 'outputs/gpt/' + PLATE_NAME
    PREDICTION_DIR = OUTPUT_DIR + '/predictions.txt'
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

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
            SOURCE_IMAGE_PATH = os.path.join(INPUT_DIR, action_label, img)
            print(SOURCE_IMAGE_PATH)
            camera_color_data = cv2.imread(SOURCE_IMAGE_PATH)

            vis_image = camera_color_data.copy()

            total_count += 1
            try:     
                action = predictor.prompt(camera_color_data, image_id = img, image_annotation = action_label)
                
                total_valid_count += 1
                if action == action_label:
                    correct_count += 1  
                
                prediction_file.write(img + ' Label: ' + action_label + ' Prediction: ' + action + '\n')

                vis_img = cv2.putText(vis_image, action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(OUTPUT_DIR, action_label, img), vis_img)
                  
            except Exception as e:
                print("Error:", e)
                prediction_file.write(img + ' - ' + action_label + ' - ' + 'Error' + '\n')
                continue
                
            print(f"Correct Count: {correct_count}, Total Valid Count: {total_valid_count}, Total Count: {total_count}")

            # to not trigger the rate limit
            # time.sleep(5)

    print("Correct Count:", correct_count)
    print("Total Valid Count:", total_valid_count)
    print("Total Count:", total_count)

    return correct_count, total_valid_count, total_count

if __name__ == '__main__':
    plate_stats = {}
    for plate_number in [3]:
        correct_count, total_valid_count, total_count = run_tests(plate_number)
        plate_stats[plate_number] = (correct_count, total_valid_count, total_count)
    
    accuracies = []
    for key in plate_stats.keys():
        correct_count, total_valid_count, total_count = plate_stats[key]
        print(f'Plate {key} Correct: {correct_count} Total Valid: {total_valid_count} Total: {total_count} Accuracy: {correct_count/total_count}')
        accuracies.append(correct_count/total_count)
    print(f'Average Accuracy: {sum(accuracies)/len(accuracies)}')