import cv2
import numpy as np
import math
import os

import base64
import requests
import json
import time
import os

from gpt_prompt.prompt import GPTPredictor

BASE_DIR = '/home/isacc/bite_acquisition'
PARAM_TUNING = True # tune gpt prompt using training data - in-context examples

if __name__ == '__main__':

    if PARAM_TUNING:
        INPUT_DIR = BASE_DIR + '/task_planning_tests/noodle_plates/log/spaghetti/classification_format/train'
        OUTPUT_DIR = BASE_DIR + '/task_planning_tests/noodle_plates/param_tuning/gpt/spaghetti'
        PREDICTION_DIR = OUTPUT_DIR + '/predictions.txt'
    else:
        INPUT_DIR = BASE_DIR + '/task_planning_tests/noodle_plates/log/spaghetti/classification_format/test'
        OUTPUT_DIR = BASE_DIR + '/task_planning_tests/noodle_plates/outputs/gpt/spaghetti'
        PREDICTION_DIR = OUTPUT_DIR + '/predictions.txt'

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    predictor = GPTPredictor()
    CANDIDATE_ACTIONS = ['twirl', 'group']

    correct_count = 0
    total_valid_count = 0
    total_count = 0

    prediction_file = open(PREDICTION_DIR, 'w')
    for action_label in CANDIDATE_ACTIONS:
        if not os.path.exists(OUTPUT_DIR + '/' + action_label):
            os.makedirs(OUTPUT_DIR + '/' + action_label)
        TEST_IMGS = os.listdir(os.path.join(INPUT_DIR, action_label))
        # sort images by number
        TEST_IMGS.sort(key=lambda x: int(x.split('.')[0]))
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
            # time.sleep(3)

    print("Correct Count:", correct_count)
    print("Total Valid Count:", total_valid_count)
    print("Total Count:", total_count)