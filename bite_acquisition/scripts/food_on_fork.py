import os
from openai import OpenAI
import ast
import sys
import requests
import base64
import cv2
import rospy

import time
from rs_ros import RealSenseROS

class FoodOnFork:
    def __init__(self, api_key):

        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
            }

    def encode_image(self, openCV_image):
        retval, buffer = cv2.imencode('.jpg', openCV_image)
        return base64.b64encode(buffer).decode('utf-8')

    def prompt_zero_shot(self, image, prompt):
        # Getting the base64 string
        base64_image = self.encode_image(image)

        payload = {
        "model": "gpt-4o",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        response_text =  response.json()['choices'][0]["message"]["content"]
        return response_text
    
    def predict(self, image, log_path=None):
        image = image[:300, 600:900, :]
        print("Image shape: ", image.shape)
        prompt = "Is there a food item skewered on the fork tip? Do not consider food items that are on the plate in the background, we only want to check if a food item has been skewered on the fork tines. Answer Yes or No with additional explanation."
        response = self.prompt_zero_shot(image, prompt).strip()
        print("Response: ", response)

        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if log_path is not None:
            # Write on image
            cv2.putText(image, response, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(log_path, image)
        if 'Yes' in response:
            print("Food on fork: True")
            return True
        else:
            print("Food on fork: False")
            return False
        
if __name__ == "__main__":
    rospy.init_node('TestFoodOnFork')

    api_key = os.environ['OPENAI_API_KEY']
    food_on_fork = FoodOnFork(api_key)
    
    camera = RealSenseROS()
    time.sleep(1)
    _, image, _, _  = camera.get_camera_data()

    food_on_fork.predict(image)