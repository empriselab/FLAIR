import cv2
import numpy as np
import os

import base64
import requests
import json
import time

class GPTPredictor:
    def __init__(self):

        self.api_key =  os.environ.get('OPENAI_API_KEY')

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
            }
        
        self.home_dir = os.path.dirname(os.path.realpath(__file__))
        self.log_file = f"{self.home_dir}/log.txt"
        
        with open(f"{self.home_dir}/prompt.txt", 'r') as f:
            self.prompt_text = f.read()
        
        self.prompt_img1 = cv2.imread(f"{self.home_dir}/example_imgs/215_color.png")
        self.prompt_img2 = cv2.imread(f"{self.home_dir}/example_imgs/252_color.png")
        self.prompt_img3 = cv2.imread(f"{self.home_dir}/example_imgs/301_color.png")
        self.prompt_img4 = cv2.imread(f"{self.home_dir}/example_imgs/357_color.png")
        self.prompt_img5 = cv2.imread(f"{self.home_dir}/example_imgs/369_color.png")
        self.prompt_img6 = cv2.imread(f"{self.home_dir}/example_imgs/380_color.png")
        self.prompt_img7 = cv2.imread(f"{self.home_dir}/example_imgs/402_color.png")
        self.prompt_img8 = cv2.imread(f"{self.home_dir}/example_imgs/411_color.png")  
        self.prompt_img9 = cv2.imread(f"{self.home_dir}/example_imgs/424_color.png")
        self.prompt_img10 = cv2.imread(f"{self.home_dir}/example_imgs/427_color.png")

        self.prompt_img1 = self.encode_image(self.prompt_img1)
        self.prompt_img2 = self.encode_image(self.prompt_img2)
        self.prompt_img3 = self.encode_image(self.prompt_img3)
        self.prompt_img4 = self.encode_image(self.prompt_img4)
        self.prompt_img5 = self.encode_image(self.prompt_img5)
        self.prompt_img6 = self.encode_image(self.prompt_img6)
        self.prompt_img7 = self.encode_image(self.prompt_img7)
        self.prompt_img8 = self.encode_image(self.prompt_img8)
        self.prompt_img9 = self.encode_image(self.prompt_img9)
        self.prompt_img10 = self.encode_image(self.prompt_img10)

        self.correct_predictions = 0
        self.incorrect_predictions = 0

    def encode_image(self, openCV_image):
        retval, buffer = cv2.imencode('.jpg', openCV_image)
        return base64.b64encode(buffer).decode('utf-8')
        
    def prompt(self, image, image_id = None, image_annotation = None):

        image_copy = image.copy()
        base64_image = self.encode_image(image_copy)

        payload = {
        "model": "gpt-4o",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": self.prompt_text
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.prompt_img1}",
                    "detail": "high"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.prompt_img2}",
                    "detail": "high"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.prompt_img3}",
                    "detail": "high"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.prompt_img4}",
                    "detail": "high"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.prompt_img5}",
                    "detail": "high"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.prompt_img6}",
                    "detail": "high"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.prompt_img7}",
                    "detail": "high"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.prompt_img8}",
                    "detail": "high"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.prompt_img9}",
                    "detail": "high"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.prompt_img10}",
                    "detail": "high"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        print("Image ID: ", image_id)
        print("Response: ", response.json())
        print("Image Annotation: ", image_annotation)
        try:
            response_text =  response.json()["choices"][0]["message"]["content"]
        except KeyError:
            print("OPENAI API ERROR")
            # input("Press Enter to try again ...")
            time.sleep(10)
            return self.prompt(image, image_id, image_annotation)

        # log response to file
        with open(self.log_file, 'a') as f:
            f.write('----------------------------------------' + '\n')
            f.write("Image: " + str(image_id) + '\n')
            f.write("GPT4-Vision: " + response_text + '\n')
            f.write("Annotated Ground Truth: " + str(image_annotation) + '\n')
            f.write('\n')

        if response_text.startswith("Yes"):
            return "scoop"
        else:
            return "push"