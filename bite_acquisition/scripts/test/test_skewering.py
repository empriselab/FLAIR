import sys
sys.path.insert(0, "..")

import cv2

from rs_ros import RealSenseROS
from skill_library import SkillLibrary
from inference_class import BiteAcquisitionInference
import rospy

if __name__ == '__main__':
    rospy.init_node('test_skewering', anonymous=True)
    
    camera = RealSenseROS()
    skill_library = SkillLibrary()
    inference_server = BiteAcquisitionInference(mode='ours')

    skill_library.reset()
    camera_header, camera_color_data, camera_info_data, camera_depth_data = camera.get_camera_data()

    # items = inference_server.recognize_items(camera_color_data)
    items = ['square orange cantaloupe']
    
    inference_server.FOOD_CLASSES = items
    annotated_image, detections, item_masks, item_portions, item_labels = inference_server.detect_items(camera_color_data)

    # visualize annotated image
    cv2.imshow('annotated_image', annotated_image)
    print("Visualizing annotated image in pop-up window, open that window and press any key to continue...")
    cv2.waitKey(0)

    # execute skewering skill on the first item
    mask = item_masks[0]
    skewer_point, skewer_angle = inference_server.get_skewer_action(mask)
    skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = skewer_point, major_axis = skewer_angle)

