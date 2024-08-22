import sys
sys.path.insert(0, "..")

import flair_utils
from skill_library import SkillLibrary
import rospy
import cv2
import numpy as np
import time
import math
from scipy.spatial.transform import Rotation


if __name__ == "__main__":

    rospy.init_node('test_speed')
    time.sleep(1.0)

    tf_utils = flair_utils.TFUtils()

    skill_library = SkillLibrary()
    skill_library.reset()

    force_threshold = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]

    input("Press ENTER to send Kinova to Pose 1 (Only Translation)")
    tool_base = tf_utils.getTransformationFromTF('base_link', 'tool_frame')

    # only translation
    tool_frame_displacement = np.eye(4)
    tool_frame_displacement[1,3] = 0.2
    tool_frame_displacement[2,3] = 0.2

    tool_base = tool_base @ tool_frame_displacement

    tf_utils.publishTransformationToTF('base_link', 'wrist_target', tool_base)
        
    print("Press enter to actually move utensil.")
    input()

    start_time = time.time()
    skill_library.set_pose(tf_utils.get_pose_msg_from_transform(tool_base), force_threshold)
    end_time = time.time()

    print("Time taken: ", end_time - start_time)

    input("Press ENTER to move back to home position")
    skill_library.reset()

    input("Press ENTER to send Kinova to Pose 2 (Translation + Rotation)")
    tool_base = tf_utils.getTransformationFromTF('base_link', 'tool_frame')

    # only translation
    tool_frame_displacement = np.eye(4)
    tool_frame_displacement[1,3] = 0.2
    tool_frame_displacement[2,3] = 0.2

    tool_base = tool_base @ tool_frame_displacement

    # only rotation
    tool_frame_displacement = np.eye(4)
    tool_frame_displacement[:3,:3] = Rotation.from_euler('xyz', [0,0,180], degrees=True).as_matrix()

    tool_base = tool_base @ tool_frame_displacement

    tf_utils.publishTransformationToTF('base_link', 'wrist_target', tool_base)
        
    print("Press enter to actually move utensil.")
    input()

    start_time = time.time()
    skill_library.set_pose(tf_utils.get_pose_msg_from_transform(tool_base), force_threshold)
    end_time = time.time()

    print("Time taken: ", end_time - start_time)

    input("Press ENTER to move back to home position")
    skill_library.reset()

    

    




    
