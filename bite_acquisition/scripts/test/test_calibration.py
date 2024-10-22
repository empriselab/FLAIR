import sys
sys.path.insert(0, "..")
import rospy
import yaml
import cv2
import time
import math

from wrist_controller import WristController
import utils
from rs_ros import RealSenseROS

# import imageio

if __name__ == "__main__":

    rospy.init_node('test_calibration')

    robot_type = rospy.get_param('/robot_type')
    if robot_type == 'franka':
        from robot_controller.franka_controller import FrankaRobotController
        config_path = rospy.get_param('/robot_config')
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        robot_controller = FrankaRobotController(config)
    elif robot_type == 'kinova_6dof' or robot_type == 'kinova_7dof':
        from robot_controller.kinova_controller import KinovaRobotController
        robot_controller = KinovaRobotController()
    wrist_controller = WristController()
    camera = RealSenseROS()

    robot_controller.reset()
    wrist_controller.reset()

    print("Visualizing the calibration at various wrist (utensil) states...")

    wrist_states = [0.4 * math.pi, 0.36 * math.pi, 0.32 * math.pi, 0.28 * math.pi, 0.24 * math.pi, 0.2 * math.pi, 0.16 * math.pi, 0.12 * math.pi, 0.08 * math.pi, 0.04 * math.pi, 0.0]
    wrist_states.reverse()

    # go through all wrist states in cyclic order
    index = 0
    positive = True
    images = []
    while True:
    # for i in range(10):
        wrist_controller.set_wrist_state(wrist_states[index],0)
        index = index + 1 if positive else index - 1
        if index == len(wrist_states):
            index = len(wrist_states) - 2
            positive = False
        elif index == -1:
            index = 1
            positive = True

        print("Sleeping for 1 second...")
        time.sleep(1.0)

        header, color_data, info_data, depth_data = camera.get_camera_data()
        
        camera_to_fork = utils.TFUtils().getTransformationFromTF("camera_color_optical_frame", "fork_tip")

        # Hack
        curr_translation = camera_to_fork[:,3]
        camera_to_fork[:,3] = curr_translation

        fork_x, fork_y = utils.world2Pixel(info_data, camera_to_fork[0,3], camera_to_fork[1,3], camera_to_fork[2,3])
        print("Fork pixel:",fork_x, fork_y)

        cv2.circle(color_data, (int(fork_x), int(fork_y)), 10, (0,255,255), -1)
        
        cv2.imshow("Image", color_data)
        cv2.waitKey(1)

        # images.append(color_data)

    # images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    # imageio.mimsave('~/Desktop/calib.gif', images, duration=400)

