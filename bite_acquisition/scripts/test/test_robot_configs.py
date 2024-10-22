import sys
sys.path.insert(0, "..")
import rospy
import yaml

if __name__ == '__main__':
    rospy.init_node('test_robot_configs', anonymous=True)

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

    input('Press ENTER to move to acquisition position...')
    robot_controller.move_to_acq_pose()

    input('Press ENTER to move to transfer position...')
    robot_controller.move_to_transfer_pose()

    input('Press ENTER to reset the robot...')
    robot_controller.reset()