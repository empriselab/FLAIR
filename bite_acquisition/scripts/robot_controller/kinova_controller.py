import rospy

from bite_acquisition.srv import PoseCommand, PoseCommandRequest, PoseCommandResponse
from bite_acquisition.srv import JointCommand, JointCommandRequest, JointCommandResponse

from .base import RobotController

class KinovaRobotController(RobotController):
    def __init__(self):
        # Do nothing
        self.DEFAULT_FORCE_THRESHOLD = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
        self.acq_pos = [0.02961542490849557, 0.06645885626898033, 5.281153370174079, 3.1258331315231405, 2.04992230955969, 4.737864779372388]
        self.transfer_pos = [0.3333664491938215, 1.4858324332736625, 1.5856359930210362, 0.8180422599332581, 1.5794872866962613, 4.604932028296647]

    def reset(self):
        self.move_to_acq_pose()

    def move_to_pose(self, pose):
        print("Calling set_pose with pose: ", pose)
        rospy.wait_for_service('set_pose')
        try:
            move_to_pose = rospy.ServiceProxy('set_pose', PoseCommand)
            resp1 = move_to_pose(pose, self.DEFAULT_FORCE_THRESHOLD)
            return resp1.success
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def set_joint_position(self, joint_position, mode = "POSITION"):
        print("Calling set_joint_positions with joint_position: ", joint_position)
        rospy.wait_for_service('set_joint_position')
        try:
            move_to_joint_position = rospy.ServiceProxy('set_joint_position', JointCommand)
            resp1 = move_to_joint_position(mode, joint_position, 100.0)
            return resp1.success
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def move_to_acq_pose(self):
        print('Moving to acq pose')
        self.set_joint_position(self.acq_pos)

    def move_to_transfer_pose(self):
        print('Moving to transfer pose')
        self.set_joint_position(self.transfer_pos)

if __name__ == '__main__':
    rospy.init_node('robot_controller', anonymous=True)
    robot_controller = KinovaRobotController()

    input('Press enter to move to acquisition position...')
    robot_controller.move_to_acq_pose()

    input('Press enter to move to transfer position...')
    robot_controller.move_to_transfer_pose()

    input('Press enter to reset the robot...')
    robot_controller.reset()
        