import rospy
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R

from bite_acquisition.srv import PoseCommand, PoseCommandRequest, PoseCommandResponse
from bite_acquisition.srv import JointCommand, JointCommandRequest, JointCommandResponse

from .base import RobotController

class KinovaRobotController(RobotController):
    def __init__(self):
        # Do nothing
        self.DEFAULT_FORCE_THRESHOLD = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]

        self.robot_type = rospy.get_param('/robot_type')

        if self.robot_type == 'kinova_6dof':
            self.acq_pos = [0.02961542490849557, 0.06645885626898033, 5.281153370174079, 3.1258331315231405, 2.04992230955969, 4.737864779372388]
            self.transfer_pos = [0.3333664491938215, 1.4858324332736625, 1.5856359930210362, 0.8180422599332581, 1.5794872866962613, 4.604932028296647]
        elif self.robot_type == 'kinova_7dof':
            self.acq_pos = [0.00420759322479749, 6.1546478371669116, 3.1930894006403285, 4.395995056555025, 0.004538526880496712, 4.969687093309387, 1.6217796163872644]
            self.transfer_pos = [5.712775616468747, 0.918896236993586, 2.5843991661045806, 4.561511849883995, 5.430918897235775, 4.673872214171092, 3.9507097286038144]

    def reset(self):
        self.move_to_acq_pose()

    def move_to_pose(self, pose):

        target_pose = Pose()
        target_pose.position.x = pose[0,3]
        target_pose.position.y = pose[1,3]
        target_pose.position.z = pose[2,3]

        quat = R.from_matrix(pose[:3,:3]).as_quat()
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]

        # print("Calling set_pose with pose: ", target_pose)
        rospy.wait_for_service('set_pose')
        try:
            move_to_pose = rospy.ServiceProxy('set_pose', PoseCommand)
            resp1 = move_to_pose(target_pose, self.DEFAULT_FORCE_THRESHOLD)
            return resp1.success
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def set_joint_position(self, joint_position, mode = "POSITION"):
        # print("Calling set_joint_positions with joint_position: ", joint_position)
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
        