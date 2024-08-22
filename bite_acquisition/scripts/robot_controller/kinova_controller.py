import rospy

from bite_acquisition.srv import PoseCommand, PoseCommandRequest, PoseCommandResponse
from bite_acquisition.srv import JointCommand, JointCommandRequest, JointCommandResponse
from bite_acquisition.srv import JointWaypointsCommand, JointWaypointsCommandRequest, JointWaypointsCommandResponse
from bite_acquisition.srv import GripperCommand, GripperCommandRequest, GripperCommandResponse

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .base import RobotController

class KinovaRobotController(RobotController):
    def __init__(self):
        # Do nothing
        self.DEFAULT_FORCE_THRESHOLD = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
        self.acq_pos = [4.119619921793763, 5.927367810785151, 4.797271913808785, 4.641709217686205, 4.980350922946283, 5.268199221999715, 4.814377930122582]
        self.transfer_pos = [3.9634926355200855, 5.7086929905176556, 4.912630464851094, 4.31408101511415, 4.877527871154977, 5.429743910562832, 3.8112093559638285]

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

    def set_joint_waypoints(self, joint_waypoints):
        print("Calling set_joint_waypoints with joint_waypoints")
        rospy.wait_for_service('set_joint_waypoints')
        try:
            move_to_joint_waypoints = rospy.ServiceProxy('set_joint_waypoints', JointWaypointsCommand)

            target_waypoints = JointTrajectory()
            target_waypoints.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
            
            for waypoint in joint_waypoints:
                point = JointTrajectoryPoint()
                point.positions = waypoint
                target_waypoints.points.append(point)
            
            resp1 = move_to_joint_waypoints(target_waypoints, 100.0)
            return resp1.success
        
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def set_gripper(self, gripper_target):
        print("Calling set_gripper with gripper_target: ", gripper_target)
        rospy.wait_for_service('set_joint_waypoints')
        try:
            set_gripper = rospy.ServiceProxy('set_gripper', GripperCommand)
            resp1 = set_gripper(gripper_target)
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

    # input('Press enter to move to acquisition position...')
    # robot_controller.move_to_acq_pose()

    # input('Press enter to move to transfer position...')
    # robot_controller.move_to_transfer_pose()
    

    robot_controller.set_gripper(0.0)
    # robot_controller.set_joint_position([6.26643082812968, 5.964520505888411, 3.226885713821761, 4.113400641700101, 0.44228980435708964, 6.056389443484003, 1.5805738564210134])

    # input('Press enter to reset the robot...')
    # robot_controller.reset()
        