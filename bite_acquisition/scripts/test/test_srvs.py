# write a program for the scooping skill as a part of a skill library consisting of food acquisition skills

import rospy
from bite_acquisition.msg import CartesianState
from bite_acquisition.srv import PoseCommand, PoseCommandRequest, PoseCommandResponse
from bite_acquisition.srv import PoseWaypointsCommand, PoseWaypointsCommandRequest, PoseWaypointsCommandResponse
from bite_acquisition.srv import TwistCommand, TwistCommandRequest, TwistCommandResponse
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from bite_acquisition.srv import JointCommand, JointCommandRequest, JointCommandResponse
import math
import numpy as np

def set_pose(pose, force_threshold):
    print("Calling set_pose with pose: ", pose)
    rospy.wait_for_service('set_pose')
    try:
        move_to_pose = rospy.ServiceProxy('set_pose', PoseCommand)
        resp1 = move_to_pose(pose, force_threshold)
        return resp1.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def set_joint_position(joint_position):
    print("Calling set_joint_positions with joint_position: ", joint_position)
    rospy.wait_for_service('set_joint_position')
    try:
        move_to_joint_position = rospy.ServiceProxy('set_joint_position', JointCommand)
        resp1 = move_to_joint_position("POSITION", joint_position)
        return resp1.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def set_joint_velocity(joint_velocity, timeout):
    print("Calling set_joint_velocities with joint_velocity: ", joint_velocity)
    rospy.wait_for_service('set_joint_velocity')
    try:
        move_to_joint_velocity = rospy.ServiceProxy('set_joint_velocity', JointCommand)
        resp1 = move_to_joint_velocity("VELOCITY", joint_velocity, timeout)
        return resp1.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def get_wrist_state():
    wrist_state = rospy.wait_for_message('robot_cartesian_state', CartesianState)
    return wrist_state

def main():
    rospy.init_node('test_srvs')
    # joint_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -math.pi/3]
    timeout = 18.0
    # success = set_joint_velocity(joint_velocity, timeout)
    print(success)

if __name__ == "__main__":
    main()