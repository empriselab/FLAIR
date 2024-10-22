import rospy
import numpy as np
import math

from sensor_msgs.msg import JointState

from wrist_driver_interfaces.msg import SimpleJointAngleCommand
from wrist_driver_interfaces.srv import SetWristMode, SetWristModeRequest, SetWristModeResponse

class WristController:
    def __init__(self):
        self.wrist_state_pub = rospy.Publisher('/cmd_wrist_joint_angles', SimpleJointAngleCommand, queue_size=10)

        # set wrist control mode to velocity
        rospy.wait_for_service('set_wrist_mode')
        try:
            set_wrist_mode = rospy.ServiceProxy('set_wrist_mode', SetWristMode)
            resp1 = set_wrist_mode(1)
            if resp1.success:
                print("Successfully set wrist mode to velocity")
            else:
                print("Failed to set wrist mode to velocity")
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def reset(self):
        wrist_joint_states = rospy.wait_for_message('/wrist_joint_states', JointState)
        current_pitch = wrist_joint_states.position[0]
        current_roll = wrist_joint_states.position[1]

        self.set_wrist_state(0,0)

    def set_wrist_state(self, pitch, roll, vel=4):

        desired_pitch = pitch
        desired_roll = roll

        wrist_joint_states = rospy.wait_for_message('/wrist_joint_states', JointState)

        pitch_achieved = np.abs(wrist_joint_states.position[0] - desired_pitch) < 0.01
        roll_achieved = np.abs(wrist_joint_states.position[1] - desired_roll) < 0.01
        while (not pitch_achieved) or (not roll_achieved):
            wrist_joint_states = rospy.wait_for_message('/wrist_joint_states', JointState)
            current_pitch = wrist_joint_states.position[0]
            current_roll = wrist_joint_states.position[1]

            pitch_achieved = np.abs(current_pitch - desired_pitch) < 0.01
            roll_achieved = np.abs(current_roll - desired_roll) < 0.01
            
            wrist_state = SimpleJointAngleCommand()
            if not pitch_achieved:
                if np.abs(desired_pitch - current_pitch) > 0.2:
                    wrist_state.q0 = 0.8 * np.sign(desired_pitch - current_pitch)
                else:
                    #wrist_state.q0 = (desired_pitch - current_pitch) * vel
                    wrist_state.q0 = (desired_pitch - current_pitch) * 4
                    #wrist_state.q0 = (desired_pitch - current_pitch) * 8
            else:
                wrist_state.q0 = 0

            if not roll_achieved:
                if np.abs(desired_roll - current_roll) > 0.2:
                    wrist_state.q1 = 1.0 * np.sign(desired_roll - current_roll)
                else:
                    wrist_state.q1 = (desired_roll - current_roll) * vel
                    #wrist_state.q1 = (desired_roll - current_roll) * 8
            else:
                wrist_state.q1 = 0

            self.wrist_state_pub.publish(wrist_state)

            # print("q0: ", wrist_state.q0)
            # print("q1: ", wrist_state.q1)

        wrist_state = SimpleJointAngleCommand()
        wrist_state.q0 = 0
        wrist_state.q1 = 0
        self.wrist_state_pub.publish(wrist_state)

        print("Wrist state set to: ", desired_pitch, desired_roll)

    def scoop_wrist_hack(self):

        # three phases: initially fast, then medium, then PD control
        # you want the initial fast such that the wrist does not hit the plate
        
        # handle wrist disconnecting and other errors
        try:
            wrist_state = SimpleJointAngleCommand()
            wrist_state.q0 = 1.0
            wrist_state.q1 = 0
            self.wrist_state_pub.publish(wrist_state)

            desired_pitch = 0.4 * math.pi
            wrist_joint_states = rospy.wait_for_message('/wrist_joint_states', JointState)
            pitch_achieved = np.abs(wrist_joint_states.position[0] - desired_pitch) < 0.05
            while not pitch_achieved:
                wrist_joint_states = rospy.wait_for_message('/wrist_joint_states', JointState)
                current_pitch = wrist_joint_states.position[0]
                pitch_achieved = np.abs(current_pitch - desired_pitch) < 0.05
                wrist_state = SimpleJointAngleCommand()
                if not pitch_achieved:
                    if np.abs(desired_pitch - current_pitch) > 0.35 * math.pi:
                        wrist_state.q0 = 1.0 * np.sign(desired_pitch - current_pitch)
                    else:
                        wrist_state.q0 = min((desired_pitch - current_pitch) * 1.0 / (0.35 * math.pi), 0.4 * np.sign(desired_pitch - current_pitch))
                else:
                    wrist_state.q0 = 0

                wrist_state.q1 = 0

                self.wrist_state_pub.publish(wrist_state)

                # print("current_pitch: ", current_pitch)
                # print("desired_pitch: ", desired_pitch)
                # print("q0: ", wrist_state.q0)
                # print("q1: ", wrist_state.q1)
                # print("error: ", np.abs(current_pitch - desired_pitch))

            wrist_state = SimpleJointAngleCommand()
            wrist_state.q0 = 0
            wrist_state.q1 = 0
            self.wrist_state_pub.publish(wrist_state)

        except:
            print("Wrist Disconnected during scooping pickup. Unlucky, restart experiment. :(")
            pass

    def scoop_wrist(self):

        wrist_state = SimpleJointAngleCommand()
        wrist_state.q0 = 1.35
        wrist_state.q1 = 0
        self.wrist_state_pub.publish(wrist_state)

        desired_pitch = 0.4 * math.pi
        wrist_joint_states = rospy.wait_for_message('/wrist_joint_states', JointState)
        pitch_achieved = np.abs(wrist_joint_states.position[0] - desired_pitch) < 0.05
        while not pitch_achieved:
            wrist_joint_states = rospy.wait_for_message('/wrist_joint_states', JointState)
            current_pitch = wrist_joint_states.position[0]
            pitch_achieved = np.abs(current_pitch - desired_pitch) < 0.05
            wrist_state = SimpleJointAngleCommand()
            if not pitch_achieved:
                if np.abs(desired_pitch - current_pitch) > 0.1:
                    wrist_state.q0 = 1.35 * np.sign(desired_pitch - current_pitch)
                else:
                    wrist_state.q0 = (desired_pitch - current_pitch) * 5
            else:
                wrist_state.q0 = 0

            wrist_state.q1 = 0

            self.wrist_state_pub.publish(wrist_state)

            # print("q0: ", wrist_state.q0)
            # print("q1: ", wrist_state.q1)
            # print("error: ", np.abs(current_pitch - desired_pitch))

        wrist_state = SimpleJointAngleCommand()
        wrist_state.q0 = 0
        wrist_state.q1 = 0
        self.wrist_state_pub.publish(wrist_state)

    def twirl_wrist(self, vel = 4):

        # get current wrist state
        wrist_joint_states = rospy.wait_for_message('/wrist_joint_states', JointState)
        current_pitch = -wrist_joint_states.position[0]
        current_roll = -wrist_joint_states.position[1]

        desired_roll = current_roll - 4 * math.pi
        self.set_wrist_state(current_pitch, desired_roll, vel=vel)

    def set_to_scoop_pos(self):
        wrist_joint_states = rospy.wait_for_message('/wrist_joint_states', JointState)
        current_pitch = -wrist_joint_states.position[0]
        current_roll = -wrist_joint_states.position[1]

        # find nearest multiple of math.pi *2 for current_pitch and current_roll
        target_roll = round(current_roll / (math.pi * 2)) * math.pi * 2
        self.set_wrist_state(1.0,target_roll)

    def set_to_dip_pos(self):
        wrist_joint_states = rospy.wait_for_message('/wrist_joint_states', JointState)
        current_pitch = -wrist_joint_states.position[0]
        current_roll = -wrist_joint_states.position[1]

        # find nearest multiple of math.pi *2 for current_pitch and current_roll
        target_roll = round(current_roll / (math.pi * 2)) * math.pi * 2
        self.set_wrist_state(0.4 * math.pi, target_roll)

    def set_to_cut_pos(self):
        wrist_joint_states = rospy.wait_for_message('/wrist_joint_states', JointState)
        current_pitch = -wrist_joint_states.position[0]
        current_roll = -wrist_joint_states.position[1]

        # find nearest multiple of math.pi *2 for current_pitch and current_roll
        target_roll = round(current_roll / (math.pi * 2)) * math.pi * 2 + math.pi/2
        self.set_wrist_state(0.4 * math.pi,target_roll)

    def scooping_scoop(self):
        wrist_joint_states = rospy.wait_for_message('/wrist_joint_states', JointState)
        current_pitch = -wrist_joint_states.position[0]
        current_roll = -wrist_joint_states.position[1]
        self.set_wrist_state(0.4 * math.pi, current_roll)
    
    def cutting_tilt(self):
        wrist_joint_states = rospy.wait_for_message('/wrist_joint_states', JointState)
        current_pitch = -wrist_joint_states.position[0]
        current_roll = -wrist_joint_states.position[1]

        self.set_wrist_state(current_pitch, current_roll + math.pi/8)
    

   