import rospy
import robots
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from threading import Thread, Lock

from sensor_msgs.msg import JointState

from .base import RobotController

# positional interpolation
def get_waypoint(start_pt, target_pt, max_delta=0.005):
    total_delta = (target_pt - start_pt)
    num_steps = (np.linalg.norm(total_delta) // max_delta) + 1
    remainder = (np.linalg.norm(total_delta) % max_delta)
    if (remainder > 1e-3):
        num_steps += 1
    delta = total_delta / num_steps
    def gen_waypoint(i):
        return start_pt + delta * min(i, num_steps)
    return gen_waypoint, int(num_steps)

# rotation interpolation
def get_ori(initial_euler, final_euler, num_steps):
    diff = np.linalg.norm(final_euler - initial_euler)
    ori_chg = R.from_euler("xyz", [initial_euler.copy(), final_euler.copy()], degrees=False)
    if diff < 0.02:
        def gen_ori(i):
            return initial_euler
    else:
        slerp = Slerp([1, num_steps], ori_chg)
        def gen_ori(i): 
            interp_euler = slerp(i).as_euler("xyz")
            return interp_euler
    return gen_ori

class FrankaRobotController(RobotController):
    def __init__(self, config):

        self.ABOVE_PLATE_POSE = np.eye(4)
        self.ABOVE_PLATE_POSE[:3, 3] = np.array([0.30496958, -0.00216635, 0.67])
        self.ABOVE_PLATE_POSE[:3, :3] = R.from_euler("xyz", [np.pi, 0, -np.pi/4]).as_matrix()

        self.TRANSFER_POSE = np.eye(4)
        self.TRANSFER_POSE[:3, 3] = np.array([0.3533, -0.2347,  0.5243])
        self.TRANSFER_POSE[:3, :3] = R.from_quat([-0.3029,  0.6604, -0.6436, -0.2408]).as_matrix()
        
        self.env = robots.RobotEnv(**config)
        _, _ = self.env.reset(reset_controller=True)

        self.joint_state_publisher = rospy.Publisher('/robot_joint_states', JointState, queue_size=10)

        self.joint_state_lock = Lock()
        self.joint_state_values = None

        # run joint state publisher in a separate thread
        joint_state_thread = Thread(target=self.publish_joint_states)
        joint_state_thread.start()

        self.reset()


    def reset(self):
        self.move_to_acq_pose()

    def move_to_pose(self, pose):

        target_pos = pose[:3, 3].reshape(3)
        target_euler = R.from_matrix(pose[:3,:3]).as_euler("xyz")
        print("target_pos: ", target_pos)
        print("target_euler: ", target_euler)

        obs = self.env._get_obs()
        ee_pos = obs['state']['ee_pos']
        ee_quat = obs['state']['ee_quat']
        ee_euler = R.from_quat(ee_quat).as_euler("xyz")
        gripper_width = obs['state']['gripper_pos']
    
        positional_delta = np.linalg.norm(target_pos - ee_pos)
        rotational_delta = np.linalg.norm(target_euler - ee_euler)

        gen_waypoint, num_steps = get_waypoint(ee_pos, target_pos, max_delta=0.005)
        gen_ori = get_ori(ee_euler, target_euler, num_steps)
        for i in range(1, num_steps+1):
            next_ee_pos = gen_waypoint(i)
            next_ee_euler = gen_ori(i)
            action = np.hstack((next_ee_pos, next_ee_euler, gripper_width))
            self.env.step(action)
            self.update_joint_states(self.env._get_obs()['state']['joint_pos'])
            # return

    def move_to_acq_pose(self):
        print('Moving to acq pose')
        self.move_to_pose(self.ABOVE_PLATE_POSE)

    def move_to_transfer_pose(self):
        print('Moving to transfer pose')
        self.move_to_pose(self.TRANSFER_POSE)

    def get_current_pose(self):
        obs = self.env._get_obs()
        ee_pos = obs['state']['ee_pos']
        ee_quat = obs['state']['ee_quat']
        return ee_pos, ee_quat

    def publish_joint_states(self):

        while True:
            joint_state_value = None
            self.joint_state_lock.acquire()
            if self.joint_state_values is not None:
                joint_state_value = self.joint_state_values
            self.joint_state_lock.release()

            if joint_state_value is not None:
                joint_state_msg = JointState()
                joint_state_msg.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']

                joint_positions = joint_state_value.tolist()
                joint_positions.append(0.0) # gripper joint position
                joint_positions.append(0.0) # gripper joint position

                # Populate joint state message
                joint_state_msg.header.stamp = rospy.Time.now()
                joint_state_msg.position = joint_positions

                # Publish joint state message
                self.joint_state_publisher.publish(joint_state_msg)
            rospy.sleep(0.1)

            # exit if ctrl+c is pressed
            if rospy.is_shutdown():
                break

    def update_joint_states(self, joint_positions):
        self.joint_state_lock.acquire()
        self.joint_state_values = joint_positions
        self.joint_state_lock.release()
        