import numpy as np
import math
import tf2_ros
import rospy
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose, PoseStamped
from geometry_msgs.msg import Pose, TransformStamped

def angle_between_pixels(source_px, target_px, image_width, image_height, orientation_symmetry = True):
    def angle_between(p1, p2):
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))
    if orientation_symmetry and source_px[1] > target_px[1]:
        source_px, target_px = target_px, source_px
    source_px_cartesian = np.array([source_px[0], image_height-source_px[1]])
    target_px_cartesian = np.array([target_px[0], image_height-target_px[1]])
    angle = angle_between(np.array([-image_width,0]), source_px_cartesian-target_px_cartesian)
    robot_angle_offset = -90
    return angle + robot_angle_offset


def pixel2World(camera_info, image_x, image_y, depth_image, box_width = 2):

    # print("(image_y,image_x): ",image_y,image_x)
    # print("depth image: ", depth_image.shape[0], depth_image.shape[1])

    if image_y >= depth_image.shape[0] or image_x >= depth_image.shape[1]:
        return False, None

    depth = depth_image[image_y, image_x]

    if math.isnan(depth) or depth < 0.05 or depth > 1.0:

        depth = []
        for i in range(-box_width,box_width):
            for j in range(-box_width,box_width):
                if image_y+i >= depth_image.shape[0] or image_x+j >= depth_image.shape[1]:
                    return False, None
                pixel_depth = depth_image[image_y+i, image_x+j]
                if not (math.isnan(pixel_depth) or pixel_depth < 50 or pixel_depth > 1000):
                    depth += [pixel_depth]

        if len(depth) == 0:
            return False, None

        depth = np.mean(np.array(depth))

    depth = depth/1000.0 # Convert from mm to m

    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]  

    # Convert to world space
    world_x = (depth / fx) * (image_x - cx)
    world_y = (depth / fy) * (image_y - cy)
    world_z = depth

    return True, np.array([world_x, world_y, world_z])

def world2Pixel(camera_info, world_x, world_y, world_z):

    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]  

    image_x = world_x * (fx / world_z) + cx
    image_y = world_y * (fy / world_z) + cy

    return image_x, image_y

def validate_with_user(question):
    user_input = input(question + "(y/n): ")
    while user_input != "y" and user_input != "n":
        user_input = input(question + "(y/n): ")
    if user_input == "y":
        return True
    else:
        return False

class TFUtils:
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer() # Using default cache time of 10 secs
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.control_rate = rospy.Rate(100)
    
    def getTransformationFromTF(self, source_frame, target_frame):

        while not rospy.is_shutdown():
            try:
                # print(f"Looking for transform from {source_frame} to {target_frame} using tfBuffer.lookup_transform...")
                transform = self.tfBuffer.lookup_transform(source_frame, target_frame, rospy.Time())
                # print("Got transform!")
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.control_rate.sleep()
                continue

        T = np.zeros((4,4))
        T[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
        T[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
        T[3,3] = 1

        print("Translation: ", T[:3,3])
        print("Rotation in quaternion: ", transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w)
        print("Rotation in euler: ", Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_euler('xyz', degrees=True))

        return T
    
    def publishTransformationToTF(self, source_frame, target_frame, transform):

        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = source_frame
        t.child_frame_id = target_frame

        t.transform.translation.x = transform[0][3]
        t.transform.translation.y = transform[1][3]
        t.transform.translation.z = transform[2][3]

        R = Rotation.from_matrix(transform[:3,:3]).as_quat()
        t.transform.rotation.x = R[0]
        t.transform.rotation.y = R[1]
        t.transform.rotation.z = R[2]
        t.transform.rotation.w = R[3]

        self.broadcaster.sendTransform(t)

    def get_pose_msg_from_transform(self, transform):

        pose = Pose()
        pose.position.x = transform[0,3]
        pose.position.y = transform[1,3]
        pose.position.z = transform[2,3]

        quat = Rotation.from_matrix(transform[:3,:3]).as_quat()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        return pose
