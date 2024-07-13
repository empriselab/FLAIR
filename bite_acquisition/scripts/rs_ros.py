import time
from threading import Lock

# ros imports
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import rospy

class RealSenseROS:
    def __init__(self):
        self.bridge = CvBridge()

        self.camera_lock = Lock()
        self.camera_header = None
        self.camera_color_data = None
        self.camera_info_data = None
        self.camera_depth_data = None

        queue_size = 1000

        self.color_image_sub = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size= queue_size, buff_size = 65536*queue_size)
        self.camera_info_sub = message_filters.Subscriber("/camera/color/camera_info", CameraInfo, queue_size= queue_size, buff_size = 65536*queue_size)
        self.depth_image_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, queue_size= queue_size, buff_size = 65536*queue_size)
        ts_top = message_filters.TimeSynchronizer([self.color_image_sub, self.camera_info_sub, self.depth_image_sub], queue_size= queue_size)
        ts_top.registerCallback(self.rgbdCallback)
        ts_top.enable_reset = True

        time.sleep(1.0)

    def rgbdCallback(self, rgb_image_msg, camera_info_msg, depth_image_msg):

        try:
            # Convert your ROS Image message to OpenCV2
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "32FC1")
        except CvBridgeError as e:
            print(e)

        with self.camera_lock:
            self.camera_header = rgb_image_msg.header
            self.camera_color_data = rgb_image
            self.camera_info_data = camera_info_msg
            self.camera_depth_data = depth_image

    def get_camera_data(self):
        with self.camera_lock:
            return self.camera_header, self.camera_color_data, self.camera_info_data, self.camera_depth_data
        

if __name__ == "__main__":
    rospy.init_node('RealSenseROS')
    rs_ros = RealSenseROS()

    # save data to file
    import os
    import cv2
    file = '/home/rkjenamani/bite_ws/src/bite_acquisition/scripts/img_data/'
    
    while True:    
        input("Press [ENTER] to save image ...")

        header, color_data, info_data, depth_data = rs_ros.get_camera_data()    
        num_files = len([name for name in os.listdir(file)])
        cv2.imwrite(file + str(num_files) + "_camera_color_data.jpg", color_data)
        cv2.imwrite(file + str(num_files) + "_camera_depth_data.jpg", depth_data)
        print("Saved image to file")

        print("Header:",header)
        print("Color Data:",color_data[0:10])
        print("Info Data:",info_data)
        print("Depth Data:",depth_data[0:10])
