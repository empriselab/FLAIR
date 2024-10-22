from rs_ros import RealSenseROS
import utils
import numpy as np
import cv2

import face_alignment

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Bool

if __name__ == '__main__':

    rospy.init_node('head_perception')

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    mouth_center_publisher = rospy.Publisher('/mouth_center', Point, queue_size=10)
    mouth_open_publisher = rospy.Publisher('/mouth_open', Bool, queue_size=10)

    camera = RealSenseROS()

    while not rospy.is_shutdown():
        camera_header, camera_color_data, camera_info_data, camera_depth_data = camera.get_camera_data()

        if camera_color_data is None or camera_info_data is None or camera_depth_data is None:
            print("No data received")
            continue

        preds = fa.get_landmarks(camera_color_data)
        if preds is None:
            print("No face detected")
            continue

        preds = preds[0]
        print("Predicitions:", preds)
        print("Number of landmarks:", preds.shape)

        preds_3d = []

        # visualize the landmarks
        for pred in preds[48:68]:
            validity, pred_3d = utils.pixel2World(camera_info_data, int(pred[0]), int(pred[1]), camera_depth_data, box_width=5)
            if validity:
                preds_3d.append(pred_3d)
                cv2.circle(camera_color_data, (int(pred[0]), int(pred[1])), 2, (0, 255, 0), -1)

        # visualize the landmarks
        # cv2.imshow("Landmarks", camera_color_data)
        # cv2.waitKey(1)
                
        if len(preds_3d) == 0:
            print("No valid 3D landmarks")
            continue

        preds_3d = np.array(preds_3d)
        mouth_center_3d = np.mean(preds_3d, axis=0)

        mouth_center_msg = Point()
        mouth_center_msg.x = mouth_center_3d[0]
        mouth_center_msg.y = mouth_center_3d[1]
        mouth_center_msg.z = mouth_center_3d[2]

        mouth_center_publisher.publish(mouth_center_msg)

        lipDist = np.sqrt( (preds[66][0] - preds[62][0]) ** 2 + (preds[66][1] - preds[62][1]) ** 2)

        lipThickness = float(np.sqrt( (preds[51][0] - preds[62][0]) ** 2 + (preds[51][1] - preds[62][1]) ** 2)/2) + \
            float(np.sqrt( (preds[57][0] - preds[66][0]) ** 2 + (preds[57][1] - preds[66][1]) ** 2)/2)

        if lipDist >= 1.5 * lipThickness:
            mouth_open_publisher.publish(True)
        else:
            mouth_open_publisher.publish(False)

        # add signal handler to stop the loop
        if cv2.waitKey(1) & 0xFF == ord('q'): # press q to break
            break