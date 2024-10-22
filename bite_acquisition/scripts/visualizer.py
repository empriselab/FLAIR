import rospy
from visualization_msgs.msg import Marker, MarkerArray
import utils

class Visualizer:
    def __init__(self):
        self.tf_utils = utils.TFUtils()

        self.utensil_visualization_pub = rospy.Publisher('utensil_visualization_marker_array', MarkerArray, queue_size=10)
        self.food_visualization_pub = rospy.Publisher('food_visualization_marker_array', MarkerArray, queue_size=10)

    def visualize_fork(self, transform):
        print("Visualizing fork")
        # visualize fork mesh in rviz
        marker_array = MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "base_link"
        marker.type = marker.MESH_RESOURCE
        marker.action = marker.ADD
        marker.scale.x = 0.001
        marker.scale.y = 0.001
        marker.scale.z = 0.001
        marker.color.a = 1.0
        
        # marker color is grey
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        
        pose = self.tf_utils.get_pose_msg_from_transform(transform)
        marker.pose = pose

        marker.mesh_resource = "package://kortex_description/tools/feeding_utensil/fork_tip.stl"
        marker_array.markers.append(marker)

        self.utensil_visualization_pub.publish(marker_array)

    def visualize_food(self, transform, id = 0):

        # publish a cube marker
        marker_array = MarkerArray()
        marker = Marker()
        marker.id = id
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "base_link"
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.a = 1.0

        # marker color is red
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        pose = self.tf_utils.get_pose_msg_from_transform(transform)
        marker.pose = pose

        marker_array.markers.append(marker)

        self.food_visualization_pub.publish(marker_array)

    def clear_visualizations(self):
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = marker.DELETEALL
        marker_array.markers.append(marker)

        self.utensil_visualization_pub.publish(marker_array)
        self.food_visualization_pub.publish(marker_array)