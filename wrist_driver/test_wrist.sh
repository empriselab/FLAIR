source /opt/ros/noetic/setup.sh
source /home/rkjenamani/flair_ws/devel/setup.sh
rostopic pub -1 /cmd_wrist_joint_angles wrist_driver_interfaces/SimpleJointAngleCommand '{q0: 0.0, q1: 0.0}'