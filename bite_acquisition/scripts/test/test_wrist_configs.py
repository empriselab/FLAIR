import sys
sys.path.insert(0, "..")
from wrist_controller import WristController
import rospy

if __name__ == "__main__":

    rospy.init_node('test_wrist_configs', anonymous=True)
    wrist_controller = WristController()

    input("Press ENTER to test moving to ready to scoop configuration")
    wrist_controller.reset()
    wrist_controller.set_to_scoop_pos()

    input("Press ENTER to test moving to cutting configuration")
    wrist_controller.reset()
    wrist_controller.set_to_cut_pos()

    input("Press ENTER to test twirling wrist")
    wrist_controller.reset()
    wrist_controller.twirl_wrist(vel=8)

    input("Press ENTER to test scooping wrist")
    wrist_controller.reset()
    wrist_controller.scoop_wrist()

    input("Press ENTER to test scooping wrist hack (fast then slow)")
    wrist_controller.reset()
    wrist_controller.scoop_wrist_hack()

    input("Press ENTER to reset wrist")
    wrist_controller.reset()