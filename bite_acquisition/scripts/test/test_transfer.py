import sys
sys.path.insert(0, "..")

from rs_ros import RealSenseROS
from skill_library import SkillLibrary
import rospy

if __name__ == '__main__':
    rospy.init_node('test_skills_pixel_selection', anonymous=True)

    inp = input("Is head_perception.py running? (y/n): ")
    while inp != 'y':
        print("This script requires head_perception.py to be running.")
        inp = input("Is head_perception.py running? (y/n): ")
    
    skill_library = SkillLibrary()
    camera = RealSenseROS()

    skill_library.reset()
    
    input("Press ENTER to execute transfer to mouth skill")
    skill_library.transfer_to_mouth()
    
    skill_library.reset()

