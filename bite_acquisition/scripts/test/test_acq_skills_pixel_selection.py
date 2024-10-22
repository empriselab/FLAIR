import sys
sys.path.insert(0, "..")

from rs_ros import RealSenseROS
from skill_library import SkillLibrary
import rospy

if __name__ == '__main__':
    rospy.init_node('test_skills_pixel_selection', anonymous=True)
    
    skill_library = SkillLibrary()
    camera = RealSenseROS()
    
    input("Press ENTER to test skewering skill")
    skill_library.reset()
    camera_header, camera_color_data, camera_info_data, camera_depth_data = camera.get_camera_data()
    skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data)

    input("Press ENTER to test scooping skill")
    skill_library.reset()
    camera_header, camera_color_data, camera_info_data, camera_depth_data = camera.get_camera_data()
    skill_library.scooping_skill(camera_color_data, camera_depth_data, camera_info_data)

    input("Press ENTER to test dipping skill")
    skill_library.reset()
    camera_header, camera_color_data, camera_info_data, camera_depth_data = camera.get_camera_data()
    skill_library.dipping_skill(camera_color_data, camera_depth_data, camera_info_data)

    input("Press ENTER to test pushing skill")
    skill_library.reset()
    camera_header, camera_color_data, camera_info_data, camera_depth_data = camera.get_camera_data()
    skill_library.pushing_skill(camera_color_data, camera_depth_data, camera_info_data)

    input("Press ENTER to test twirling skill")
    skill_library.reset()
    camera_header, camera_color_data, camera_info_data, camera_depth_data = camera.get_camera_data()
    skill_library.twirling_skill(camera_color_data, camera_depth_data, camera_info_data)

    # A bit sensitive to tuning
    # input("Press ENTER to test cutting skill")
    # skill_library.reset()
    # camera_header, camera_color_data, camera_info_data, camera_depth_data = camera.get_camera_data()
    # skill_library.cutting_skill(camera_color_data, camera_depth_data, camera_info_data)