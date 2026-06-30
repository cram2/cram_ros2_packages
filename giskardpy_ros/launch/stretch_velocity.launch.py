from giskardpy_ros.launch_helpers import giskard_launch_description


def generate_launch_description():
    return giskard_launch_description(
        executable="stretch_velocity",
        root_links=["link_straight_gripper", "map"],
        tip_links=["link_gripper_fingertip_left", "base_link"],
    )
