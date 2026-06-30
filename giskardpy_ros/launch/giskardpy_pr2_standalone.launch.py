from giskardpy_ros.launch_helpers import giskard_launch_description


def generate_launch_description():
    return giskard_launch_description(
        executable="pr2_standalone",
        xacro_package="iai_pr2_description",
        xacro_file="robots/pr2_with_ft2_cableguide.xacro",
        root_links=["map", "map", "map", "map"],
        tip_links=[
            "r_gripper_tool_frame",
            "l_gripper_tool_frame",
            "base_footprint",
            "high_def_frame",
        ],
        rviz=True,
    )
