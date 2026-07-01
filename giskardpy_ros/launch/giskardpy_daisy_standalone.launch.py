from giskardpy_ros.launch_helpers import giskard_launch_description


def generate_launch_description():
    return giskard_launch_description(
        executable="daisy_standalone",
        xacro_package="iai_daisy_description",
        xacro_file="robots/daisy.urdf.xacro",
        root_links=["map", "map"],
        tip_links=["left_gripper_tool_frame", "right_gripper_tool_frame"],
        rviz=True,
        map_world_tf=True,
    )
