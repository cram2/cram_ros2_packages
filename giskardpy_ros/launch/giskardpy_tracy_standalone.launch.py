from giskardpy_ros.launch_helpers import giskard_launch_description


def generate_launch_description():
    return giskard_launch_description(
        executable="tracy_standalone",
        xacro_package="iai_tracy_description",
        xacro_file="urdf/tracy.urdf.xacro",
        root_links=["map", "map"],
        tip_links=["l_gripper_tool_frame", "r_gripper_tool_frame"],
        rviz=True,
        map_world_tf=True,
    )
