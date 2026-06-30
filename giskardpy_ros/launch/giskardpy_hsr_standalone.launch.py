from giskardpy_ros.launch_helpers import giskard_launch_description


def generate_launch_description():
    return giskard_launch_description(
        executable="hsr_standalone",
        xacro_package="hsr_description",
        xacro_file="robots/hsrb4s.urdf.xacro",
        root_links=["map", "map", "map"],
        tip_links=["hand_palm_link", "base_footprint", "head_rgbd_sensor_link"],
        rviz=True,
    )
