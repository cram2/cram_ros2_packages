from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def giskard_launch_description(
    executable,
    root_links,
    tip_links,
    xacro_package=None,
    xacro_file=None,
    rviz=False,
    map_world_tf=False,
):
    """Build a LaunchDescription for a giskard robot.

    :param executable: giskard node executable, e.g. "hsr_standalone".
    :param root_links: interactive_marker root_links list.
    :param tip_links: interactive_marker tip_links list.
    :param xacro_package: package share name holding the robot xacro (optional).
    :param xacro_file: path to the xacro within that package, e.g.
        "robots/hsrb4s.urdf.xacro" (optional).
    :param rviz: include the rviz2 node.
    :param map_world_tf: include a map->world static_transform_publisher.
    """
    nodes = []

    if map_world_tf:
        nodes.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_transform_publisher",
                output="screen",
                arguments=["0", "0", "0", "0", "0", "0", "map", "world"],
            )
        )

    if rviz:
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
            )
        )

    giskard_parameters = []
    if xacro_package and xacro_file:
        robot_description = Command(
            [
                FindExecutable(name="xacro"),
                " ",
                PathJoinSubstitution(
                    [FindPackageShare(xacro_package), *xacro_file.split("/")]
                ),
            ]
        )
        giskard_parameters = [{"robot_description": robot_description}]

    nodes.append(
        Node(
            package="giskardpy_ros",
            executable=executable,
            name="giskard",
            parameters=giskard_parameters,
            output="screen",
        )
    )

    nodes.append(
        Node(
            package="giskardpy_ros",
            executable="interactive_marker",
            name="giskard_interactive_marker",
            parameters=[{"root_links": root_links, "tip_links": tip_links}],
            output="screen",
        )
    )

    return LaunchDescription(nodes)
