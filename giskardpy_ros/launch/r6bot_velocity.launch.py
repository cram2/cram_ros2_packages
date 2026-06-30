from giskardpy_ros.launch_helpers import giskard_launch_description


def generate_launch_description():
    return giskard_launch_description(
        executable="r6bot",
        root_links=["base_link", "map"],
        tip_links=["tool0", "base_link"],
    )
