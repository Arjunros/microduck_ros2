"""Microduck + Nav2 in the apartment scene.

  ros2 launch microduck_bringup nav.launch.py              # known map (generated from the scene)
  ros2 launch microduck_bringup nav.launch.py mode:=slam   # build the map with slam_toolbox as it walks

Then click "2D Goal Pose" in RViz.

mode:=map uses a ground-truth map rasterized from the MuJoCo scene, and since
/odom is ground truth too, map -> odom is identity (no localization needed).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

NAV_NODES = ["controller_server", "planner_server", "behavior_server", "velocity_smoother", "bt_navigator"]


def _ensure_map(context, *args, **kwargs):
    """Generate the ground-truth map on first use."""
    if context.launch_configurations.get("mode") != "map":
        return []
    path = os.path.expanduser(context.launch_configurations["map"])
    if not os.path.isfile(path):
        from microduck_sim.mapgen import main as make_map
        out = path[:-5] if path.endswith(".yaml") else path
        make_map(["--scene", context.launch_configurations["scene"],
                  "--microduck-rl-path", context.launch_configurations["microduck_rl_path"],
                  "--out", out])
    return [Node(package="nav2_map_server", executable="map_server", name="map_server", output="screen",
                 parameters=[context.launch_configurations["nav_params"],
                             {"yaml_filename": path, "use_sim_time": True}]),
            Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
                 name="lifecycle_manager_map", output="screen",
                 parameters=[{"use_sim_time": True, "autostart": True, "node_names": ["map_server"]}])]


def generate_launch_description():
    share = get_package_share_directory("microduck_bringup")
    rl_default = os.environ.get("MICRODUCK_RL_PATH", os.path.expanduser("~/microduck_rl"))

    args = [
        DeclareLaunchArgument("mode", default_value="map", description="map | slam"),
        DeclareLaunchArgument("scene", default_value="apartment"),
        DeclareLaunchArgument("map", default_value=os.path.expanduser("~/microduck_maps/apartment.yaml"),
                              description="Map YAML for mode:=map (generated if missing)"),
        DeclareLaunchArgument("microduck_rl_path", default_value=rl_default),
        DeclareLaunchArgument("nav_params", default_value=os.path.join(share, "config", "nav2.yaml")),
        DeclareLaunchArgument("slam_params", default_value=os.path.join(share, "config", "slam.yaml")),
        DeclareLaunchArgument("rviz", default_value="true"),
        # passed through to sim.launch.py
        DeclareLaunchArgument("policy_dir",
                              default_value=os.environ.get("MICRODUCK_POLICIES_PATH",
                                                           os.path.expanduser("~/microduck-policies"))),
        DeclareLaunchArgument("walking_policy", default_value=""),
        DeclareLaunchArgument("use_bam", default_value="true"),
        DeclareLaunchArgument("joy", default_value="false"),
    ]
    lc = LaunchConfiguration
    nav_params = lc("nav_params")
    sim_time = {"use_sim_time": True}

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "sim.launch.py")),
        launch_arguments={
            "scene": lc("scene"), "microduck_rl_path": lc("microduck_rl_path"),
            "policy_dir": lc("policy_dir"), "walking_policy": lc("walking_policy"),
            "use_bam": lc("use_bam"), "joy": lc("joy"), "rviz": lc("rviz"),
            "rviz_config": os.path.join(share, "rviz", "nav.rviz"),
        }.items(),
    )

    map_to_odom = Node(  # ground-truth odom: the map frame is the odom frame
        package="tf2_ros", executable="static_transform_publisher", name="map_to_odom",
        arguments=["--frame-id", "map", "--child-frame-id", "odom"],
        parameters=[sim_time], condition=LaunchConfigurationEquals("mode", "map"),
    )
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("slam_toolbox"), "launch", "online_async_launch.py")),
        launch_arguments={"use_sim_time": "true", "slam_params_file": lc("slam_params")}.items(),
        condition=LaunchConfigurationEquals("mode", "slam"),
    )

    nav = [
        Node(package="nav2_controller", executable="controller_server", output="screen",
             parameters=[nav_params], remappings=[("cmd_vel", "cmd_vel_nav")]),
        Node(package="nav2_planner", executable="planner_server", name="planner_server", output="screen",
             parameters=[nav_params]),
        Node(package="nav2_behaviors", executable="behavior_server", name="behavior_server", output="screen",
             parameters=[nav_params], remappings=[("cmd_vel", "cmd_vel_nav")]),
        Node(package="nav2_velocity_smoother", executable="velocity_smoother", name="velocity_smoother",
             output="screen", parameters=[nav_params],
             remappings=[("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")]),
        Node(package="nav2_bt_navigator", executable="bt_navigator", name="bt_navigator", output="screen",
             parameters=[nav_params]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_navigation", output="screen",
             parameters=[{"use_sim_time": True, "autostart": True, "node_names": NAV_NODES,
                          "bond_timeout": 10.0}]),
    ]
    return LaunchDescription(args + [sim, OpaqueFunction(function=_ensure_map), map_to_odom, slam] + nav)
