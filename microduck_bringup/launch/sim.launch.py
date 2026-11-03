"""Microduck in MuJoCo + ONNX policy + RViz.

  ros2 launch microduck_bringup sim.launch.py walking_policy:=/path/to/walk.onnx
  ros2 run teleop_twist_keyboard teleop_twist_keyboard     # in another terminal
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("microduck_bringup")
    params = os.path.join(share, "config", "sim.yaml")
    rviz_cfg = os.path.join(share, "rviz", "microduck.rviz")

    args = [
        DeclareLaunchArgument("walking_policy", default_value="",
                              description="Path to the walking .onnx (61D obs -> 14 actions)"),
        DeclareLaunchArgument("standing_policy", default_value="",
                              description="Optional standing .onnx, used when |cmd_vel| ~ 0"),
        DeclareLaunchArgument("microduck_rl_path",
                              default_value=os.environ.get("MICRODUCK_RL_PATH",
                                                           os.path.expanduser("~/microduck_rl")),
                              description="Clone of pollen-robotics/microduck_rl (model + meshes)"),
        DeclareLaunchArgument("use_bam", default_value="true"),
        DeclareLaunchArgument("sync_mode", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("joy", default_value="false",
                              description="Start joy + teleop_twist_joy for a gamepad"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
    ]
    sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}

    sim = Node(
        package="microduck_sim", executable="sim_node", name="microduck_sim", output="screen",
        parameters=[params, {
            "microduck_rl_path": LaunchConfiguration("microduck_rl_path"),
            "use_bam": LaunchConfiguration("use_bam"),
            "sync_mode": LaunchConfiguration("sync_mode"),
        }],
    )
    policy = Node(
        package="microduck_policy", executable="policy_node", name="microduck_policy", output="screen",
        parameters=[params, sim_time, {
            "walking_policy": LaunchConfiguration("walking_policy"),
            "standing_policy": LaunchConfiguration("standing_policy"),
        }],
    )
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2", arguments=["-d", rviz_cfg],
        parameters=[sim_time], condition=IfCondition(LaunchConfiguration("rviz")),
    )
    joy = Node(package="joy", executable="joy_node", name="joy",
               condition=IfCondition(LaunchConfiguration("joy")))
    teleop = Node(
        package="teleop_twist_joy", executable="teleop_node", name="teleop_twist_joy",
        parameters=[{
            "enable_button": 4,              # LB, hold to drive
            "axis_linear.x": 1, "scale_linear.x": 0.3,
            "axis_linear.y": 0, "scale_linear.y": 0.2,
            "axis_angular.yaw": 3, "scale_angular.yaw": 1.5,
        }],
        condition=IfCondition(LaunchConfiguration("joy")),
    )
    return LaunchDescription(args + [sim, policy, rviz, joy, teleop])
