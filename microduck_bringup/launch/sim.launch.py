"""Microduck in MuJoCo + ONNX policy (+ skills) + RViz.

  ros2 launch microduck_bringup sim.launch.py                       # velstand from ~/microduck-policies
  ros2 launch microduck_bringup sim.launch.py scene:=ball           # with a ball to kick
  ros2 run teleop_twist_keyboard teleop_twist_keyboard              # in another terminal
  ros2 service call /microduck/skill/kick_left std_srvs/srv/Trigger
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

    args = [
        DeclareLaunchArgument("policy_dir",
                              default_value=os.environ.get("MICRODUCK_POLICIES_PATH",
                                                           os.path.expanduser("~/microduck-policies")),
                              description="Policy set with manifest.json (pollen-robotics/microduck-policies)"),
        DeclareLaunchArgument("walking_policy", default_value="",
                              description="File in policy_dir or absolute path; empty = the set's walk slot"),
        DeclareLaunchArgument("standing_policy", default_value="",
                              description="Optional standing policy, used when |cmd_vel| ~ 0"),
        DeclareLaunchArgument("scene", default_value="flat",
                              description="flat | ball | apartment, or a scene XML path"),
        DeclareLaunchArgument("microduck_rl_path",
                              default_value=os.environ.get("MICRODUCK_RL_PATH",
                                                           os.path.expanduser("~/microduck_rl")),
                              description="Clone of pollen-robotics/microduck_rl (model + meshes)"),
        DeclareLaunchArgument("use_bam", default_value="true"),
        DeclareLaunchArgument("sync_mode", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=os.path.join(share, "rviz", "microduck.rviz")),
        DeclareLaunchArgument("joy", default_value="false",
                              description="Gamepad: LB + sticks to walk, A/B/X/Y/RB for skills"),
    ]
    sim_time = {"use_sim_time": True}

    sim = Node(
        package="microduck_sim", executable="sim_node", name="microduck_sim", output="screen",
        parameters=[params, {
            "microduck_rl_path": LaunchConfiguration("microduck_rl_path"),
            "scene": LaunchConfiguration("scene"),
            "use_bam": LaunchConfiguration("use_bam"),
            "sync_mode": LaunchConfiguration("sync_mode"),
        }],
    )
    policy = Node(
        package="microduck_policy", executable="policy_node", name="microduck_policy", output="screen",
        parameters=[params, sim_time, {
            "policy_dir": LaunchConfiguration("policy_dir"),
            "walking_policy": LaunchConfiguration("walking_policy"),
            "standing_policy": LaunchConfiguration("standing_policy"),
        }],
    )
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[sim_time], condition=IfCondition(LaunchConfiguration("rviz")),
    )
    joy_cond = IfCondition(LaunchConfiguration("joy"))
    joy = Node(package="joy", executable="joy_node", name="joy", condition=joy_cond)
    teleop = Node(
        package="teleop_twist_joy", executable="teleop_node", name="teleop_twist_joy",
        parameters=[{
            "enable_button": 4,              # LB, hold to drive
            "axis_linear.x": 1, "scale_linear.x": 0.3,
            "axis_linear.y": 0, "scale_linear.y": 0.2,
            "axis_angular.yaw": 3, "scale_angular.yaw": 1.5,
        }],
        condition=joy_cond,
    )
    skills = Node(package="microduck_bringup", executable="joy_skills", name="joy_skills", condition=joy_cond)
    return LaunchDescription(args + [sim, policy, rviz, joy, teleop, skills])
