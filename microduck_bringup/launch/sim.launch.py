"""Microduck in MuJoCo + ONNX policy (+ skills) + URDF/TF + head camera + RViz.

  ros2 launch microduck_bringup sim.launch.py                       # velstand from ~/microduck-policies
  ros2 launch microduck_bringup sim.launch.py scene:=ball           # with a ball to kick
  ros2 run teleop_twist_keyboard teleop_twist_keyboard              # in another terminal
  ros2 service call /microduck/skill/kick_left std_srvs/srv/Trigger
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _robot_state_publisher(context, *args, **kwargs):
    """Generate the URDF from the MuJoCo model at launch time and hand it to robot_state_publisher."""
    from microduck_sim.urdf import resolve_scene, urdf_for_scene
    lc = context.launch_configurations
    urdf = urdf_for_scene(resolve_scene(lc["scene"], lc["microduck_rl_path"]))
    return [Node(package="robot_state_publisher", executable="robot_state_publisher",
                 name="robot_state_publisher", output="screen",
                 parameters=[{"robot_description": urdf, "use_sim_time": True}])]


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
        DeclareLaunchArgument("ball", default_value="false",
                              description="Add the kick ball to a scene without one (e.g. the apartment)"),
        DeclareLaunchArgument("ball_x", default_value="0.6"),
        DeclareLaunchArgument("ball_y", default_value="0.0"),
        DeclareLaunchArgument("microduck_rl_path",
                              default_value=os.environ.get("MICRODUCK_RL_PATH",
                                                           os.path.expanduser("~/microduck_rl")),
                              description="Clone of pollen-robotics/microduck_rl (model + meshes)"),
        DeclareLaunchArgument("use_bam", default_value="true"),
        DeclareLaunchArgument("sync_mode", default_value="true"),
        DeclareLaunchArgument("camera", default_value="true", description="Render the head camera"),
        DeclareLaunchArgument("camera_rate", default_value="15.0"),
        DeclareLaunchArgument("gl_backend", default_value="auto", description="auto | egl | glfw | osmesa"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=os.path.join(share, "rviz", "microduck.rviz")),
        DeclareLaunchArgument("joy", default_value="false",
                              description="Gamepad: LB + sticks to walk, A/B/X/Y/RB for skills"),
    ]
    lc = LaunchConfiguration
    sim_time = {"use_sim_time": True}
    scene_args = {"microduck_rl_path": lc("microduck_rl_path"), "scene": lc("scene"),
                  "add_ball": lc("ball"), "ball_x": lc("ball_x"), "ball_y": lc("ball_y")}

    sim = Node(
        package="microduck_sim", executable="sim_node", name="microduck_sim", output="screen",
        parameters=[params, scene_args, {"use_bam": lc("use_bam"), "sync_mode": lc("sync_mode")}],
    )
    policy = Node(
        package="microduck_policy", executable="policy_node", name="microduck_policy", output="screen",
        parameters=[params, sim_time, {
            "policy_dir": lc("policy_dir"),
            "walking_policy": lc("walking_policy"),
            "standing_policy": lc("standing_policy"),
        }],
    )
    camera = Node(
        package="microduck_sim", executable="camera_node", name="microduck_camera", output="screen",
        parameters=[scene_args, sim_time, {"rate": lc("camera_rate"), "gl_backend": lc("gl_backend")}],
        condition=IfCondition(lc("camera")),
    )
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2", arguments=["-d", lc("rviz_config")],
        parameters=[sim_time], condition=IfCondition(lc("rviz")),
    )
    joy_cond = IfCondition(lc("joy"))
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
    return LaunchDescription(args + [sim, policy, OpaqueFunction(function=_robot_state_publisher),
                                     camera, rviz, joy, teleop, skills])
