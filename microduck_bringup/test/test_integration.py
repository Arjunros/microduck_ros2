"""Launch-level integration tests: the real nodes, talking over ROS 2.

Policies are stand-in linear ONNX nets (each with a distinct constant output),
so these check plumbing, timing and the skill/Nav2 wiring, not walking quality.
Needs a built, sourced workspace and a microduck_rl clone (MICRODUCK_RL_PATH).

  pixi run test-integration
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("onnx")
RL_PATH = os.path.expanduser(os.environ.get("MICRODUCK_RL_PATH", "~/microduck_rl"))
pytestmark = [
    pytest.mark.skipif(shutil.which("ros2") is None, reason="ros2 not on PATH (source the workspace)"),
    pytest.mark.skipif(not os.path.isdir(RL_PATH), reason="microduck_rl clone not found"),
]

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "microduck_policy", "test"))
from test_policy import _make_onnx  # noqa: E402

ENV = dict(os.environ, ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST", MICRODUCK_RL_PATH=RL_PATH)


@pytest.fixture(scope="module")
def fake_set(tmp_path_factory):
    d = tmp_path_factory.mktemp("policies")
    for f, b in {"velstand.onnx": 0.01, "alpha_sitstand.onnx": 0.04, "alpha_ground_pick.onnx": 0.05,
                 "roulade.onnx": 0.06, "ball_kick_left.onnx": 0.07, "ball_kick_right.onnx": 0.08}.items():
        _make_onnx(str(d / f), bias=b)
    (d / "manifest.json").write_text(json.dumps({"policies": [
        {"file": "velstand.onnx", "kind": "perpetual", "slot": "walk"},
        {"file": "alpha_sitstand.onnx", "kind": "scripted", "unwind_s": 1.0, "ramp_s": 2.0,
         "command": {"encoding": "posture_flag"}},
        {"file": "alpha_ground_pick.onnx", "kind": "episodic", "mode": "walk",
         "command": {"encoding": "phase", "period_s": 4.0, "end_phase": 0.7}},
        {"file": "roulade.onnx", "kind": "episodic", "duration_s": 1.0},
        {"file": "ball_kick_left.onnx", "name": "kick_left", "kind": "episodic", "duration_s": 0.5},
        {"file": "ball_kick_right.onnx", "name": "kick_right", "kind": "episodic", "duration_s": 0.5}]}))
    return d


class Launch:
    def __init__(self, *args, log):
        self.log = open(log, "w")
        self.p = subprocess.Popen(["ros2", "launch", "microduck_bringup", *args, "rviz:=false"],
                                  stdout=self.log, stderr=subprocess.STDOUT, env=ENV)

    def stop(self):
        self.p.send_signal(signal.SIGINT)
        try:
            return self.p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.p.kill()
            return None


@pytest.fixture
def node():
    from rclpy.node import Node
    rclpy.init()
    n = Node("integration_probe")
    yield n
    n.destroy_node()
    rclpy.shutdown()


def _spin(n, sec, until=None):
    end = time.time() + sec
    while time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.05)
        if until is not None and until():
            return True
    return until() if until is not None else True


def _call(n, name, timeout=30):
    from std_srvs.srv import Trigger
    cli = n.create_client(Trigger, name)
    assert cli.wait_for_service(timeout_sec=timeout), f"{name} unavailable"
    fut = cli.call_async(Trigger.Request())
    _spin(n, 5, fut.done)
    return fut.result().success, fut.result().message


def test_sync_loop_matches_direct_loop(node, fake_set, tmp_path):
    """Sim + policy over ROS reproduce an in-process run bit for bit."""
    from nav_msgs.msg import Odometry
    rows = {}
    node.create_subscription(Odometry, "/odom", lambda m: rows.__setitem__(
        m.header.stamp.sec * 10**9 + m.header.stamp.nanosec,
        (m.pose.pose.position.x, m.pose.pose.position.y)), 100)
    launch = Launch("sim.launch.py", f"policy_dir:={fake_set}", log=str(tmp_path / "launch.log"))
    try:
        assert _spin(node, 90, lambda: len(rows) >= 120), "sim did not publish"
    finally:
        assert launch.stop() == 0, "launch did not shut down cleanly on SIGINT"

    from microduck_policy.policy import load_policy_set
    from microduck_sim.engine import MicroduckSim
    sim = MicroduckSim(os.path.join(RL_PATH, "src/mjlab_microduck/robot/microduck/scene.xml"))
    ctrl = load_policy_set(str(fake_set))
    ref = {}
    for _ in range(1000):
        st = sim.state()
        ref[int(round(st.time * 1e9))] = sim.footprint_pose()[:2]
        tg, _, _ = ctrl.step(st.gyro, st.trunk_quat_wxyz, st.joint_pos, st.joint_vel, 0.02)
        sim.set_targets(tg)
        sim.step()
    common = sorted(set(rows) & set(ref))
    assert len(common) >= 100
    err = max(np.abs(np.subtract(rows[t], ref[t])).max() for t in common)
    assert err == 0.0, f"ROS loop diverged from the direct loop by {err} m"


def test_skills_over_ros(node, fake_set, tmp_path):
    from std_msgs.msg import String
    states = []
    node.create_subscription(String, "/microduck/policy_state",
                             lambda m: (not states or states[-1] != m.data) and states.append(m.data), 50)
    launch = Launch("sim.launch.py", f"policy_dir:={fake_set}", "scene:=ball", log=str(tmp_path / "launch.log"))
    try:
        assert _spin(node, 90, lambda: bool(states)), "policy never ran"
        assert _call(node, "/microduck/skill/kick_left")[0]
        assert not _call(node, "/microduck/skill/roulade")[0]          # one move at a time
        assert _spin(node, 10, lambda: states[-1] == "walk" and "kick_left" in states)
        assert _call(node, "/microduck/skill/ground_pick")[0]
        assert _spin(node, 15, lambda: states[-1] == "walk" and "ground_pick" in states)
        assert _call(node, "/microduck/sit")[0]
        assert not _call(node, "/microduck/skill/kick_right")[0]       # no moves from the seat
        assert _call(node, "/microduck/stand")[0]
        assert _spin(node, 20, lambda: states[-1] == "walk" and "rise" in states)
        assert _call(node, "/microduck/place_ball_left")[0]
    finally:
        launch.stop()
    assert states == ["walk", "kick_left", "walk", "ground_pick", "walk", "sit", "rise", "walk"]


def test_nav2_plans_across_the_apartment(node, fake_set, tmp_path):
    from geometry_msgs.msg import Twist
    from nav2_msgs.action import NavigateToPose
    from nav_msgs.msg import OccupancyGrid, Path
    from rclpy.action import ActionClient
    got = {"costmap": 0, "plan": None, "cmd": 0}
    node.create_subscription(OccupancyGrid, "/global_costmap/costmap",
                             lambda m: got.__setitem__("costmap", got["costmap"] + 1), 1)
    node.create_subscription(Path, "/plan", lambda m: got.__setitem__("plan", m), 10)
    node.create_subscription(Twist, "/cmd_vel", lambda m: got.__setitem__(
        "cmd", got["cmd"] + (abs(m.angular.z) + abs(m.linear.x) > 1e-3)), 10)
    launch = Launch("nav.launch.py", f"policy_dir:={fake_set}", f"map:={tmp_path}/apartment.yaml",
                    log=str(tmp_path / "launch.log"))
    try:
        ac = ActionClient(node, NavigateToPose, "navigate_to_pose")
        assert ac.wait_for_server(timeout_sec=90)
        assert _spin(node, 60, lambda: got["costmap"] > 0), "Nav2 never became active"
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.pose.position.x, goal.pose.pose.position.y = -2.5, 1.8   # kitchen
        goal.pose.pose.orientation.w = 1.0
        # bt_navigator activates last; goals sent before that are rejected, so retry.
        accepted = False
        for _ in range(15):
            fut = ac.send_goal_async(goal)
            if _spin(node, 5, fut.done) and fut.result().accepted:
                accepted = True
                break
            _spin(node, 2)
        assert accepted, "navigate_to_pose kept rejecting the goal"
        assert _spin(node, 20, lambda: got["plan"] is not None and got["cmd"] > 5)
    finally:
        launch.stop()
    assert os.path.isfile(tmp_path / "apartment.yaml")                       # generated on first use
    end = got["plan"].poses[-1].pose.position
    assert len(got["plan"].poses) > 50 and abs(end.x + 2.5) < 0.1 and abs(end.y - 1.8) < 0.1
