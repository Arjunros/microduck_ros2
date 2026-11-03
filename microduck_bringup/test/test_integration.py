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
        # Back-to-back resets: time must keep moving forward and the loop must not freeze.
        from sensor_msgs.msg import JointState
        stamps = []
        node.create_subscription(JointState, "/joint_states", lambda m: stamps.append(
            m.header.stamp.sec * 10**9 + m.header.stamp.nanosec), 50)
        assert _spin(node, 10, lambda: len(stamps) > 5)
        assert _call(node, "/microduck/reset")[0] and _call(node, "/microduck/reset")[0]
        n = len(stamps)
        assert _spin(node, 15, lambda: len(stamps) > n + 50), "sim froze after reset"
        assert all(b >= a for a, b in zip(stamps, stamps[1:])), "sim time went backwards"
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


def _renderer_available():
    from microduck_sim.camera_node import pick_gl_backend
    return pick_gl_backend("auto", log=lambda m: None) is not None


def test_camera_urdf_and_tf(node, fake_set, tmp_path):
    if not _renderer_available():
        pytest.skip("no OpenGL backend for MuJoCo here (use the pixi 'ci' environment)")
    import tf2_ros
    from geometry_msgs.msg import PointStamped
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String
    got = {}
    for topic, typ, key in (("/camera/image_raw", Image, "rgb"), ("/camera/depth/image_raw", Image, "depth"),
                            ("/camera/camera_info", CameraInfo, "info")):
        node.create_subscription(typ, topic, lambda m, k=key: got.__setitem__(k, m), qos_profile_sensor_data)
    node.create_subscription(PointStamped, "/microduck/ball", lambda m: got.__setitem__("ball", m), 10)
    from rclpy.qos import DurabilityPolicy, QoSProfile
    node.create_subscription(String, "/robot_description", lambda m: got.__setitem__("urdf", m.data),
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, node)
    launch = Launch("sim.launch.py", f"policy_dir:={fake_set}", "scene:=apartment", "ball:=true",
                    "camera_rate:=2.0", log=str(tmp_path / "launch.log"))
    try:
        assert _spin(node, 120, lambda: {"rgb", "depth", "info", "ball", "urdf"} <= set(got)), sorted(got)
        _spin(node, 3)
        tf = buf.lookup_transform("odom", "head_camera_optical_frame", rclpy.time.Time())
    finally:
        launch.stop()
    rgb, depth, info = got["rgb"], got["depth"], got["info"]
    assert (rgb.width, rgb.height, rgb.encoding) == (640, 360, "rgb8")
    assert rgb.header.frame_id == "head_camera_optical_frame" == info.header.frame_id
    d = np.frombuffer(bytes(depth.data), np.float32)
    assert depth.encoding == "32FC1" and np.isfinite(d).all() and 0.0 < d.min() and d.max() <= 15.0
    assert abs(info.k[0] - (360 / 2) / np.tan(np.radians(45) / 2)) < 1e-6      # fx from the 45 deg fovy
    assert '<joint name="head_yaw" type="revolute">' in got["urdf"]
    assert 0.0 < tf.transform.translation.z < 0.4                                # camera on the duck's head


def _agent_probe(node):
    from std_msgs.msg import String
    replies = []
    node.create_subscription(String, "/microduck/agent/reply", lambda m: replies.append(m.data), 10)
    pub = node.create_publisher(String, "/microduck/command", 10)

    def say(text, timeout=60):
        n = len(replies)
        pub.publish(String(data=text))
        assert _spin(node, timeout, lambda: len(replies) > n), f"no reply to {text!r}"
        return replies[-1]
    return say


def test_agent_drives_nav2(node, fake_set, tmp_path):
    from nav_msgs.msg import Path
    plans = []
    node.create_subscription(Path, "/plan", lambda m: plans.append(m), 10)
    launch = Launch("nav.launch.py", f"policy_dir:={fake_set}", f"map:={tmp_path}/apartment.yaml",
                    "agent:=true", "camera:=false", log=str(tmp_path / "launch.log"))
    try:
        from std_msgs.msg import String
        say = _agent_probe(node)
        assert _spin(node, 90, lambda: node.count_publishers("/microduck/agent/reply") > 0)
        _spin(node, 5)
        reply = say("where are you?")
        assert "nearest place: start" in reply, reply
        import yaml
        kitchen = yaml.safe_load(open(tmp_path / "apartment_places.yaml"))["places"]["kitchen"]
        # Nav2 activates last; the agent's goal is rejected until then, so ask until it plans.
        cmd = node.create_publisher(String, "/microduck/command", 10)
        for _ in range(6):
            cmd.publish(String(data="go to the kitchen"))
            if _spin(node, 15, lambda: bool(plans)):
                break
        assert plans, "the agent never got Nav2 to plan"
        end = plans[-1].poses[-1].pose.position
        assert abs(end.x - kitchen["x"]) < 0.1 and abs(end.y - kitchen["y"]) < 0.1
        assert say("stop", timeout=10) == "Stopped."
    finally:
        launch.stop()


def test_agent_triggers_skills(node, fake_set, tmp_path):
    """Agent -> skill service -> policy_state round trip (creates service clients at runtime)."""
    from std_msgs.msg import String
    states = []
    node.create_subscription(String, "/microduck/policy_state",
                             lambda m: (not states or states[-1] != m.data) and states.append(m.data), 50)
    launch = Launch("nav.launch.py", f"policy_dir:={fake_set}", f"map:={tmp_path}/apartment.yaml",
                    "agent:=true", "camera:=false", "ball:=true", log=str(tmp_path / "launch.log"))
    try:
        say = _agent_probe(node)
        assert _spin(node, 90, lambda: bool(states) and node.count_publishers("/microduck/agent/reply") > 0)
        _spin(node, 3)
        reply = say("kick with your left foot")
        assert "kick left" in reply and "wrong" not in reply, reply
        reply = say("sit then stand up")
        assert "did sit" in reply and "did stand" in reply, reply
    finally:
        launch.stop()
    assert "kick_left" in states and "sit" in states and "rise" in states
