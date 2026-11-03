"""ROS 2 node: natural-language control of Microduck.

  /microduck/command        std_msgs/String  in   "go to the kitchen and sit"
  /microduck/agent/reply    std_msgs/String  out  the duck's answer
  /microduck/agent/events   std_msgs/String  out  thinking / tool calls / results, for UIs

"stop" (or halt/freeze) on /microduck/command is handled at once: navigation is
cancelled and the robot stops, even mid-request. Other commands queue.

Params: backend (rule | ollama | openai | anthropic), model, base_url, api_key_env,
        places_file, walk_speed, turn_speed, nav_timeout.
"""

import io
import math
import os
import queue
import re
import threading
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from .agent import Agent
from .backends import BackendError, make_backend
from .places import load_places
from .tools import SKILL_NAMES, Robot, ToolResult

STOP_RE = re.compile(r"^\s*(stop|halt|freeze|stop it|stop now)\W*$", re.I)
# head_cmd = [neck_pitch, head_pitch, head_yaw, head_roll] deltas. Measured on the model:
# +head_yaw turns the view left, +head_pitch tilts it down.
LOOK = {"left": [0, 0, 0.5, 0], "right": [0, 0, -0.5, 0], "up": [0, -0.35, 0, 0],
        "down": [0, 0.35, 0, 0], "center": [0, 0, 0, 0]}


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class RosRobot(Robot):
    """Robot interface implemented with ROS topics, services and Nav2."""

    def __init__(self, node, places, aliases):
        self.n = node
        self.places = places
        self.aliases = aliases
        self.has_camera = True
        self.skills = SKILL_NAMES
        self._goal = None

    @property
    def has_ball(self):
        return self.n.ball is not None

    # ------------------------------------------------------------------ helpers
    def _wait(self, fut, timeout):
        end = time.monotonic() + timeout
        while not fut.done():
            if time.monotonic() > end or self.n.agent.cancel.is_set():
                return False
            time.sleep(0.02)
        return True

    def _pose(self):
        """(x, y, yaw) in the map frame: TF if available (SLAM), else ground-truth /odom."""
        try:
            tf = self.n.tf_buffer.lookup_transform("map", "base_footprint", rclpy.time.Time())
            t, q = tf.transform.translation, tf.transform.rotation
            return t.x, t.y, _yaw(q)
        except Exception:
            o = self.n.odom
            if o is None:
                return None
            p = o.pose.pose
            return p.position.x, p.position.y, _yaw(p.orientation)

    def _navigate(self, x, y, yaw, label):
        if not self.n.nav.wait_for_server(timeout_sec=2.0):
            return ToolResult(False, "navigation isn't running (start nav.launch.py)")
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        send = self.n.nav.send_goal_async(goal)
        if not self._wait(send, 10.0) or not send.result().accepted:
            return ToolResult(False, f"navigation rejected the goal for {label}")
        self._goal = send.result()
        res = self._goal.get_result_async()
        ok = self._wait(res, self.n.nav_timeout)
        if not ok:
            self._goal.cancel_goal_async()
            self._goal = None
            return ToolResult(False, f"gave up going to {label}" +
                              (" (cancelled)" if self.n.agent.cancel.is_set() else " (timeout)"))
        self._goal = None
        status = res.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            return ToolResult(True, f"arrived at {label}")
        return ToolResult(False, f"couldn't reach {label} (Nav2 status {status})")

    # -------------------------------------------------------------------- tools
    def go_to(self, place):
        name = self.resolve_place(place)
        if name is None:
            return ToolResult(False, f"unknown place '{place}'. Known: {', '.join(sorted(self.places))}")
        x, y = self.places[name]
        pose = self._pose()
        yaw = math.atan2(y - pose[1], x - pose[0]) if pose else 0.0
        return self._navigate(x, y, yaw, f"the {name}")

    def go_to_point(self, x, y):
        pose = self._pose()
        yaw = math.atan2(y - pose[1], x - pose[0]) if pose else 0.0
        return self._navigate(x, y, yaw, f"({x:.2f}, {y:.2f})")

    def go_to_ball(self):
        b, pose = self.n.ball, self._pose()
        if b is None:
            return ToolResult(False, "I don't know where a ball is")
        bx, by = b.point.x, b.point.y          # odom frame; map == odom in map mode
        dx, dy = bx - pose[0], by - pose[1]
        dist = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)
        if dist < 0.3:
            return self.turn(math.degrees(_wrap(yaw - pose[2])))
        stand_off = 0.2
        res = self._navigate(bx - stand_off * dx / dist, by - stand_off * dy / dist, yaw, "the ball")
        if res.ok:
            res.text = "next to the ball, facing it"
        return res

    def walk(self, distance_m):
        d = float(distance_m)
        start = self._pose()
        if start is None:
            return ToolResult(False, "no odometry")
        v = math.copysign(self.n.walk_speed, d)
        deadline = time.monotonic() + abs(d) / self.n.walk_speed * 5 + 3   # it walks slower than asked
        moved = 0.0
        while moved < abs(d) and time.monotonic() < deadline and not self.n.agent.cancel.is_set():
            self.n.drive(v, 0.0)
            time.sleep(0.1)
            p = self._pose()
            moved = math.hypot(p[0] - start[0], p[1] - start[1])
        self.n.drive(0.0, 0.0)
        ok = moved >= abs(d) * 0.8
        return ToolResult(ok, f"walked {moved:.2f} m" + ("" if ok else f" of {abs(d):.2f} m"))

    def turn(self, degrees):
        target = math.radians(float(degrees))
        start = self._pose()
        if start is None:
            return ToolResult(False, "no odometry")
        w = math.copysign(self.n.turn_speed, target)
        turned, last = 0.0, start[2]
        deadline = time.monotonic() + abs(target) / self.n.turn_speed * 3 + 3
        while abs(turned) < abs(target) and time.monotonic() < deadline and not self.n.agent.cancel.is_set():
            self.n.drive(0.0, w)
            time.sleep(0.05)
            yaw = self._pose()[2]
            turned += _wrap(yaw - last)
            last = yaw
        self.n.drive(0.0, 0.0)
        ok = abs(turned) >= abs(target) * 0.8
        return ToolResult(ok, f"turned {math.degrees(turned):.0f} degrees")

    def do_skill(self, skill):
        srv = {"sit": "/microduck/sit", "stand": "/microduck/stand"}.get(skill, f"/microduck/skill/{skill}")
        cli = self.n.client(srv)
        if not cli.wait_for_service(timeout_sec=2.0):
            return ToolResult(False, f"{skill} isn't available (no policy for it)")
        fut = cli.call_async(Trigger.Request())
        if not self._wait(fut, 5.0):
            return ToolResult(False, f"{skill}: no answer")
        r = fut.result()
        if not r.success:
            return ToolResult(False, f"{skill}: {r.message}")
        # Wait for the move to play out: back to walking (or seated, for sit).
        done = {"sit": ("sit",), "stand": ("walk", "stand")}.get(skill, ("walk", "stand"))
        time.sleep(0.3)
        end = time.monotonic() + 15.0
        while self.n.policy_state not in done and time.monotonic() < end and not self.n.agent.cancel.is_set():
            time.sleep(0.05)
        return ToolResult(True, f"did {skill.replace('_', ' ')}")

    def look(self, direction):
        cmd = LOOK.get(direction)
        if cmd is None:
            return ToolResult(False, f"can't look '{direction}'")
        self.n.pub_head.publish(Float64MultiArray(data=[float(v) for v in cmd]))
        time.sleep(0.8)
        return ToolResult(True, f"looking {direction}")

    def where_am_i(self):
        pose = self._pose()
        if pose is None:
            return ToolResult(False, "no position yet")
        x, y, yaw = pose
        near = min(self.places.items(), key=lambda kv: math.hypot(kv[1][0] - x, kv[1][1] - y),
                   default=(None, None))[0]
        where = f", nearest place: {near}" if near else ""
        return ToolResult(True, f"at ({x:.2f}, {y:.2f}) facing {math.degrees(yaw):.0f} deg{where}")

    def stop(self):
        if self._goal is not None:
            self._goal.cancel_goal_async()
        self.n.drive(0.0, 0.0)
        return ToolResult(True, "stopped")

    def see(self):
        img = self.n.image
        if img is None:
            return ToolResult(False, "no camera image (is the camera running?)")
        try:
            from PIL import Image as PILImage
        except ImportError:
            return ToolResult(False, "Pillow is not installed, can't encode the image")
        arr = np.frombuffer(bytes(img.data), np.uint8).reshape(img.height, img.width, 3)
        pil = PILImage.fromarray(arr)
        pil.thumbnail((512, 512))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=80)
        return ToolResult(True, f"camera image {pil.width}x{pil.height} attached", buf.getvalue())


class AgentNode(Node):
    def __init__(self):
        super().__init__("microduck_agent")
        p = self.declare_parameter
        p("backend", "rule")
        p("model", "")
        p("base_url", "")
        p("api_key_env", "")
        p("places_file", os.path.expanduser("~/microduck_maps/apartment_places.yaml"))
        p("walk_speed", 0.3)     # commanded; velstand achieves well under half (see nav2.yaml)
        p("turn_speed", 1.0)     # in-place yaw is only tracked from ~0.6 rad/s up
        p("nav_timeout", 180.0)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        self.walk_speed, self.turn_speed = float(gp("walk_speed")), float(gp("turn_speed"))
        self.nav_timeout = float(gp("nav_timeout"))
        self.odom = self.ball = self.image = None
        self.policy_state = "walk"
        cb = ReentrantCallbackGroup()

        import tf2_ros
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose", callback_group=cb)
        self._trigger_clients = {}
        self._cb = cb
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_head = self.create_publisher(Float64MultiArray, "/microduck/head_cmd", 10)
        self.pub_reply = self.create_publisher(String, "/microduck/agent/reply", 10)
        self.pub_events = self.create_publisher(String, "/microduck/agent/events", 10)
        self.create_subscription(Odometry, "/odom", lambda m: setattr(self, "odom", m), 10, callback_group=cb)
        self.create_subscription(PointStamped, "/microduck/ball", lambda m: setattr(self, "ball", m), 10,
                                 callback_group=cb)
        self.create_subscription(Image, "/camera/image_raw", lambda m: setattr(self, "image", m),
                                 qos_profile_sensor_data, callback_group=cb)
        self.create_subscription(String, "/microduck/policy_state",
                                 lambda m: setattr(self, "policy_state", m.data), 10, callback_group=cb)
        self.create_subscription(String, "/microduck/command", self._on_command, 10, callback_group=cb)

        places, aliases = load_places(gp("places_file"))
        self.robot = RosRobot(self, places, aliases)
        try:
            backend = make_backend(gp("backend"), self.robot, gp("model"), gp("base_url"), gp("api_key_env"))
        except BackendError as e:
            self.get_logger().fatal(str(e))
            raise
        self.agent = Agent(backend, self.robot, on_event=self._event)
        self._queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()
        self.get_logger().info(
            f"agent ready: backend={backend.name}{(' model=' + gp('model')) if gp('model') else ''}, "
            f"{len(places)} places ({', '.join(sorted(places)) or 'none: run make_map'}). "
            "Send text on /microduck/command.")

    def client(self, name):
        if name not in self._trigger_clients:
            self._trigger_clients[name] = self.create_client(Trigger, name, callback_group=self._cb)
        return self._trigger_clients[name]

    def drive(self, vx, wz):
        t = Twist()
        t.linear.x, t.angular.z = float(vx), float(wz)
        self.pub_cmd.publish(t)

    def _event(self, kind, text):
        self.pub_events.publish(String(data=f"{kind}: {text}" if text else kind))

    def _on_command(self, msg):
        text = msg.data.strip()
        if not text:
            return
        if STOP_RE.match(text):
            self.agent.cancel.set()
            self.robot.stop()
            with self._queue.mutex:
                self._queue.queue.clear()
            self.pub_reply.publish(String(data="Stopped."))
            return
        self._queue.put(text)

    def _worker(self):
        while rclpy.ok():
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self.get_logger().info(f"> {text}")
            try:
                reply = self.agent.run(text)
            except BackendError as e:
                reply = f"My brain isn't answering: {e}"
            except Exception as e:  # never let one bad request kill the agent
                self.get_logger().error(f"agent error: {e!r}")
                reply = f"Something went wrong: {e}"
            self.get_logger().info(f"< {reply}")
            self.pub_reply.publish(String(data=reply))
            self._event("idle", "")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = AgentNode()
    except BackendError:
        rclpy.shutdown()
        return 1
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.drive(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
