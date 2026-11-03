"""ROS 2 node running a Microduck ONNX locomotion policy at the state rate (50 Hz).

Subscribes  /joint_states, /imu/data (orientation = trunk orientation, gyro),
            /cmd_vel (geometry_msgs/Twist: linear.x, linear.y, angular.z),
            /microduck/head_cmd  (std_msgs/Float64MultiArray, 4: neck_pitch, head_pitch, head_yaw, head_roll deltas [rad])
            /microduck/body_cmd  (std_msgs/Float64MultiArray, 6: x, y, z [m], roll, pitch, yaw [rad])
Publishes   /microduck/joint_targets (sensor_msgs/JointState, header.stamp copied from the state)
            /microduck/policy_state (std_msgs/String: walk, stand, sit, rise, ground_pick, <skill>)
            /microduck/policy_obs (std_msgs/Float32MultiArray, when publish_debug is true)
Services    /microduck/skill/<name>   (std_srvs/Trigger) one per skill in the policy set,
                                      e.g. kick_left, kick_right, roulade, ground_pick
            /microduck/sit, /microduck/stand (std_srvs/Trigger)
            /microduck/policy/reset   (std_srvs/Trigger)

It steps once per /joint_states message, so it runs at whatever rate the sim
(or, later, the hardware driver) publishes state: 50 Hz for Microduck.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32MultiArray, Float64MultiArray, String
from std_srvs.srv import Trigger

from .constants import (BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_XY, BODY_CMD_MAX_Z, JOINT_NAMES,
                        NUM_JOINTS, VEL_LIMITS)
from .observation import make_command
from .policy import PolicyLoadError, load_policy_set


def _stamp_ns(s):
    return s.sec * 1_000_000_000 + s.nanosec


class MicroduckPolicyNode(Node):
    def __init__(self):
        super().__init__("microduck_policy")
        p = self.declare_parameter
        p("policy_dir", "")              # a policy set (with manifest.json): enables skills
        p("walking_policy", "")          # file in policy_dir or absolute path; "" = the set's walk slot
        p("standing_policy", "")
        p("place_ball_on_kick", True)    # call the sim's /microduck/place_ball_* when a kick starts
        p("action_scale", -1.0)          # <0: take it from manifest.json, else 1.0
        p("switch_threshold", 0.05)
        p("cmd_timeout", 0.5)            # s without /cmd_vel -> zero twist
        p("vx_limits", list(VEL_LIMITS["vx"]))
        p("vy_limits", list(VEL_LIMITS["vy"]))
        p("wz_limits", list(VEL_LIMITS["wz"]))
        p("publish_debug", False)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        scale = gp("action_scale")
        scale = None if scale is None or scale < 0 else scale
        try:
            self.ctrl = load_policy_set(gp("policy_dir"), gp("walking_policy"), gp("standing_policy"),
                                        gp("switch_threshold"), scale,
                                        log=lambda m: self.get_logger().warn(m))
        except PolicyLoadError as e:
            self.get_logger().fatal(str(e))
            raise
        c = self.ctrl
        for name, pol in (("walking", c.walking), ("standing", c.standing)):
            if pol is not None:
                self.get_logger().info(f"{name} policy: {pol.path} (action_scale={pol.action_scale})")
        extras = sorted(c.skills) + (["ground_pick"] if c.ground_pick_def else [])
        self.get_logger().info(
            f"skills: {', '.join(extras) or 'none'}; sit/stand: {'yes' if c.sitstand_def else 'no'}"
            + ("" if gp("policy_dir") else "  (set policy_dir to enable skills)"))

        self.limits = (gp("vx_limits"), gp("vy_limits"), gp("wz_limits"))
        self.cmd_timeout = gp("cmd_timeout")
        self.publish_debug = gp("publish_debug")

        self.twist = np.zeros(3)
        self.head = np.zeros(4)
        self.body = np.zeros(6)
        self._last_twist_time = None
        self._imu = None
        self._last_state_stamp = None
        self._index = None  # JointState index for each policy joint
        self._prev_label = self.ctrl.label
        self.place_ball_on_kick = gp("place_ball_on_kick")

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.pub_targets = self.create_publisher(JointState, "/microduck/joint_targets", qos)
        self.pub_obs = self.create_publisher(Float32MultiArray, "/microduck/policy_obs", 10)
        self.create_subscription(Imu, "/imu/data", self._on_imu, qos)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, qos)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(Float64MultiArray, "/microduck/head_cmd", self._on_head, 10)
        self.create_subscription(Float64MultiArray, "/microduck/body_cmd", self._on_body, 10)
        self.pub_state = self.create_publisher(String, "/microduck/policy_state", 10)
        self.create_service(Trigger, "/microduck/policy/reset", self._on_reset)
        for name in self.ctrl.skills:
            self.create_service(Trigger, f"/microduck/skill/{name}",
                                lambda rq, rs, n=name: self._answer(rs, *self._start_skill(n)))
        if self.ctrl.ground_pick_def is not None:
            self.create_service(Trigger, "/microduck/skill/ground_pick",
                                lambda rq, rs: self._answer(rs, *self.ctrl.start_ground_pick()))
        if self.ctrl.sitstand_def is not None:
            self.create_service(Trigger, "/microduck/sit", lambda rq, rs: self._answer(rs, *self.ctrl.sit()))
            self.create_service(Trigger, "/microduck/stand", lambda rq, rs: self._answer(rs, *self.ctrl.stand()))
        self._ball_clients = {side: self.create_client(Trigger, f"/microduck/place_ball_{side}")
                              for side in ("left", "right")}

    # ------------------------------------------------------------- commands
    def _on_cmd_vel(self, msg):
        (vx0, vx1), (vy0, vy1), (wz0, wz1) = self.limits
        self.twist = np.array([
            np.clip(msg.linear.x, vx0, vx1),
            np.clip(msg.linear.y, vy0, vy1),
            np.clip(msg.angular.z, wz0, wz1),
        ])
        self._last_twist_time = self.get_clock().now()

    def _on_head(self, msg):
        if len(msg.data) == 4:
            self.head = np.array(msg.data)
        else:
            self.get_logger().warn("head_cmd needs 4 values", throttle_duration_sec=5.0)

    def _on_body(self, msg):
        if len(msg.data) != 6:
            self.get_logger().warn("body_cmd needs 6 values", throttle_duration_sec=5.0)
            return
        b = np.array(msg.data)
        b[0:2] = np.clip(b[0:2], -BODY_CMD_MAX_XY, BODY_CMD_MAX_XY)
        b[2] = np.clip(b[2], -BODY_CMD_MAX_Z, BODY_CMD_MAX_Z)
        b[3:6] = np.clip(b[3:6], -BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE)
        self.body = b

    def _current_twist(self):
        if self._last_twist_time is None:
            return np.zeros(3)
        age = (self.get_clock().now() - self._last_twist_time).nanoseconds * 1e-9
        return self.twist if age <= self.cmd_timeout else np.zeros(3)

    def _answer(self, response, ok, message):
        response.success, response.message = bool(ok), message
        # rclpy forbids one call site logging at two severities, hence two calls.
        if ok:
            self.get_logger().info(message)
        else:
            self.get_logger().warn(message)
        return response

    def _start_skill(self, name):
        ok, msg = self.ctrl.start_skill(name)
        if ok and self.place_ball_on_kick and "kick" in name:
            side = "left" if "left" in name else "right" if "right" in name else None
            cli = self._ball_clients.get(side)
            if cli is not None and cli.service_is_ready():
                cli.call_async(Trigger.Request())
        return ok, msg

    def _on_reset(self, request, response):
        self.ctrl.reset()
        self._prev_label = self.ctrl.label
        self._last_state_stamp = None
        response.success = True
        response.message = "policy state cleared"
        return response

    # ---------------------------------------------------------------- state
    def _on_imu(self, msg):
        self._imu = msg

    def _on_joint_states(self, msg):
        if self._imu is None:
            return
        stamp = _stamp_ns(msg.header.stamp)
        if self._last_state_stamp is not None:
            if stamp < self._last_state_stamp:
                self.get_logger().info("state time went backwards (sim reset): clearing last action")
                self.ctrl.reset()
            elif stamp == self._last_state_stamp:
                return  # same tick republished (sim paused): don't double-step
        dt = 0.02
        if self._last_state_stamp is not None and 0 < stamp - self._last_state_stamp < 100_000_000:
            dt = (stamp - self._last_state_stamp) * 1e-9
        self._last_state_stamp = stamp

        if self._index is None or len(self._index) != NUM_JOINTS:
            try:
                self._index = [list(msg.name).index(n) for n in JOINT_NAMES]
            except ValueError as e:
                self.get_logger().error(f"/joint_states is missing a joint: {e}", throttle_duration_sec=5.0)
                self._index = None
                return
        pos = np.asarray(msg.position)[self._index]
        vel = np.asarray(msg.velocity)[self._index] if len(msg.velocity) else np.zeros(NUM_JOINTS)

        o = self._imu.orientation
        quat_wxyz = (o.w, o.x, o.y, o.z)
        g = self._imu.angular_velocity
        gyro = (g.x, g.y, g.z)

        self.ctrl.set_command(make_command(self._current_twist(), self.head, self.body))
        if self.ctrl.label != self._prev_label:
            self.get_logger().info(f"policy: {self._prev_label} -> {self.ctrl.label}")
            self._prev_label = self.ctrl.label
        self.pub_state.publish(String(data=self.ctrl.label))
        try:
            targets, obs, _ = self.ctrl.step(gyro, quat_wxyz, pos, vel, dt)
        except FloatingPointError as e:
            self.get_logger().error(str(e), throttle_duration_sec=1.0)
            return

        out = JointState()
        out.header.stamp = msg.header.stamp
        out.name = list(JOINT_NAMES)
        out.position = targets.astype(float).tolist()
        self.pub_targets.publish(out)

        if self.publish_debug:
            self.pub_obs.publish(Float32MultiArray(data=obs.tolist()))


def main(args=None):
    rclpy.init(args=args)
    try:
        node = MicroduckPolicyNode()
    except PolicyLoadError:
        rclpy.shutdown()
        return 1
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
