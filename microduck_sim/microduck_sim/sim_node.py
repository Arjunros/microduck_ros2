"""ROS 2 node wrapping MicroduckSim.

Publishes   /clock, /joint_states, /imu/data, /odom (ground truth), /tf,
            /microduck/visual (MarkerArray of the robot meshes, for RViz)
Subscribes  /microduck/joint_targets (sensor_msgs/JointState, positions in rad)
Services    /microduck/reset (std_srvs/Trigger)

Sync mode (default): each control tick publishes the state, then waits for the
joint targets computed FROM THAT STATE (matched on header.stamp) before
stepping. Physics stays deterministic no matter how slow DDS or the policy is.
Async mode: free-runs in real time and uses whatever targets arrived last,
which is closer to what the real robot sees.
"""

import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .engine import MicroduckSim

DEFAULT_SCENE_REL = "src/mjlab_microduck/robot/microduck/scene.xml"


def _to_ns(t):
    """Sim time [s] -> integer nanoseconds. Rounding the whole value (not just
    the fraction) keeps 0.9999999999 from becoming a stamp of 0.0."""
    return int(round(t * 1e9))


def _stamp(ns):
    from builtin_interfaces.msg import Time
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def _stamp_ns(s):
    return s.sec * 1_000_000_000 + s.nanosec


def _mat_to_quat_xyzw(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(9))
    return q[1], q[2], q[3], q[0]


class MicroduckSimNode(Node):
    def __init__(self):
        super().__init__("microduck_sim")
        p = self.declare_parameter
        p("microduck_rl_path", os.environ.get("MICRODUCK_RL_PATH", os.path.expanduser("~/microduck_rl")))
        p("scene", "")  # absolute path, or relative to microduck_rl_path
        p("use_bam", True)
        p("vin", 7.4)
        p("vin_drop_gain", 0.1)
        p("timestep", 0.005)
        p("decimation", 4)
        p("init_z", 0.125)
        p("sync_mode", True)
        p("sync_timeout", 0.5)
        p("wait_for_policy", True)
        p("real_time_factor", 1.0)  # 0 = as fast as possible (sync mode only)
        p("publish_visual", True)
        p("visual_rate", 20.0)
        p("auto_reset_on_fall", False)
        p("world_frame", "world")
        p("base_frame", "trunk_base")

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        scene = gp("scene") or DEFAULT_SCENE_REL
        if not os.path.isabs(scene):
            scene = os.path.join(os.path.expanduser(gp("microduck_rl_path")), scene)

        self.sim = MicroduckSim(
            scene, use_bam=gp("use_bam"), vin=gp("vin"), vin_drop_gain=gp("vin_drop_gain"),
            timestep=gp("timestep"), decimation=gp("decimation"), init_z=gp("init_z"),
        )
        self.get_logger().info(
            f"scene={scene} actuators={'BAM M6' if self.sim.use_bam else 'XML position (no BAM)'} "
            f"control={1.0 / self.sim.control_dt:.0f} Hz"
        )

        self.sync_mode = gp("sync_mode")
        self.sync_timeout = gp("sync_timeout")
        self.wait_for_policy = gp("wait_for_policy")
        self.rtf = gp("real_time_factor")
        self.auto_reset = gp("auto_reset_on_fall")
        self.world_frame = gp("world_frame")
        self.base_frame = gp("base_frame")

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.pub_js = self.create_publisher(JointState, "/joint_states", qos)
        self.pub_imu = self.create_publisher(Imu, "/imu/data", qos)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.pub_vis = self.create_publisher(MarkerArray, "/microduck/visual", 1)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(JointState, "/microduck/joint_targets", self._on_targets, qos)
        self.create_service(Trigger, "/microduck/reset", self._on_reset)

        self._lock = threading.Lock()
        self._targets_event = threading.Event()
        self._last_target_stamp = None
        self._got_any_targets = False
        self._reset_requested = False
        self._stop = False

        self._visuals = self.sim.mesh_visuals() if gp("publish_visual") else []
        self._visual_period = 1.0 / max(gp("visual_rate"), 1e-3)
        self._last_visual = -1e9
        if gp("publish_visual"):
            self.get_logger().info(f"publishing {len(self._visuals)} mesh markers on /microduck/visual")

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ callbacks
    def _on_targets(self, msg):
        if len(msg.position) == 0:
            return
        with self._lock:
            self.sim.set_targets(np.array(msg.position), list(msg.name) or None)
            self._last_target_stamp = _stamp_ns(msg.header.stamp)
            self._got_any_targets = True
        self._targets_event.set()

    def _on_reset(self, request, response):
        self._reset_requested = True
        response.success = True
        response.message = "reset scheduled"
        return response

    # ----------------------------------------------------------- main loop
    def _run(self):
        wall_start, sim_start = time.monotonic(), self.sim.time
        while rclpy.ok() and not self._stop:
            if self._reset_requested or (self.auto_reset and self.sim.fallen()):
                with self._lock:
                    self.sim.reset()
                    self._got_any_targets = False
                self._reset_requested = False
                wall_start, sim_start = time.monotonic(), self.sim.time
                self.get_logger().info("simulation reset")

            t_k = _to_ns(self.sim.time)
            self._targets_event.clear()
            self._publish_state()

            if self.wait_for_policy and not self._got_any_targets:
                # Hold the robot frozen at its initial pose until a policy answers.
                self._targets_event.wait(0.1)
                wall_start, sim_start = time.monotonic(), self.sim.time
                continue

            if self.sync_mode:
                deadline = time.monotonic() + self.sync_timeout
                while not self._stop and rclpy.ok():
                    with self._lock:
                        ok = self._last_target_stamp == t_k
                    if ok:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.get_logger().warn("no targets for this tick within sync_timeout; reusing last",
                                               throttle_duration_sec=2.0)
                        break
                    self._targets_event.wait(remaining)
                    self._targets_event.clear()

            with self._lock:
                self.sim.step()

            # Real-time pacing.
            if self.rtf > 0 or not self.sync_mode:
                rtf = self.rtf if self.rtf > 0 else 1.0
                ahead = (self.sim.time - sim_start) / rtf - (time.monotonic() - wall_start)
                if ahead > 0:
                    time.sleep(ahead)
                elif ahead < -0.25:  # fell badly behind: rebase instead of sprinting
                    wall_start, sim_start = time.monotonic(), self.sim.time

    def _publish_state(self):
        with self._lock:
            st = self.sim.state()
            vis_poses = None
            if self._visuals and st.time - self._last_visual >= self._visual_period:
                vis_poses = [self.sim.geom_pose(v.geom_id) @ v.T_geom_mesh for v in self._visuals]
                self._last_visual = st.time
        stamp = _stamp(_to_ns(st.time))

        self.pub_clock.publish(Clock(clock=stamp))

        # IMU first: the policy node steps when /joint_states arrives.
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = self.base_frame
        w, x, y, z = st.trunk_quat_wxyz
        imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w = x, y, z, w
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = map(float, st.gyro)
        ax, ay, az = map(float, st.accel)
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = ax, ay, az
        self.pub_imu.publish(imu)

        js = JointState()
        js.header.stamp = stamp
        js.name = list(st.joint_names)
        js.position = st.joint_pos.tolist()
        js.velocity = st.joint_vel.tolist()
        js.effort = st.joint_effort.tolist()
        self.pub_js.publish(js)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.base_frame
        pp = odom.pose.pose
        pp.position.x, pp.position.y, pp.position.z = map(float, st.trunk_pos)
        pp.orientation.x, pp.orientation.y, pp.orientation.z, pp.orientation.w = x, y, z, w
        tw = odom.twist.twist
        tw.linear.x, tw.linear.y, tw.linear.z = map(float, st.trunk_lin_vel)
        tw.angular.x, tw.angular.y, tw.angular.z = map(float, st.trunk_ang_vel)
        self.pub_odom.publish(odom)

        tf = TransformStamped()
        tf.header = odom.header
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, st.trunk_pos)
        tf.transform.rotation = pp.orientation
        self.tf.sendTransform(tf)

        if vis_poses is not None:
            self._publish_visual(stamp, vis_poses)

    def _publish_visual(self, stamp, poses):
        arr = MarkerArray()
        for i, (v, T) in enumerate(zip(self._visuals, poses)):
            mk = Marker()
            mk.header.stamp = stamp
            mk.header.frame_id = self.world_frame
            mk.ns = "microduck"
            mk.id = i
            mk.type = Marker.MESH_RESOURCE
            mk.action = Marker.ADD
            mk.mesh_resource = "file://" + v.mesh_file
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = map(float, T[:3, 3])
            qx, qy, qz, qw = _mat_to_quat_xyzw(T[:3, :3])
            mk.pose.orientation.x, mk.pose.orientation.y = float(qx), float(qy)
            mk.pose.orientation.z, mk.pose.orientation.w = float(qz), float(qw)
            mk.scale.x = mk.scale.y = mk.scale.z = 1.0
            r, g, b, a = v.rgba
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = float(r), float(g), float(b), float(a) or 1.0
            arr.markers.append(mk)
        self.pub_vis.publish(arr)

    def destroy_node(self):
        self._stop = True
        self._targets_event.set()
        self._thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MicroduckSimNode()
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
