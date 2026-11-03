"""ROS 2 node wrapping MicroduckSim.

Publishes   /clock, /joint_states, /imu/data, /odom (ground truth), /tf,
            /scan (virtual planar lidar, for Nav2),
            /microduck/visual (MarkerArray: scene geoms, for RViz),
            /microduck/sim_qpos (full state for the camera node),
            /microduck/ball (ground-truth ball position, ball scenes only)
Subscribes  /microduck/joint_targets (sensor_msgs/JointState, positions in rad)
Services    /microduck/reset, /microduck/place_ball_{left,right} (ball scenes)

TF: odom -> base_footprint (x, y, yaw) -> trunk_base (height, roll, pitch),
    base_footprint -> laser_link (static).

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
from geometry_msgs.msg import PointStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .engine import SCENE_DIR_REL, SCENES, MicroduckSim




def _to_ns(t):
    """Sim time [s] -> integer nanoseconds. Rounding the whole value (not just
    the fraction) keeps 0.9999999999 from becoming a stamp of 0.0."""
    return int(round(t * 1e9))


def _stamp(ns):
    from builtin_interfaces.msg import Time
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def _stamp_ns(s):
    return s.sec * 1_000_000_000 + s.nanosec


def _quat_mul(a, b):
    """Hamilton product of (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _set_pose(marker, T):
    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, T[:3, 3])
    qx, qy, qz, qw = _mat_to_quat_xyzw(T[:3, :3])
    o = marker.pose.orientation
    o.x, o.y, o.z, o.w = float(qx), float(qy), float(qz), float(qw)


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
        p("scene", "flat")  # flat | ball | apartment, a path relative to microduck_rl_path, or absolute
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
        p("publish_visual", True)          # scene geoms (+ robot meshes if publish_robot_markers)
        p("publish_robot_markers", False)  # RViz shows the robot from the URDF instead
        p("add_ball", False)               # add Pollen's kick ball to scenes that lack one
        p("ball_x", 0.6)
        p("ball_y", 0.0)
        p("state_rate", 30.0)              # /microduck/sim_qpos for the camera node [Hz]
        p("visual_rate", 20.0)
        p("auto_reset_on_fall", False)
        p("odom_frame", "odom")
        p("footprint_frame", "base_footprint")
        p("base_frame", "trunk_base")
        p("publish_scan", True)
        p("scan_frame", "laser_link")
        p("scan_rate", 10.0)
        p("scan_rays", 360)
        p("scan_height", 0.12)
        p("scan_range_max", 6.0)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        scene = gp("scene") or "flat"
        if scene in SCENES:
            scene = os.path.join(SCENE_DIR_REL, SCENES[scene])
        if not os.path.isabs(scene):
            scene = os.path.join(os.path.expanduser(gp("microduck_rl_path")), scene)

        self.sim = MicroduckSim(
            scene, use_bam=gp("use_bam"), vin=gp("vin"), vin_drop_gain=gp("vin_drop_gain"),
            timestep=gp("timestep"), decimation=gp("decimation"), init_z=gp("init_z"),
            add_ball=gp("add_ball"), ball_xy=(gp("ball_x"), gp("ball_y")),
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
        self.odom_frame = gp("odom_frame")
        self.footprint_frame = gp("footprint_frame")
        self.base_frame = gp("base_frame")
        self.scan_frame = gp("scan_frame")
        self.publish_scan = gp("publish_scan")
        self.scan_period = 1.0 / max(gp("scan_rate"), 1e-3)
        self.scan_rays = int(gp("scan_rays"))
        self.scan_height = float(gp("scan_height"))
        self.scan_range_max = float(gp("scan_range_max"))
        self._last_scan = -1e9

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.pub_js = self.create_publisher(JointState, "/joint_states", qos)
        self.pub_imu = self.create_publisher(Imu, "/imu/data", qos)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.pub_scan = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.pub_vis = self.create_publisher(MarkerArray, "/microduck/visual", 1)
        self.pub_qpos = self.create_publisher(Float64MultiArray, "/microduck/sim_qpos", 1)
        self.pub_ball = self.create_publisher(PointStamped, "/microduck/ball", 10)
        self._state_period = 1.0 / max(gp("state_rate"), 1e-3)
        self._last_qpos = -1e9
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.create_subscription(JointState, "/microduck/joint_targets", self._on_targets, qos)
        self.create_service(Trigger, "/microduck/reset", self._on_reset)
        if self.sim.has_ball:
            self.create_service(Trigger, "/microduck/place_ball_left", lambda rq, rs: self._on_ball(rs, "left"))
            self.create_service(Trigger, "/microduck/place_ball_right", lambda rq, rs: self._on_ball(rs, "right"))
            self.get_logger().info("scene has a ball: /microduck/place_ball_{left,right} available")
        if self.publish_scan:
            laser = TransformStamped()
            laser.header.frame_id = self.footprint_frame
            laser.child_frame_id = self.scan_frame
            laser.transform.translation.z = self.scan_height
            laser.transform.rotation.w = 1.0
            self.static_tf.sendTransform(laser)
        self._world_geoms = self._collect_world_geoms() if gp("publish_visual") else []

        self._lock = threading.Lock()
        self._targets_event = threading.Event()
        self._last_target_stamp = None
        self._got_any_targets = False
        self._reset_requested = False
        self._stop = False

        self._visuals = self.sim.mesh_visuals() if gp("publish_visual") and gp("publish_robot_markers") else []
        self._visual_period = 1.0 / max(gp("visual_rate"), 1e-3)
        self._last_visual = -1e9
        if self._visuals:
            self.get_logger().info(f"publishing {len(self._visuals)} robot mesh markers on /microduck/visual")

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

    def _on_ball(self, response, side):
        with self._lock:
            response.success = self.sim.place_ball(side)
        response.message = f"ball in front of the {side} foot" if response.success else "no ball in scene"
        return response

    def _collect_world_geoms(self):
        """Primitive geoms of the scene (not the robot), drawn as RViz markers."""
        import mujoco
        m = self.sim.model
        trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        shapes = {int(mujoco.mjtGeom.mjGEOM_BOX): Marker.CUBE,
                  int(mujoco.mjtGeom.mjGEOM_CYLINDER): Marker.CYLINDER,
                  int(mujoco.mjtGeom.mjGEOM_SPHERE): Marker.SPHERE}
        out = []
        for g in range(m.ngeom):
            if m.body_rootid[m.geom_bodyid[g]] == trunk or int(m.geom_type[g]) not in shapes:
                continue
            if m.geom_group[g] > 2 or m.geom_rgba[g][3] == 0.0:
                continue
            out.append((g, shapes[int(m.geom_type[g])]))
        return out

    # ----------------------------------------------------------- main loop
    def _run(self):
        wall_start, sim_start = time.monotonic(), self.sim.time
        while rclpy.ok() and not self._stop:
            if self._reset_requested or (self.auto_reset and self.sim.fallen()):
                with self._lock:
                    self.sim.reset()
                    self._got_any_targets = False
                    self._last_visual = self._last_scan = self._last_qpos = -1e9
                self._reset_requested = False
                wall_start, sim_start = time.monotonic(), self.sim.time
                self.get_logger().info("simulation reset")

            t_k = _to_ns(self.sim.time)
            self._targets_event.clear()
            self._publish_state()

            if self.wait_for_policy and not self._got_any_targets:
                # Hold the robot frozen at its initial pose until a policy answers,
                # republishing everything so late subscribers (camera, RViz) see it.
                self._targets_event.wait(0.1)
                self._last_qpos = self._last_visual = -1e9
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
            fx, fy, fyaw = self.sim.footprint_pose()
            vis = None
            if (self._visuals or self._world_geoms) and st.time - self._last_visual >= self._visual_period:
                vis = ([self.sim.geom_pose(v.geom_id) @ v.T_geom_mesh for v in self._visuals],
                       [(g, shape, self.sim.geom_pose(g), self.sim.model.geom_size[g].copy(),
                         self.sim.model.geom_rgba[g].copy()) for g, shape in self._world_geoms])
                self._last_visual = st.time
            qpos = None
            if st.time - self._last_qpos >= self._state_period:
                qpos = self.sim.qpos()
                self._last_qpos = st.time
            ball = self.sim.ball_position()
            ranges = None
            if self.publish_scan and st.time - self._last_scan >= self.scan_period:
                ranges = self.sim.scan(self.scan_rays, self.scan_height, 0.05, self.scan_range_max)
                self._last_scan = st.time
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

        # Ground-truth odometry of the footprint (planar), twist in the footprint frame.
        c, s_ = np.cos(fyaw), np.sin(fyaw)
        vx_w, vy_w = float(st.trunk_lin_vel[0]), float(st.trunk_lin_vel[1])
        qz, qw = np.sin(fyaw / 2), np.cos(fyaw / 2)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.footprint_frame
        odom.pose.pose.position.x, odom.pose.pose.position.y = fx, fy
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = float(qz), float(qw)
        odom.twist.twist.linear.x = c * vx_w + s_ * vy_w
        odom.twist.twist.linear.y = -s_ * vx_w + c * vy_w
        odom.twist.twist.angular.z = float(st.trunk_ang_vel[2])
        self.pub_odom.publish(odom)

        t_fp = TransformStamped()
        t_fp.header = odom.header
        t_fp.child_frame_id = self.footprint_frame
        t_fp.transform.translation.x, t_fp.transform.translation.y = fx, fy
        t_fp.transform.rotation = odom.pose.pose.orientation
        # trunk relative to the footprint: height + the roll/pitch part of its attitude
        q_yaw_inv = np.array([qw, 0.0, 0.0, -qz])
        q_rel = _quat_mul(q_yaw_inv, np.array([w, x, y, z]))
        t_base = TransformStamped()
        t_base.header.stamp = stamp
        t_base.header.frame_id = self.footprint_frame
        t_base.child_frame_id = self.base_frame
        t_base.transform.translation.z = float(st.trunk_pos[2])
        r = t_base.transform.rotation
        r.w, r.x, r.y, r.z = map(float, q_rel)
        self.tf.sendTransform([t_fp, t_base])

        if ranges is not None:
            scan = LaserScan()
            scan.header.stamp = stamp
            scan.header.frame_id = self.scan_frame
            n = len(ranges)
            scan.angle_min = -np.pi
            scan.angle_increment = 2 * np.pi / n
            scan.angle_max = scan.angle_min + (n - 1) * scan.angle_increment
            scan.scan_time = self.scan_period
            scan.range_min = 0.05
            scan.range_max = self.scan_range_max
            scan.ranges = ranges.astype(float).tolist()
            self.pub_scan.publish(scan)

        if vis is not None:
            self._publish_visual(stamp, *vis)

        if qpos is not None:
            # [sim time, qpos...]: lets the camera node rebuild the exact scene.
            self.pub_qpos.publish(Float64MultiArray(data=[st.time] + qpos.tolist()))
        if ball is not None and qpos is not None:
            pt = PointStamped()
            pt.header.stamp = stamp
            pt.header.frame_id = self.odom_frame
            pt.point.x, pt.point.y, pt.point.z = map(float, ball)
            self.pub_ball.publish(pt)

    def _publish_visual(self, stamp, robot_poses, world):
        arr = MarkerArray()
        for i, (v, T) in enumerate(zip(self._visuals, robot_poses)):
            mk = Marker()
            mk.header.stamp = stamp
            mk.header.frame_id = self.odom_frame
            mk.ns = "microduck"
            mk.id = i
            mk.type = Marker.MESH_RESOURCE
            mk.action = Marker.ADD
            mk.mesh_resource = "file://" + v.mesh_file
            _set_pose(mk, T)
            mk.scale.x = mk.scale.y = mk.scale.z = 1.0
            r, g, b, a = v.rgba
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = float(r), float(g), float(b), float(a) or 1.0
            arr.markers.append(mk)
        for g, shape, T, size, rgba in world:
            mk = Marker()
            mk.header.stamp = stamp
            mk.header.frame_id = self.odom_frame
            # static scene geoms vs moving objects (the ball), so RViz can toggle them separately
            mk.ns = "scene" if self.sim.model.geom_bodyid[g] == 0 else "objects"
            mk.id = int(g)
            mk.type = shape
            mk.action = Marker.ADD
            _set_pose(mk, T)
            if shape == Marker.CUBE:
                mk.scale.x, mk.scale.y, mk.scale.z = (2 * float(v) for v in size[:3])
            elif shape == Marker.CYLINDER:
                mk.scale.x = mk.scale.y = 2 * float(size[0])
                mk.scale.z = 2 * float(size[1])
            else:
                mk.scale.x = mk.scale.y = mk.scale.z = 2 * float(size[0])
            mk.color.r, mk.color.g, mk.color.b = (float(v) for v in rgba[:3])
            mk.color.a = float(rgba[3])
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
