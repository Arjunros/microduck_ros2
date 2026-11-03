"""Head camera for the Microduck sim: RGB + depth from MuJoCo's `head_camera`.

Runs as its own process so rendering never slows the physics: it keeps a
kinematic replica of the scene, updated from /microduck/sim_qpos, and renders it.

Publishes  /camera/image_raw        sensor_msgs/Image  rgb8
           /camera/depth/image_raw  sensor_msgs/Image  32FC1, metres along the optical axis
           /camera/camera_info      sensor_msgs/CameraInfo (pinhole from the camera's fovy)
           all with frame_id head_camera_optical_frame (from the URDF) and the sim stamp.

GL backend (param gl_backend): auto tries egl, then glfw (needs a display), then osmesa.
"""

import os
import subprocess
import sys
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64MultiArray

_PROBE = """
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 1"/><geom size=".1"/>'
                                   '<camera name="c" pos="0 0 1"/></worldbody></mujoco>')
r = mujoco.Renderer(m, 8, 8)
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r.update_scene(d, camera="c"); r.render()
print("OK")
"""


def pick_gl_backend(requested="auto", log=print):
    """Choose MUJOCO_GL before mujoco is imported (it only reads it once)."""
    if requested != "auto":
        return requested
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    candidates = ["egl"] + (["glfw"] if os.environ.get("DISPLAY") else []) + ["osmesa"]
    for backend in candidates:
        try:
            out = subprocess.run([sys.executable, "-c", _PROBE], env=dict(os.environ, MUJOCO_GL=backend),
                                 capture_output=True, text=True, timeout=60)
            if "OK" in out.stdout:
                return backend
            log(f"MUJOCO_GL={backend} unavailable: {(out.stderr.strip().splitlines() or ['?'])[-1][:120]}")
        except subprocess.TimeoutExpired:
            log(f"MUJOCO_GL={backend} timed out")
    return None


def _stamp(t):
    from builtin_interfaces.msg import Time
    ns = int(round(t * 1e9))
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def camera_info_for(width, height, fovy_deg, frame_id):
    """Pinhole intrinsics for a MuJoCo camera (vertical fov, square pixels, centred)."""
    f = (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    info = CameraInfo()
    info.header.frame_id = frame_id
    info.width, info.height = width, height
    info.distortion_model = "plumb_bob"
    info.d = [0.0] * 5
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    info.k = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info


class CameraNode(Node):
    def __init__(self):
        super().__init__("microduck_camera")
        p = self.declare_parameter
        p("microduck_rl_path", os.environ.get("MICRODUCK_RL_PATH", os.path.expanduser("~/microduck_rl")))
        p("scene", "flat")
        p("add_ball", False)
        p("ball_x", 0.6)
        p("ball_y", 0.0)
        p("camera", "head_camera")
        p("width", 640)    # the real sensor: 640x360 @ 15 fps (Pollen sim/camera.py)
        p("height", 360)
        p("rate", 15.0)
        p("depth", True)
        p("gl_backend", "auto")
        p("near_clip", 0.02)   # m. MuJoCo scales clip planes by scene size otherwise
        p("far_clip", 15.0)    # (14 cm / 710 m in the apartment: z-fighting everywhere)
        p("frame_id", "head_camera_optical_frame")
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        backend = pick_gl_backend(gp("gl_backend"), log=lambda m: self.get_logger().warn(m))
        if backend is None:
            raise RuntimeError("no working OpenGL backend for MuJoCo (tried egl, glfw, osmesa)")
        os.environ["MUJOCO_GL"] = backend
        import mujoco  # noqa: E402  (after MUJOCO_GL is set)
        from .engine import load_scene_spec
        from .urdf import resolve_scene
        self.mujoco = mujoco

        scene = resolve_scene(gp("scene"), gp("microduck_rl_path"))
        spec = load_scene_spec(scene, gp("add_ball"), (gp("ball_x"), gp("ball_y")))
        self.model = spec.compile()
        ext = float(self.model.stat.extent)
        self.model.vis.map.znear = float(gp("near_clip")) / ext
        self.model.vis.map.zfar = float(gp("far_clip")) / ext
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, gp("camera"))
        if self.cam < 0:
            raise RuntimeError(f"camera '{gp('camera')}' not in {scene}")
        self.w, self.h = int(gp("width")), int(gp("height"))
        self.with_depth = bool(gp("depth"))
        self.frame_id = gp("frame_id")
        self.info = camera_info_for(self.w, self.h, float(self.model.cam_fovy[self.cam]), self.frame_id)

        self._lock = threading.Lock()
        self._latest = None          # (t, qpos)
        self._rendered_t = None
        self._renderer = None        # created on the render thread (GL contexts are thread-bound)

        self.pub_rgb = self.create_publisher(Image, "/camera/image_raw", qos_profile_sensor_data)
        self.pub_depth = self.create_publisher(Image, "/camera/depth/image_raw", qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, "/camera/camera_info", qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, "/microduck/sim_qpos", self._on_qpos, 1)
        self._period = 1.0 / max(float(gp("rate")), 1e-3)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(f"head camera {self.w}x{self.h} @ {gp('rate')} Hz, "
                               f"fovy {self.model.cam_fovy[self.cam]:.0f} deg, MUJOCO_GL={backend}")

    def _on_qpos(self, msg):
        data = np.asarray(msg.data)
        if len(data) != self.model.nq + 1:
            self.get_logger().error(f"sim_qpos has {len(data) - 1} values, model has nq={self.model.nq}: "
                                    "camera and sim must use the same scene/add_ball", throttle_duration_sec=5.0)
            return
        with self._lock:
            self._latest = (float(data[0]), data[1:])

    def _render_loop(self):
        mj = self.mujoco
        while not self._stop.is_set():
            self._stop.wait(self._period)
            with self._lock:
                latest = self._latest
            if latest is None or latest[0] == self._rendered_t:
                continue
            t, qpos = latest
            self.data.qpos[:] = qpos
            mj.mj_forward(self.model, self.data)
            if self._renderer is None:
                self._renderer = mj.Renderer(self.model, self.h, self.w)
            r = self._renderer
            r.disable_depth_rendering()
            r.update_scene(self.data, camera=self.cam)
            rgb = r.render()
            depth = None
            if self.with_depth:
                r.enable_depth_rendering()
                r.update_scene(self.data, camera=self.cam)
                depth = r.render().astype(np.float32)
                r.disable_depth_rendering()
            self._rendered_t = t
            if not rclpy.ok():
                break
            try:
                self._publish(t, rgb, depth)
            except Exception:  # context shut down between the check and the publish
                break

    def _publish(self, t, rgb, depth):
        stamp = _stamp(t)
        img = Image()
        img.header.stamp, img.header.frame_id = stamp, self.frame_id
        img.height, img.width = rgb.shape[:2]
        img.encoding, img.step = "rgb8", rgb.shape[1] * 3
        img.data = rgb.tobytes()
        self.pub_rgb.publish(img)
        self.info.header.stamp = stamp
        self.pub_info.publish(self.info)
        if depth is not None:
            dm = Image()
            dm.header = img.header
            dm.height, dm.width = depth.shape
            dm.encoding, dm.step = "32FC1", depth.shape[1] * 4
            dm.data = depth.tobytes()
            self.pub_depth.publish(dm)

    def destroy_node(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        # Skip interpreter teardown: some GL drivers raise from the renderer's
        # finalizer at exit, after all the useful work is done.
        os._exit(0)


if __name__ == "__main__":
    main()
