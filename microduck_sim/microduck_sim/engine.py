"""MuJoCo simulation of Microduck, with the BAM XL330 actuator model. No ROS.

The actuator setup is ported from microduck_rl/scripts/infer_policy.py
(load_mujoco_with_bam), which mirrors what training does on the GPU. Without
BAM the XML position actuators are used instead; policies are NOT trained on
those, so expect worse walking.
"""

import logging
import os
from dataclasses import dataclass

import mujoco
import numpy as np

log = logging.getLogger("microduck_sim")

# BAM M6 defaults, must mirror _BAM_ACTUATOR_KWARGS in microduck_rl.
BAM_MOTOR_NAME = "xl330"
BAM_MODEL = "m6"
BAM_KP_FW = 200.0
BAM_VIN = 7.4
BAM_VIN_DROP_GAIN = 0.1
BAM_VIN_MIN = 6.0
BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)

TRUNK_BODY = "trunk_base"
TRUNK_FREEJOINT = "trunk_base_freejoint"
GYRO_SENSOR = "imu_ang_vel"
BALL_FREEJOINT = "ball_free"

SCENE_DIR_REL = "src/mjlab_microduck/robot/microduck"
# Short names for the scenes shipped with microduck_rl.
SCENES = {"flat": "scene.xml", "ball": "scene_ball.xml", "apartment": "scene_apartment.xml"}

# STAND2 pose, used when the scene has no STAND keyframe (scene_ball.xml has
# none: the ball changes nq). Same values as microduck_policy.constants.
STAND_POSE = np.array([
    0.0, -0.08726646259971647, -0.457924, -0.004940, 0.452984,
    0.3490658503988659, 0.3490658503988659, 0.0, 0.0,
    0.0, 0.08726646259971647, 0.457924, 0.004940, -0.452984,
])

# Ball placement for the kick policies, in the robot's yaw frame. Must match
# microduck_rl's reset_ball_in_front_of_foot (infer_policy.py BALL_OFFSET_*).
BALL_OFFSET_X = 0.09
BALL_OFFSET_ABS_Y = 0.042
BALL_RADIUS = 0.035


def bam_available():
    try:
        import bam.model  # noqa: F401
        import bam.mujoco  # noqa: F401
        return True
    except Exception:  # ImportError, or Python version too old for bam
        return False


@dataclass
class RobotState:
    time: float
    joint_names: tuple
    joint_pos: np.ndarray       # actuated joints, actuator order [rad]
    joint_vel: np.ndarray       # [rad/s]
    joint_effort: np.ndarray    # actuator force [Nm]
    gyro: np.ndarray            # IMU gyro, sensor frame [rad/s]
    accel: np.ndarray           # IMU accelerometer, sensor frame [m/s^2]
    trunk_pos: np.ndarray       # world [m]
    trunk_quat_wxyz: np.ndarray # world
    trunk_lin_vel: np.ndarray   # world [m/s]
    trunk_ang_vel: np.ndarray   # world [rad/s]


@dataclass
class MeshVisual:
    geom_id: int
    mesh_file: str              # absolute path to the original STL
    T_geom_mesh: np.ndarray     # 4x4: original-mesh frame expressed in the geom frame
    rgba: tuple


class MicroduckSim:
    def __init__(self, scene_path, use_bam=True, vin=BAM_VIN, vin_drop_gain=BAM_VIN_DROP_GAIN,
                 kp_fw=BAM_KP_FW, timestep=0.005, decimation=4, init_z=0.125,
                 init_keyframe="STAND"):
        if not os.path.isfile(scene_path):
            raise FileNotFoundError(
                f"scene not found: {scene_path}. Run scripts/fetch_microduck_assets.sh "
                "or set the microduck_rl_path parameter."
            )
        self.scene_path = os.path.abspath(scene_path)
        self.decimation = int(decimation)
        self.init_z = float(init_z)
        self.bam_ctrl = None

        if use_bam and not bam_available():
            log.warning("BAM (better-actuator-models) not importable: falling back to XML position "
                        "actuators. Policies are trained against BAM, so walking will degrade. "
                        "BAM needs Python 3.12 (ROS 2 Jazzy).")
            use_bam = False
        self.use_bam = use_bam

        self.spec = mujoco.MjSpec.from_file(self.scene_path)
        if use_bam:
            self._load_with_bam(vin, vin_drop_gain, kp_fw, timestep)
        else:
            self.model = self.spec.compile()
            self.model.opt.timestep = timestep
            self.data = mujoco.MjData(self.model)

        m = self.model
        self.joint_names = tuple(
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0]))
            for i in range(m.nu)
        )
        self._qpos_idx = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self._qvel_idx = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self._name_to_act = {n: i for i, n in enumerate(self.joint_names)}

        self._trunk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TRUNK_BODY)
        fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, TRUNK_FREEJOINT)
        if self._trunk_id < 0 or fj < 0:
            raise ValueError(f"{scene_path}: no '{TRUNK_BODY}' body / '{TRUNK_FREEJOINT}' joint")
        self._free_qpos = int(m.jnt_qposadr[fj])
        self._free_qvel = int(m.jnt_dofadr[fj])
        self._gyro_adr = self._sensor_adr(GYRO_SENSOR)
        self._accel_adr = self._sensor_adr("imu_accel")
        bj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, BALL_FREEJOINT)
        self._ball_qpos = int(m.jnt_qposadr[bj]) if bj >= 0 else None
        self._ball_qvel = int(m.jnt_dofadr[bj]) if bj >= 0 else None
        # Lidar sees only the world (geom group 0); the robot's geoms are in
        # groups 2-3, so rays pass through it.
        self._world_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self._scan_cache = {}

        self.default_pose = self._keyframe_ctrl(init_keyframe)
        self.targets = self.default_pose.copy()
        self.reset()

    # ------------------------------------------------------------------ setup
    def _load_with_bam(self, vin, vin_drop_gain, kp_fw, timestep):
        from bam.model import load_model
        from bam.mujoco import MujocoController

        bam_model = load_model(motor_name=BAM_MOTOR_NAME, model=BAM_MODEL)
        bam_model.actuator.kp = kp_fw
        bam_model.actuator.vin = vin
        bam_model.actuator.max_current = None  # training runs without the current limiter
        kt, R = bam_model.kt.value, bam_model.R.value
        force_limit = bam_model.actuator.vin * kt / R

        names = []
        for act in self.spec.actuators:
            tgt = act.target
            tgt_name = tgt.name if hasattr(tgt, "name") else str(tgt)
            if tgt_name.startswith("passive_"):
                continue
            act.set_to_motor()
            act.forcelimited = True
            act.forcerange = (-force_limit, force_limit)
            act.ctrllimited = False
            act.gear = [1.0, 0, 0, 0, 0, 0]
            names.append(act.name)
            for joint in self.spec.joints:
                if joint.name == tgt_name:
                    joint.damping = np.zeros((3, 1))
                    joint.frictionloss = 0.0
                    joint.solref_friction = BAM_STIFF_SOLREF_FRICTION
                    joint.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
                    break

        self.model = self.spec.compile()
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.bam_ctrl = MujocoController(
            bam_model, names, self.model, self.data,
            vin_drop_gain=(vin_drop_gain if vin_drop_gain and vin_drop_gain > 0 else None),
            vin_min=BAM_VIN_MIN,
        )

    def _sensor_adr(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(self.model.sensor_adr[sid]) if sid >= 0 else None

    def _keyframe_ctrl(self, name):
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid < 0:
            if self.model.nu == len(STAND_POSE):
                return STAND_POSE.copy()
            log.warning("keyframe '%s' missing; default pose = zeros", name)
            return np.zeros(self.model.nu)
        # Keyframe ctrl is in position-actuator units (joint angles) in the XML,
        # even when BAM has turned the actuators into torque motors.
        qpos = self.model.key_qpos[kid]
        return qpos[self._qpos_idx].astype(np.float64).copy()

    # --------------------------------------------------------------- control
    @property
    def control_dt(self):
        return self.decimation * self.model.opt.timestep

    @property
    def time(self):
        return float(self.data.time)

    def reset(self, xy=(0.0, 0.0), yaw=0.0):
        mujoco.mj_resetData(self.model, self.data)
        q = self.data.qpos
        a = self._free_qpos
        q[a:a + 3] = [xy[0], xy[1], self.init_z]
        q[a + 3:a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        q[self._qpos_idx] = self.default_pose
        self.targets = self.default_pose.copy()
        if self.bam_ctrl is not None:
            self.bam_ctrl.reset(self.data.qpos)
        self._apply_targets()
        mujoco.mj_forward(self.model, self.data)

    def set_targets(self, positions, names=None):
        """Set joint position targets [rad], in actuator order or by name."""
        positions = np.asarray(positions, dtype=np.float64)
        if names:
            for n, p in zip(names, positions):
                i = self._name_to_act.get(n)
                if i is not None:
                    self.targets[i] = p
        else:
            self.targets[:] = positions.reshape(self.model.nu)
        self._apply_targets()

    def _apply_targets(self):
        if self.bam_ctrl is not None:
            self.bam_ctrl.q_target[:] = self.targets
        else:
            self.data.ctrl[:] = self.targets

    def step(self):
        """Advance one control period (decimation physics steps)."""
        for _ in range(self.decimation):
            if self.bam_ctrl is not None:
                self.bam_ctrl.update()
            mujoco.mj_step(self.model, self.data)

    def fallen(self, min_height=0.06, max_tilt_deg=60.0):
        """Trunk low, or tilted past max_tilt_deg (height alone misses side flops)."""
        up_z = float(self.data.xmat[self._trunk_id][8])  # R[2,2]: trunk z-axis . world z
        return (float(self.data.xpos[self._trunk_id][2]) < min_height
                or up_z < np.cos(np.radians(max_tilt_deg)))

    # ------------------------------------------------------------------ ball
    @property
    def has_ball(self):
        return self._ball_qpos is not None

    def place_ball(self, side):
        """Put the ball in front of the left/right foot, as the kick training reset does."""
        if not self.has_ball:
            return False
        x, y, yaw = self.footprint_pose()
        off_y = BALL_OFFSET_ABS_Y if side == "left" else -BALL_OFFSET_ABS_Y
        bx = x + np.cos(yaw) * BALL_OFFSET_X - np.sin(yaw) * off_y
        by = y + np.sin(yaw) * BALL_OFFSET_X + np.cos(yaw) * off_y
        a, v = self._ball_qpos, self._ball_qvel
        self.data.qpos[a:a + 7] = [bx, by, BALL_RADIUS, 1, 0, 0, 0]
        self.data.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return True

    # ----------------------------------------------------------- navigation
    def footprint_pose(self):
        """(x, y, yaw) of the trunk projected on the floor."""
        p = self.data.xpos[self._trunk_id]
        R = self.data.xmat[self._trunk_id].reshape(3, 3)
        return float(p[0]), float(p[1]), float(np.arctan2(R[1, 0], R[0, 0]))

    def scan(self, n_rays=360, height=0.12, range_min=0.05, range_max=6.0):
        """Virtual planar lidar: level rays at `height` above the floor (z=0),
        centred on the trunk and rotating with its yaw. Returns ranges [m],
        inf where nothing was hit within range_max (REP 117).

        Microduck has no lidar; this stands in for perception so Nav2 can run.
        """
        x, y, yaw = self.footprint_pose()
        key = n_rays
        if key not in self._scan_cache:
            ang = -np.pi + np.arange(n_rays) * (2 * np.pi / n_rays)
            self._scan_cache[key] = ang
        ang = self._scan_cache[key] + yaw
        vec = np.zeros((n_rays, 3))
        vec[:, 0], vec[:, 1] = np.cos(ang), np.sin(ang)
        pnt = np.array([x, y, height], dtype=np.float64)
        geomid = np.zeros(n_rays, dtype=np.int32)
        dist = np.zeros(n_rays, dtype=np.float64)
        try:
            mujoco.mj_multiRay(self.model, self.data, pnt, vec.reshape(-1), self._world_groups, 1, -1,
                               geomid, dist, None, n_rays, range_max)
        except TypeError:  # older mujoco without the `normal` argument
            mujoco.mj_multiRay(self.model, self.data, pnt, vec.reshape(-1), self._world_groups, 1, -1,
                               geomid, dist, n_rays, range_max)
        ranges = np.where((geomid < 0) | (dist < 0) | (dist > range_max), np.inf, dist)
        ranges = np.where(ranges < range_min, -np.inf, ranges)
        return ranges

    # ----------------------------------------------------------------- state
    def state(self):
        d = self.data
        a, v = self._free_qpos, self._free_qvel
        gyro = d.sensordata[self._gyro_adr:self._gyro_adr + 3].copy() if self._gyro_adr is not None \
            else np.zeros(3)
        accel = d.sensordata[self._accel_adr:self._accel_adr + 3].copy() if self._accel_adr is not None \
            else np.zeros(3)
        return RobotState(
            time=float(d.time),
            joint_names=self.joint_names,
            joint_pos=d.qpos[self._qpos_idx].copy(),
            joint_vel=d.qvel[self._qvel_idx].copy(),
            joint_effort=d.actuator_force.copy(),
            gyro=gyro,
            accel=accel,
            trunk_pos=d.xpos[self._trunk_id].copy(),
            # Same quantity Pollen's rehearsal uses for projected gravity.
            trunk_quat_wxyz=d.xquat[self._trunk_id].copy(),
            trunk_lin_vel=d.qvel[v:v + 3].copy(),
            trunk_ang_vel=d.qvel[v + 3:v + 6].copy(),
        )

    # --------------------------------------------------------------- visuals
    def mesh_visuals(self, groups=(0, 1, 2)):
        """Visual mesh geoms, with the transform needed to draw the ORIGINAL STL.

        MuJoCo recentres each mesh on load (mesh_pos / mesh_quat). Drawing the
        file as-is at the geom pose would be offset, so we return the inverse
        of that recentring as T_geom_mesh.
        """
        m = self.model
        meshdir = self.spec.meshdir or ""
        base = os.path.dirname(self.scene_path)
        files = {}
        for mesh in self.spec.meshes:
            path = mesh.file
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(base, meshdir, path))
            files[mesh.name] = path

        visuals = []
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if m.geom_group[g] not in groups:
                continue
            # Skip pure collision geoms (no rendering contribution).
            if m.geom_rgba[g][3] == 0.0:
                continue
            mesh_id = int(m.geom_dataid[g])
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            path = files.get(name)
            if not path or not os.path.isfile(path) or not path.lower().endswith(".stl"):
                continue
            # MuJoCo stores v_mj = R(mesh_quat)^T (v_file - mesh_pos)
            # => v_file = R v_mj + mesh_pos, so T_geom_file = inv([R | p]).
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.mesh_quat[mesh_id])
            R = R.reshape(3, 3)
            p = m.mesh_pos[mesh_id]
            T = np.eye(4)
            T[:3, :3] = R.T
            T[:3, 3] = -R.T @ p
            mat_id = int(m.geom_matid[g])
            rgba = tuple(m.mat_rgba[mat_id]) if mat_id >= 0 else tuple(m.geom_rgba[g])
            visuals.append(MeshVisual(g, path, T, rgba))
        return visuals

    def geom_pose(self, geom_id):
        """World pose of a geom as a 4x4 matrix."""
        T = np.eye(4)
        T[:3, :3] = self.data.geom_xmat[geom_id].reshape(3, 3)
        T[:3, 3] = self.data.geom_xpos[geom_id]
        return T
