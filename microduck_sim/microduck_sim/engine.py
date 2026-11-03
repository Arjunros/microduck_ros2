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

    def fallen(self, min_height=0.06):
        return float(self.data.xpos[self._trunk_id][2]) < min_height

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
