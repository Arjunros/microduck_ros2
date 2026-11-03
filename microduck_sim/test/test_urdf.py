"""The generated URDF must reproduce MuJoCo's kinematics exactly."""

import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from microduck_sim.engine import load_scene_spec  # noqa: E402
from microduck_sim.urdf import mjcf_to_urdf, urdf_for_scene  # noqa: E402

RL_PATH = os.path.expanduser(os.environ.get("MICRODUCK_RL_PATH", "~/microduck_rl"))
SCENE = os.path.join(RL_PATH, "src/mjlab_microduck/robot/microduck/scene.xml")
pytestmark = pytest.mark.skipif(not os.path.isfile(SCENE), reason="microduck_rl clone not found")


def _rot(axis, a):
    x, y, z = axis / np.linalg.norm(axis)
    c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _origin(el):
    T = np.eye(4)
    if el is None:
        return T
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    r, p, y = (float(v) for v in el.get("rpy", "0 0 0").split())
    T[:3, :3] = _rot(np.array([0, 0, 1.]), y) @ _rot(np.array([0, 1., 0]), p) @ _rot(np.array([1., 0, 0]), r)
    T[:3, 3] = xyz
    return T


def urdf_fk(urdf, q, T_root):
    """World transforms of every link, from the URDF alone (independent of the generator)."""
    robot = ET.fromstring(urdf)
    links = {l.get("name") for l in robot.findall("link")}
    joints = {j.find("child").get("link"): j for j in robot.findall("joint")}
    roots = links - set(joints)
    assert len(roots) == 1, roots
    out = {next(iter(roots)): T_root}

    def pose(link):
        if link in out:
            return out[link]
        j = joints[link]
        T = pose(j.find("parent").get("link")) @ _origin(j.find("origin"))
        if j.get("type") in ("revolute", "continuous"):
            R = np.eye(4)
            R[:3, :3] = _rot(np.array([float(v) for v in j.find("axis").get("xyz").split()]), q[j.get("name")])
            T = T @ R
        out[link] = T
        return T

    for l in links:
        pose(l)
    return out


def test_urdf_matches_mujoco_kinematics():
    spec = load_scene_spec(SCENE)
    m = spec.compile()
    urdf = mjcf_to_urdf(m, spec, os.path.dirname(SCENE))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(1)
    for _ in range(10):
        d.qpos[:] = 0
        d.qpos[0:3] = rng.normal(size=3)
        quat = rng.normal(size=4)
        d.qpos[3:7] = quat / np.linalg.norm(quat)
        names = []
        for j in range(m.njnt):
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                lo, hi = m.jnt_range[j]
                d.qpos[m.jnt_qposadr[j]] = rng.uniform(lo, hi)
                names.append(j)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        q = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): d.qpos[m.jnt_qposadr[j]] for j in names}
        trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        T_root = np.eye(4)
        T_root[:3, :3] = d.xmat[trunk].reshape(3, 3)
        T_root[:3, 3] = d.xpos[trunk]
        fk = urdf_fk(urdf, q, T_root)
        for b in range(1, m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            np.testing.assert_allclose(fk[name][:3, 3], d.xpos[b], atol=1e-7, err_msg=name)
            np.testing.assert_allclose(fk[name][:3, :3], d.xmat[b].reshape(3, 3), atol=1e-7, err_msg=name)
        cam = 0
        np.testing.assert_allclose(fk["head_camera_mujoco"][:3, 3], d.cam_xpos[cam], atol=1e-7)
        np.testing.assert_allclose(fk["head_camera_mujoco"][:3, :3], d.cam_xmat[cam].reshape(3, 3), atol=1e-7)
        # optical frame: +z = camera viewing direction (MuJoCo -z), +y = image down (MuJoCo -y)
        R_mj, R_opt = d.cam_xmat[cam].reshape(3, 3), fk["head_camera_optical_frame"][:3, :3]
        np.testing.assert_allclose(R_opt[:, 2], -R_mj[:, 2], atol=1e-7)
        np.testing.assert_allclose(R_opt[:, 1], -R_mj[:, 1], atol=1e-7)
        imu = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
        np.testing.assert_allclose(fk["imu"][:3, 3], d.site_xpos[imu], atol=1e-7)


def test_urdf_structure_and_meshes():
    urdf = urdf_for_scene(SCENE)
    robot = ET.fromstring(urdf)
    joints = {j.get("name"): j.get("type") for j in robot.findall("joint")}
    sides = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
    names = [f"{s}_{j}" for s in ("left", "right") for j in sides]
    names += ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
    for n in names:                                        # /joint_states names drive the TF tree
        assert joints.get(n) == "revolute", n
    meshes = [m.get("filename") for m in robot.iter("mesh")]
    assert len(meshes) > 20 and all(f.startswith("file://") and os.path.isfile(f[7:]) for f in meshes)
    assert robot.find("link[@name='trunk_base']") is not None
