"""Policy-side tests: observation layout, upstream parity, ONNX loading.

Parity tests need a clone of microduck_rl (MICRODUCK_RL_PATH, default
~/microduck_rl) and mujoco; they are skipped otherwise.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

from microduck_policy.constants import DEFAULT_POSE, JOINT_NAMES, OBS_DIM
from microduck_policy.observation import build_observation, make_command, projected_gravity
from microduck_policy.policy import MicroduckController, OnnxPolicy, PolicyLoadError

RL_PATH = os.path.expanduser(os.environ.get("MICRODUCK_RL_PATH", "~/microduck_rl"))
SCENE = os.path.join(RL_PATH, "src/mjlab_microduck/robot/microduck/scene.xml")
INFER = os.path.join(RL_PATH, "scripts/infer_policy.py")
needs_rl = pytest.mark.skipif(not os.path.isfile(INFER), reason="microduck_rl clone not found")


def _make_onnx(path, weight=0.0, bias=0.0, in_dim=OBS_DIM, out_dim=14):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    W = np.full((in_dim, out_dim), weight, dtype=np.float32)
    B = np.full((out_dim,), bias, dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["obs", "W", "B"], ["actions"])],
        "linear",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, in_dim])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, out_dim])],
        [helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten()),
         helper.make_tensor("B", TensorProto.FLOAT, B.shape, B)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


def test_layout_and_upright_gravity():
    obs = build_observation(np.zeros(3), (1, 0, 0, 0), DEFAULT_POSE, np.zeros(14), np.zeros(14),
                            make_command((0.1, 0.0, 0.5)))
    assert obs.shape == (OBS_DIM,) == (61,)
    np.testing.assert_allclose(obs[3:6], [0, 0, -1])
    np.testing.assert_allclose(obs[6:20], 0)          # at default pose
    np.testing.assert_allclose(obs[48:51], [0.1, 0, 0.5])


def test_projected_gravity_rolled_90deg():
    # +90 deg about x: world -z appears along body -y... check against a rotation matrix.
    c = np.cos(np.pi / 4)
    q = (c, c, 0.0, 0.0)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    np.testing.assert_allclose(projected_gravity(q), R.T @ np.array([0, 0, -1.0]), atol=1e-6)


@needs_rl
def test_default_pose_matches_stand_keyframe():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(SCENE)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0])) for i in range(m.nu)]
    assert tuple(names) == JOINT_NAMES
    qadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
    np.testing.assert_allclose(m.key_qpos[kid][qadr], DEFAULT_POSE, atol=1e-4)


@needs_rl
def test_obs_parity_with_upstream_infer_policy(tmp_path):
    """Our observation == PolicyInference.get_observations() on random states."""
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    spec = importlib.util.spec_from_file_location("upstream_infer", INFER)
    up = importlib.util.module_from_spec(spec)
    sys.modules["upstream_infer"] = up
    spec.loader.exec_module(up)

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    onnx_path = _make_onnx(str(tmp_path / "zero.onnx"))
    ref = up.PolicyInference(model, data, walking_onnx_path=onnx_path,
                             use_projected_gravity=True, new_cmd_obs=True)

    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    gyro_adr = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")]
    qidx, vidx = ref.joint_qpos_indices, ref.joint_qvel_indices
    rng = np.random.default_rng(0)
    for _ in range(25):
        data.qpos[:] = 0
        data.qpos[2] = 0.12
        quat = rng.normal(size=4)
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        data.qpos[qidx] = DEFAULT_POSE + rng.normal(scale=0.2, size=14)
        data.qvel[:] = rng.normal(scale=1.0, size=model.nv)
        mujoco.mj_forward(model, data)

        last = rng.normal(size=14).astype(np.float32)
        ref.last_action = last.copy()
        ref.vel_cmd = rng.uniform(-0.3, 0.3, size=3).astype(np.float32)
        ref.head_offset = rng.uniform(-0.3, 0.3, size=4).astype(np.float32)
        ref.body_cmd = rng.uniform(-0.02, 0.02, size=6).astype(np.float32)
        ref.current_policy = "walking"
        ref._update_command()
        expected = ref.get_observations()

        got = build_observation(
            data.sensordata[gyro_adr:gyro_adr + 3], data.xquat[trunk], data.qpos[qidx], data.qvel[vidx],
            last, make_command(ref.vel_cmd, ref.head_offset, ref.body_cmd))
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_onnx_shape_check(tmp_path):
    bad = _make_onnx(str(tmp_path / "bad.onnx"), in_dim=51)
    with pytest.raises(PolicyLoadError):
        OnnxPolicy(bad)


def test_manifest_action_scale_and_recurrent_refusal(tmp_path):
    _make_onnx(str(tmp_path / "walk.onnx"))
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"schema_version": 2, "policies": [{"file": "walk.onnx", "action_scale": 0.5},
                                           {"file": "lstm.onnx", "model_api": 2}]}))
    assert OnnxPolicy(str(tmp_path / "walk.onnx")).action_scale == 0.5
    assert OnnxPolicy(str(tmp_path / "walk.onnx"), action_scale=0.25).action_scale == 0.25
    _make_onnx(str(tmp_path / "lstm.onnx"))
    with pytest.raises(PolicyLoadError):
        OnnxPolicy(str(tmp_path / "lstm.onnx"))


def test_controller_targets_and_switching(tmp_path):
    walk = OnnxPolicy(_make_onnx(str(tmp_path / "w.onnx"), bias=0.1))
    stand = OnnxPolicy(_make_onnx(str(tmp_path / "s.onnx"), bias=-0.1))
    ctrl = MicroduckController(walk, stand)

    ctrl.set_command(make_command((0.2, 0, 0)))
    assert ctrl.active_name == "walking"
    targets, obs, action = ctrl.step(np.zeros(3), (1, 0, 0, 0), DEFAULT_POSE, np.zeros(14))
    np.testing.assert_allclose(targets, DEFAULT_POSE + 0.1, atol=1e-6)
    np.testing.assert_allclose(ctrl.last_action, 0.1, atol=1e-6)

    ctrl.set_command(make_command((0.01, 0, 0)))
    assert ctrl.active_name == "standing"
    assert np.all(ctrl.command[0:3] == 0)
    _, obs, _ = ctrl.step(np.zeros(3), (1, 0, 0, 0), DEFAULT_POSE, np.zeros(14))
    np.testing.assert_allclose(obs[34:48], 0.1, atol=1e-6)   # last action carried across switch
