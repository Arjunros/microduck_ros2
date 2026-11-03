import os

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from microduck_sim.engine import MicroduckSim, bam_available  # noqa: E402

RL_PATH = os.path.expanduser(os.environ.get("MICRODUCK_RL_PATH", "~/microduck_rl"))
SCENE = os.path.join(RL_PATH, "src/mjlab_microduck/robot/microduck/scene.xml")
pytestmark = pytest.mark.skipif(not os.path.isfile(SCENE), reason="microduck_rl clone not found")


@pytest.mark.parametrize("use_bam", [True, False])
def test_reset_step_state(use_bam):
    if use_bam and not bam_available():
        pytest.skip("BAM not installed")
    sim = MicroduckSim(SCENE, use_bam=use_bam)
    assert sim.use_bam == use_bam
    assert len(sim.joint_names) == 14
    st = sim.state()
    assert st.trunk_pos[2] == pytest.approx(0.125)
    np.testing.assert_allclose(st.joint_pos, sim.default_pose)
    sim.step()
    assert sim.time == pytest.approx(sim.control_dt)
    st = sim.state()
    assert np.all(np.isfinite(st.joint_pos)) and np.all(np.isfinite(st.gyro))

    sim.set_targets([0.1], names=["head_yaw"])
    assert sim.targets[sim.joint_names.index("head_yaw")] == pytest.approx(0.1)
    t = sim.time
    sim.reset()
    assert sim.time == pytest.approx(t + sim.control_dt)   # clock keeps running (TF/Nav2 hate time travel)
    assert sim.state().trunk_pos[2] == pytest.approx(0.125)
    np.testing.assert_allclose(sim.targets, sim.default_pose)
    sim.reset(keep_time=False)
    assert sim.time == 0.0


def test_mesh_transform_reproduces_mujoco_vertices():
    """Drawing the original STL at geom_pose @ T_geom_mesh must land on MuJoCo's mesh."""
    stl = pytest.importorskip("stl")
    sim = MicroduckSim(SCENE, use_bam=False)
    m = sim.model
    vis = sim.mesh_visuals()
    assert len(vis) > 20
    for v in vis:
        mid = int(m.geom_dataid[v.geom_id])
        adr, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        vmj = m.mesh_vert[adr:adr + n]
        raw = stl.mesh.Mesh.from_file(v.mesh_file).vectors.reshape(-1, 3)
        pts = (v.T_geom_mesh[:3, :3] @ raw.T).T + v.T_geom_mesh[:3, 3]
        np.testing.assert_allclose(pts.min(0), vmj.min(0), atol=1e-4)
        np.testing.assert_allclose(pts.max(0), vmj.max(0), atol=1e-4)


APARTMENT = os.path.join(RL_PATH, "src/mjlab_microduck/robot/microduck/scene_apartment.xml")
BALL = os.path.join(RL_PATH, "src/mjlab_microduck/robot/microduck/scene_ball.xml")


def test_scan_ignores_robot_and_sees_corridor_walls():
    sim = MicroduckSim(APARTMENT, use_bam=False)
    r = sim.scan(n_rays=360)
    assert np.all(np.isfinite(r))                  # enclosed: every ray hits something
    assert r.min() > 0.3                           # robot's own geoms are not seen
    # The duck starts in the north-south corridor: far along y, close along x.
    assert r[90] > 2.5 and r[270] > 2.5            # +-90 deg (index = deg + 180)
    assert r[180] < 1.0 and r[0] < 1.0


def test_ball_scene_and_placement():
    sim = MicroduckSim(BALL, use_bam=False)        # no STAND keyframe: falls back to STAND_POSE
    np.testing.assert_allclose(sim.default_pose, MicroduckSim(SCENE, use_bam=False).default_pose)
    assert sim.has_ball and sim.place_ball("right")
    a = sim._ball_qpos
    np.testing.assert_allclose(sim.data.qpos[a:a + 3], [0.09, -0.042, 0.035])


def test_fallen_detects_tilt():
    sim = MicroduckSim(SCENE, use_bam=False)
    assert not sim.fallen()
    a = sim._free_qpos
    sim.data.qpos[a + 3:a + 7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]   # rolled 90 deg
    mujoco.mj_forward(sim.model, sim.data)
    assert sim.fallen()


def test_apartment_map(tmp_path):
    from microduck_sim.mapgen import main as make_map
    make_map(["--scene", "apartment", "--microduck-rl-path", RL_PATH, "--out", str(tmp_path / "apt")])
    text = (tmp_path / "apt.yaml").read_text()
    assert "image: apt.pgm" in text and "resolution: 0.05" in text
    raw = (tmp_path / "apt.pgm").read_bytes().split(b"\n", 3)
    w, h = map(int, raw[1].split())
    img = np.frombuffer(raw[3], np.uint8).reshape(h, w)[::-1]
    origin = [float(v) for v in text.split("origin: [")[1].split("]")[0].split(",")[:2]]
    cell = lambda x, y: img[int((y - origin[1]) / 0.05), int((x - origin[0]) / 0.05)]  # noqa: E731
    assert cell(0.0, 0.0) == 254                    # the start pose is free
    assert 5.0 < w * 0.05 < 10.0 and 4.0 < h * 0.05 < 8.0   # ~7 x 6 m flat plus margin
