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
    sim.reset()
    assert sim.time == 0.0
    np.testing.assert_allclose(sim.targets, sim.default_pose)


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
