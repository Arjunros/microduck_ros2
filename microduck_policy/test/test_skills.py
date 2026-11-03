"""Skill controller: manifest parsing and the robotd control cascade."""

import json

import numpy as np
import pytest

from microduck_policy.constants import DEFAULT_POSE
from microduck_policy.observation import make_command
from microduck_policy.policy import load_policy_set

from test_policy import _make_onnx

UP = (1.0, 0.0, 0.0, 0.0)


def _tick(ctrl, n=1, dt=0.02):
    out = None
    for _ in range(n):
        out = ctrl.step(np.zeros(3), UP, DEFAULT_POSE, np.zeros(14), dt)
    return out


@pytest.fixture
def policy_set(tmp_path):
    """A set shaped like pollen-robotics/microduck-policies; each net has a distinct bias."""
    files = {"velstand.onnx": 0.01, "alpha_walking.onnx": 0.02, "alpha_stand.onnx": 0.03,
             "alpha_sitstand.onnx": 0.04, "alpha_ground_pick.onnx": 0.05, "roulade.onnx": 0.06,
             "ball_kick_left.onnx": 0.07, "ball_kick_right.onnx": 0.08, "roller.onnx": 0.09,
             "roller_crouch.onnx": 0.10}
    for f, b in files.items():
        _make_onnx(str(tmp_path / f), bias=b)
    manifest = {"schema_version": 2, "policies": [
        {"file": "alpha_walking.onnx", "kind": "perpetual"},
        {"file": "velstand.onnx", "kind": "perpetual", "slot": "walk"},
        {"file": "alpha_stand.onnx", "kind": "perpetual"},
        {"file": "roller.onnx", "kind": "perpetual", "mode": "roller"},
        {"file": "alpha_sitstand.onnx", "kind": "scripted", "unwind_s": 1.0, "ramp_s": 2.0,
         "command": {"encoding": "posture_flag", "sit": 1, "stand": 0}},
        {"file": "alpha_ground_pick.onnx", "kind": "episodic", "mode": "walk",
         "command": {"encoding": "phase", "period_s": 4.0, "end_phase": 0.7}},
        {"file": "roller_crouch.onnx", "kind": "episodic", "mode": "roller",
         "command": {"encoding": "phase", "period_s": 5.0, "end_phase": 0.7}, "action_scale": 0.8},
        {"file": "roulade.onnx", "kind": "episodic", "duration_s": 1.0, "chain": True},
        {"file": "ball_kick_left.onnx", "name": "kick_left", "kind": "episodic", "duration_s": 0.5},
        {"file": "ball_kick_right.onnx", "name": "kick_right", "kind": "episodic", "duration_s": 0.5,
         "unwind_s": 0.2, "command": {"idle": [0.0, 0.0, 0.0]}},
    ]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_loads_set_like_robotd(policy_set):
    c = load_policy_set(str(policy_set))
    assert c.walking.path.endswith("velstand.onnx")          # the slot=walk entry
    assert c.standing is None                                 # velstand stands on its own
    assert sorted(c.skills) == ["kick_left", "kick_right", "roulade"]
    assert c.skills["kick_left"].duration_s == 0.5
    assert c.ground_pick_def.policy.path.endswith("alpha_ground_pick.onnx")   # not the roller crouch
    assert (c.ground_pick_def.period_s, c.ground_pick_def.end_phase) == (4.0, 0.7)
    assert (c.sitstand_def.rise_s, c.sitstand_def.ramp_s) == (1.0, 2.0)


def test_fallback_skills_without_manifest(policy_set):
    (policy_set / "manifest.json").unlink()
    c = load_policy_set(str(policy_set))
    assert c.walking.path.endswith("velstand.onnx")
    assert {n: d.duration_s for n, d in c.skills.items()} == {"kick_left": 0.5, "kick_right": 0.5, "roulade": 1.0}
    assert c.ground_pick_def is not None and c.sitstand_def is not None


def test_kick_runs_for_its_duration_with_zero_command(policy_set):
    c = load_policy_set(str(policy_set))
    c.set_command(make_command((0.2, 0.0, 0.3), head=(0.1, 0.1, 0.1, 0.1), body=(0.01,) * 6))
    ok, _ = c.start_skill("kick_left")
    assert ok and c.label == "kick_left"
    targets, obs, action = _tick(c)
    np.testing.assert_allclose(action, 0.07, atol=1e-6)        # the kick net ran
    np.testing.assert_allclose(obs[48:61], 0.0)                 # twist, head and body all zero
    _tick(c, 23)
    assert c.label == "kick_left"                               # 24 ticks = 0.48 s
    _tick(c)
    assert c.label == "walk"                                    # 25 ticks = 0.5 s: hand back
    _, obs, action = _tick(c)
    np.testing.assert_allclose(action, 0.01, atol=1e-6)         # velstand again
    np.testing.assert_allclose(obs[48:51], (0.2, 0.0, 0.3), atol=1e-6)


def test_unwind_phase(policy_set):
    c = load_policy_set(str(policy_set))
    c.start_skill("kick_right")
    _tick(c, 25)
    assert c.label == "kick_right:unwind"
    _tick(c, 10)
    assert c.label == "walk"


def test_one_move_at_a_time(policy_set):
    c = load_policy_set(str(policy_set))
    assert c.start_skill("roulade")[0]
    assert not c.start_skill("kick_left")[0]
    assert not c.start_ground_pick()[0]
    assert not c.start_skill("nope")[0]


def test_ground_pick_phase_encoding_and_end(policy_set):
    c = load_policy_set(str(policy_set))
    assert c.start_ground_pick()[0]
    _, obs, _ = _tick(c)
    np.testing.assert_allclose(obs[48:51], (1.0, 0.0, 0.0), atol=1e-6)   # phase 0
    _tick(c, 49)                                                          # phase 0.25 next
    _, obs, _ = _tick(c)
    np.testing.assert_allclose(obs[48:51], (0.0, 1.0, 0.0), atol=1e-5)
    _tick(c, 89)                                                          # 140 ticks = 2.8 s = 0.7 * 4
    assert c.label == "walk"


def test_sit_settle_then_rise(policy_set):
    c = load_policy_set(str(policy_set))
    c.set_command(make_command((0.0, 0.0, 0.0), head=(0.0, 0.2, 0.0, 0.0)))
    assert c.sit()[0] and c.label == "sit"
    _, obs, action = _tick(c)
    np.testing.assert_allclose(action, 0.04, atol=1e-6)
    np.testing.assert_allclose(obs[48:51], (1.0, 0.0, 0.0))     # posture flag = sit
    np.testing.assert_allclose(obs[51:55], (0.0, 0.2, 0.0, 0.0), atol=1e-6)  # head stays live
    assert not c.start_skill("kick_left")[0]                    # no moves from the seat
    ok, msg = c.stand()                                         # asked before the 2 s settle
    assert ok and "settles" in msg and c.label == "sit"
    _tick(c, 99)                                                # 100 ticks = 2.0 s
    assert c.label == "rise"
    _, obs, _ = _tick(c)
    np.testing.assert_allclose(obs[48:51], 0.0)                 # flag 0 = stand
    _tick(c, 49)                                                # 1.0 s rise
    assert c.label == "walk"


def test_reset_clears_everything(policy_set):
    c = load_policy_set(str(policy_set))
    c.sit()
    _tick(c, 3)
    c.reset()
    assert c.label == "walk" and np.all(c.last_action == 0)


def test_explicit_paths_without_set(policy_set):
    c = load_policy_set("", walking=str(policy_set / "alpha_walking.onnx"),
                        standing=str(policy_set / "alpha_stand.onnx"))
    assert c.skills == {} and c.ground_pick_def is None and c.sitstand_def is None
    c.set_command(make_command((0.0, 0.0, 0.0)))
    assert c.label == "stand"
