"""ONNX policy loading and the 50 Hz control step. Pure python, no ROS."""

import json
import os
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from .constants import ACTION_DIM, CMD_DIM, DEFAULT_POSE, OBS_DIM
from .observation import build_observation


class PolicyLoadError(RuntimeError):
    pass


def _find_manifest(onnx_path):
    """Return the manifest dict sitting next to a policy file, if any.

    Handles both a single-policy repo (fields at top level) and the official
    set (fields under ``policies[]``, matched on ``file``).
    """
    path = os.path.join(os.path.dirname(os.path.abspath(onnx_path)), "manifest.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    entries = manifest.get("policies")
    if isinstance(entries, list):
        base = os.path.basename(onnx_path)
        for entry in entries:
            if entry.get("file") == base:
                merged = {k: v for k, v in manifest.items() if k != "policies"}
                merged.update(entry)
                return merged
        return {k: v for k, v in manifest.items() if k != "policies"}
    return manifest


class OnnxPolicy:
    """One feed-forward Microduck policy: obs[1,61] -> action[1,14]."""

    def __init__(self, onnx_path, action_scale=None):
        if not os.path.isfile(onnx_path):
            raise PolicyLoadError(f"policy file not found: {onnx_path}")
        self.path = onnx_path
        self.manifest = _find_manifest(onnx_path)
        self._check_manifest()

        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise PolicyLoadError(
                f"{onnx_path}: expected 1 input / 1 output, got {len(inputs)} / {len(outputs)}. "
                "Recurrent (LSTM, model_api 2) policies are not supported yet."
            )
        in_shape, out_shape = inputs[0].shape, outputs[0].shape
        if in_shape[-1] != OBS_DIM or out_shape[-1] != ACTION_DIM:
            raise PolicyLoadError(
                f"{onnx_path}: interface {in_shape} -> {out_shape}, expected [1,{OBS_DIM}] -> [1,{ACTION_DIM}]"
            )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        # Precedence: explicit argument > manifest > 1.0 (upstream default).
        if action_scale is None:
            action_scale = float(self.manifest.get("action_scale", 1.0))
        self.action_scale = float(action_scale)

    def _check_manifest(self):
        m = self.manifest
        if not m:
            return
        if m.get("obs_len", OBS_DIM) != OBS_DIM or m.get("action_len", ACTION_DIM) != ACTION_DIM:
            raise PolicyLoadError(f"{self.path}: manifest obs/action length does not match 61/14")
        model = (m.get("robot") or {}).get("model", "microduck")
        if model != "microduck":
            raise PolicyLoadError(f"{self.path}: manifest is for robot '{model}'")
        if int(m.get("model_api", 1)) > 1:
            raise PolicyLoadError(f"{self.path}: model_api {m['model_api']} (recurrent) is not supported yet")

    def __call__(self, obs):
        out = self.session.run([self.output_name], {self.input_name: obs.reshape(1, OBS_DIM)})[0]
        return np.asarray(out, dtype=np.float32).reshape(ACTION_DIM)


@dataclass
class SkillDef:
    """A one-shot: episodic, constant command, fixed duration (manifest schema 2)."""
    name: str
    policy: OnnxPolicy
    duration_s: float
    command: tuple = (0.0, 0.0, 0.0)   # twist fed while it runs (zero for every published skill)
    unwind: tuple = (0.0, 0.0, 0.0)    # manifest command.idle, fed for unwind_s afterwards
    unwind_s: float = 0.0
    action_scale: float = None         # None: the walking scale


@dataclass
class GroundPickDef:
    """Phase-encoded: twist = [cos 2pi phi, sin 2pi phi, 0], phi 0 -> end_phase over period_s."""
    policy: OnnxPolicy
    period_s: float = 4.0
    end_phase: float = 0.7
    action_scale: float = 1.0


@dataclass
class SitStandDef:
    """Posture flag in the twist vx slot: 1 = sit, 0 = stand."""
    policy: OnnxPolicy
    rise_s: float = 1.0   # seconds on the sitstand net after flag -> 0, then the gait
    ramp_s: float = 2.0   # seconds the seat takes to settle after flag -> 1


class MicroduckController:
    """Gait + skills, following Pollen's robotd control cascade:

        skill > ground pick > sit / rise > stand-by-magnitude > walk

    One move at a time; no move from the seat (stand up first). Skills and the
    ground pick see zero head/body commands; sit and rise keep them live.
    """

    def __init__(self, walking, standing=None, switch_threshold=0.05, default_pose=DEFAULT_POSE,
                 skills=None, ground_pick=None, sitstand=None):
        if walking is None and standing is None:
            raise PolicyLoadError("need at least a walking or a standing policy")
        self.walking = walking
        self.standing = standing
        self.switch_threshold = float(switch_threshold)
        self.default_pose = np.asarray(default_pose, dtype=np.float32)
        self.skills = dict(skills or {})
        self.ground_pick_def = ground_pick
        self.sitstand_def = sitstand
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.user_command = np.zeros(CMD_DIM, dtype=np.float32)
        self.command = np.zeros(CMD_DIM, dtype=np.float32)   # what the active net was fed last
        self.label = "walk" if walking is not None else "stand"
        # Mode state
        self._skill = None        # [SkillDef, phase("hold"|"unwind"), remaining_s]
        self._pick_phase = None   # float or None
        self._sit = "up"          # "up" | "sitting" | "rising"
        self._sit_settle = 0.0
        self._stand_pending = False
        self._rise_left = 0.0
        self._select()

    # ----------------------------------------------------------- commands
    @property
    def active_name(self):
        """Backwards-compatible name of the base gait in use ("walking"/"standing")."""
        return "standing" if self.label == "stand" else "walking"

    def set_command(self, command):
        self.user_command = np.asarray(command, dtype=np.float32).reshape(CMD_DIM).copy()
        self._select()

    def busy(self):
        return self._skill is not None or self._pick_phase is not None

    def _move_blocked(self):
        if self.busy():
            return "another move is running"
        if self._sit != "up":
            return "sitting: stand up first"
        return None

    def start_skill(self, name):
        if name not in self.skills:
            return False, f"unknown skill '{name}' (have: {', '.join(sorted(self.skills)) or 'none'})"
        why = self._move_blocked()
        if why:
            return False, why
        d = self.skills[name]
        self._skill = [d, "hold", d.duration_s]
        self._select()
        return True, f"{name} for {d.duration_s:.2f} s"

    def start_ground_pick(self):
        if self.ground_pick_def is None:
            return False, "no ground-pick policy loaded"
        why = self._move_blocked()
        if why:
            return False, why
        self._pick_phase = 0.0
        self._select()
        return True, f"ground pick, {self.ground_pick_def.period_s * self.ground_pick_def.end_phase:.1f} s"

    def sit(self):
        if self.sitstand_def is None:
            return False, "no sit/stand policy loaded"
        if self.busy():
            return False, "another move is running"
        if self._sit == "sitting":
            return True, "already sitting"
        self._sit = "sitting"
        self._sit_settle = self.sitstand_def.ramp_s
        self._stand_pending = False
        self._select()
        return True, "sitting"

    def stand(self):
        if self._sit == "up":
            return True, "already standing"
        if self._sit == "rising":
            return True, "already rising"
        if self._sit_settle > 0.0:
            self._stand_pending = True
            return True, f"standing once the seat settles ({self._sit_settle:.1f} s)"
        self._begin_rise()
        return True, "standing up"

    def _begin_rise(self):
        self._sit = "rising"
        self._rise_left = self.sitstand_def.rise_s
        self._stand_pending = False
        self._select()

    # -------------------------------------------------------------- tick
    def reset(self):
        self.last_action[:] = 0.0
        self._skill = None
        self._pick_phase = None
        self._sit = "up"
        self._sit_settle = 0.0
        self._stand_pending = False
        self._select()

    def _select(self):
        """Pick the net, the command it sees, the action scale and the label."""
        cmd = self.user_command.copy()
        if self._skill is not None:
            d, phase, _ = self._skill
            cmd[:] = 0.0
            cmd[0:3] = d.command if phase == "hold" else d.unwind
            scale = d.action_scale if d.action_scale is not None else self._gait_scale()
            net, label = d.policy, d.name if phase == "hold" else d.name + ":unwind"
        elif self._pick_phase is not None:
            ang = 2 * np.pi * self._pick_phase
            cmd[:] = 0.0
            cmd[0:3] = (np.cos(ang), np.sin(ang), 0.0)
            gp = self.ground_pick_def
            net, scale, label = gp.policy, gp.action_scale, "ground_pick"
        elif self._sit == "sitting":
            cmd[0:3] = (1.0, 0.0, 0.0)
            net, scale, label = self.sitstand_def.policy, 1.0, "sit"
        elif self._sit == "rising":
            cmd[0:3] = 0.0
            net, scale, label = self.sitstand_def.policy, 1.0, "rise"
        else:
            standing = self.standing is not None and (
                self.walking is None or float(np.linalg.norm(cmd[0:3])) <= self.switch_threshold)
            if standing:
                cmd[0:3] = 0.0   # standing policies see a zero twist (upstream _update_command)
                net, label = self.standing, "stand"
            else:
                net, label = self.walking, "walk"
            scale = net.action_scale
        self.command, self._net, self._scale, self.label = cmd, net, float(scale), label

    def _gait_scale(self):
        return (self.walking or self.standing).action_scale

    @property
    def active(self):
        return self._net

    def _advance(self, dt):
        if self._skill is not None:
            self._skill[2] -= dt
            if self._skill[2] <= 1e-9:
                d, phase, _ = self._skill
                if phase == "hold" and d.unwind_s > 0.0:
                    self._skill = [d, "unwind", d.unwind_s]
                else:
                    self._skill = None
        elif self._pick_phase is not None:
            self._pick_phase += dt / self.ground_pick_def.period_s
            if self._pick_phase >= self.ground_pick_def.end_phase - 1e-9:
                self._pick_phase = None
        elif self._sit == "sitting" and self._sit_settle > 0.0:
            self._sit_settle = max(0.0, self._sit_settle - dt)
            if self._sit_settle == 0.0 and self._stand_pending:
                self._begin_rise()
                return
        elif self._sit == "rising":
            self._rise_left -= dt
            if self._rise_left <= 1e-9:
                self._sit = "up"
        self._select()

    def step(self, gyro, trunk_quat_wxyz, joint_pos, joint_vel, dt=0.02):
        """One control tick. Returns (joint_targets[14], obs[61], action[14]).

        Timers advance AFTER the tick, so a skill's first tick runs at phase 0.
        """
        obs = build_observation(gyro, trunk_quat_wxyz, joint_pos, joint_vel,
                                self.last_action, self.command, self.default_pose)
        action = self._net(obs)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("policy produced a non-finite action")
        self.last_action = action.copy()
        targets = self.default_pose + action * self._scale
        self._advance(dt)
        return targets.astype(np.float32), obs, action


# Skills every robot has had since the prototype, for a set without a manifest
# (robotd-params fallback_skills).
_FALLBACK_SKILLS = (("kick_left", "ball_kick_left.onnx", 0.5),
                    ("kick_right", "ball_kick_right.onnx", 0.5),
                    ("roulade", "roulade.onnx", 1.0))
_DAEMON_OWNED = ("ground_pick", "sit_toggle")


def _read_set_manifest(policy_dir):
    path = os.path.join(policy_dir, "manifest.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _skill_name(entry):
    return entry.get("name") or os.path.splitext(entry["file"])[0]


def _resolve(policy_dir, name):
    if not name:
        return None
    path = os.path.expanduser(name)
    if not os.path.isabs(path) and policy_dir:
        path = os.path.join(policy_dir, path)
    return path


def load_policy_set(policy_dir, walking=None, standing=None, switch_threshold=0.05,
                    action_scale=None, log=print):
    """Build a controller from a policy set directory (e.g. pollen-robotics/microduck-policies).

    walking / standing: file names in the set, or absolute paths. Unset walking
    means the set's `slot: walk` entry, else velstand.onnx, else alpha_walking.onnx.
    Skills, ground pick and sit/stand come from the manifest, the way robotd reads it.
    """
    policy_dir = os.path.expanduser(policy_dir) if policy_dir else ""
    manifest = _read_set_manifest(policy_dir) if policy_dir else {}
    entries = [e for e in manifest.get("policies", []) if isinstance(e, dict) and e.get("file")]
    exists = lambda f: bool(policy_dir) and os.path.isfile(os.path.join(policy_dir, f))  # noqa: E731

    if not walking and policy_dir:
        walk_slot = [e["file"] for e in entries if e.get("slot") == "walk"]
        for cand in walk_slot + ["velstand.onnx", "alpha_walking.onnx"]:
            if exists(cand):
                walking = cand
                break
    walk_pol = OnnxPolicy(_resolve(policy_dir, walking), action_scale) if walking else None
    stand_pol = OnnxPolicy(_resolve(policy_dir, standing), action_scale) if standing else None

    skills, ground_pick, sitstand = {}, None, None
    if policy_dir:
        def roller(e):
            return e.get("mode") == "roller" or "roller" in e["file"]

        for e in entries:
            cmd = e.get("command") or {}
            enc = cmd.get("encoding", "constant")
            if not exists(e["file"]) or roller(e):
                continue
            try:
                if enc == "phase" and ground_pick is None:
                    ground_pick = GroundPickDef(
                        OnnxPolicy(os.path.join(policy_dir, e["file"]), e.get("action_scale", 1.0)),
                        float(cmd.get("period_s", 4.0)), float(cmd.get("end_phase", 0.7)))
                elif enc == "posture_flag" and sitstand is None:
                    sitstand = SitStandDef(OnnxPolicy(os.path.join(policy_dir, e["file"]), 1.0),
                                           float(e.get("unwind_s", 1.0)), float(e.get("ramp_s", 2.0)))
                elif (e.get("kind") == "episodic" and enc == "constant" and e.get("duration_s")
                      and _skill_name(e) not in _DAEMON_OWNED):
                    name = _skill_name(e)
                    skills[name] = SkillDef(
                        name, OnnxPolicy(os.path.join(policy_dir, e["file"])), float(e["duration_s"]),
                        unwind=tuple(cmd.get("idle") or (0.0, 0.0, 0.0)),
                        unwind_s=float(e.get("unwind_s", 0.0)), action_scale=e.get("action_scale"))
            except PolicyLoadError as err:
                log(f"skipping {e['file']}: {err}")
        if not skills:
            for name, f, dur in _FALLBACK_SKILLS:
                if exists(f):
                    skills[name] = SkillDef(name, OnnxPolicy(os.path.join(policy_dir, f)), dur)
        if ground_pick is None and exists("alpha_ground_pick.onnx"):
            ground_pick = GroundPickDef(OnnxPolicy(os.path.join(policy_dir, "alpha_ground_pick.onnx"), 1.0))
        if sitstand is None and exists("alpha_sitstand.onnx"):
            sitstand = SitStandDef(OnnxPolicy(os.path.join(policy_dir, "alpha_sitstand.onnx"), 1.0))

    return MicroduckController(walk_pol, stand_pol, switch_threshold,
                               skills=skills, ground_pick=ground_pick, sitstand=sitstand)
