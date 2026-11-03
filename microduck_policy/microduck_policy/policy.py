"""ONNX policy loading and the 50 Hz control step. Pure python, no ROS."""

import json
import os

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


class MicroduckController:
    """Walking (+ optional standing) policy with Pollen's switching rule.

    Upstream switches to the standing policy when |twist| <= switch_threshold
    and back to walking above it; last_action carries across the switch.
    """

    def __init__(self, walking, standing=None, switch_threshold=0.05, default_pose=DEFAULT_POSE):
        if walking is None and standing is None:
            raise PolicyLoadError("need at least a walking or a standing policy")
        self.walking = walking
        self.standing = standing
        self.switch_threshold = float(switch_threshold)
        self.default_pose = np.asarray(default_pose, dtype=np.float32)
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.command = np.zeros(CMD_DIM, dtype=np.float32)
        self.active_name = "walking" if walking is not None else "standing"

    @property
    def active(self):
        return self.walking if self.active_name == "walking" else self.standing

    def set_command(self, command):
        self.command = np.asarray(command, dtype=np.float32).reshape(CMD_DIM).copy()
        if self.walking is not None and self.standing is not None:
            mag = float(np.linalg.norm(self.command[0:3]))
            self.active_name = "standing" if mag <= self.switch_threshold else "walking"
        if self.active_name == "standing":
            # Standing policies see a zero twist (upstream _update_command).
            self.command[0:3] = 0.0

    def reset(self):
        self.last_action[:] = 0.0

    def step(self, gyro, trunk_quat_wxyz, joint_pos, joint_vel):
        """One control tick. Returns (joint_targets[14], obs[61], action[14])."""
        obs = build_observation(gyro, trunk_quat_wxyz, joint_pos, joint_vel,
                                self.last_action, self.command, self.default_pose)
        action = self.active(obs)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("policy produced a non-finite action")
        self.last_action = action.copy()
        targets = self.default_pose + action * self.active.action_scale
        return targets.astype(np.float32), obs, action
