"""Builds the 61D Microduck policy observation. Pure numpy, no ROS.

Mirrors PolicyInference.get_observations() in microduck_rl/scripts/infer_policy.py
(projected-gravity mode, unified 13D command). Field order is part of the
policy contract: never reorder.
"""

import numpy as np

from .constants import CMD_DIM, DEFAULT_POSE, NUM_JOINTS, OBS_DIM

_GRAVITY_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def quat_rotate_inverse(quat_wxyz, vec):
    """Rotate ``vec`` by the inverse of ``quat_wxyz`` (w, x, y, z).

    Same formula as upstream, so float results match bit-for-bit in float32.
    """
    q = np.asarray(quat_wxyz, dtype=np.float32)
    v = np.asarray(vec, dtype=np.float32)
    w = q[0]
    xyz = q[1:4]
    t = np.cross(xyz, v) * 2
    return (v - w * t + np.cross(xyz, t)).astype(np.float32)


def projected_gravity(trunk_quat_wxyz):
    """Unit gravity expressed in the trunk frame."""
    return quat_rotate_inverse(trunk_quat_wxyz, _GRAVITY_WORLD)


def make_command(twist=(0.0, 0.0, 0.0), head=(0.0, 0.0, 0.0, 0.0), body=(0.0,) * 6):
    """13D command block: twist(3) + head_pose deltas(4) + body_pose(6)."""
    cmd = np.zeros(CMD_DIM, dtype=np.float32)
    cmd[0:3] = twist
    cmd[3:7] = head
    cmd[7:13] = body
    return cmd


def build_observation(gyro, trunk_quat_wxyz, joint_pos, joint_vel, last_action, command,
                      default_pose=DEFAULT_POSE):
    """Assemble the policy input.

    Args:
        gyro: trunk angular velocity from the IMU gyro [rad/s], sensor frame.
        trunk_quat_wxyz: trunk orientation in the world, (w, x, y, z).
        joint_pos / joint_vel: 14 values in policy joint order [rad, rad/s].
        last_action: the raw (unscaled) action the policy output last tick.
        command: 13D command block (see make_command).
    """
    obs = np.concatenate(
        [
            np.asarray(gyro, dtype=np.float32).reshape(3),
            projected_gravity(trunk_quat_wxyz),
            np.asarray(joint_pos, dtype=np.float32).reshape(NUM_JOINTS) - default_pose,
            np.asarray(joint_vel, dtype=np.float32).reshape(NUM_JOINTS),
            np.asarray(last_action, dtype=np.float32).reshape(NUM_JOINTS),
            np.asarray(command, dtype=np.float32).reshape(CMD_DIM),
        ]
    ).astype(np.float32)
    assert obs.shape == (OBS_DIM,), obs.shape
    return obs
