"""Microduck policy contract constants.

These values MUST match what the policies were trained with in
pollen-robotics/microduck_rl (see scripts/infer_policy.py and AGENTS.md there).
Ported at microduck_rl commit bae39f5. If upstream changes them, update here
and re-run the parity tests.
"""

import numpy as np

# 14 actuated joints, in policy order (= ctrl order on walk/groundcontact models):
# 0-4 left leg, 5-8 neck/head, 9-13 right leg.
JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
NUM_JOINTS = len(JOINT_NAMES)

# STAND2 / HOME_FRAME pose. Actions are offsets from it, joint-position
# observations are relative to it.
DEFAULT_POSE = np.array(
    [
        0.0, -0.0873, -0.4579, -0.0049, 0.4530,   # left leg
        0.3491, 0.3491, 0.0, 0.0,                 # neck / head
        0.0, 0.0873, 0.4579, 0.0049, -0.4530,     # right leg
    ],
    dtype=np.float32,
)

# Observation layout (61D, shared across the whole policy family):
#   base_ang_vel(3) projected_gravity(3) joint_pos-default(14) joint_vel(14)
#   last_action(14) command(13)
# command = twist[vx, vy, wz](3) head_pose(4) body_pose[x, y, z, roll, pitch, yaw](6)
CMD_DIM = 13
OBS_DIM = 3 + 3 + 3 * NUM_JOINTS + CMD_DIM  # 61
ACTION_DIM = NUM_JOINTS

CONTROL_HZ = 50.0

# Velocity command limits used by Pollen's CPU rehearsal for the walking model.
VEL_LIMITS = {
    "vx": (-0.3, 0.3),
    "vy": (-0.2, 0.2),
    "wz": (-1.5, 1.5),
}

# Body pose command limits (training constants).
BODY_CMD_MAX_XY = 0.02   # m
BODY_CMD_MAX_Z = 0.03    # m
BODY_CMD_MAX_ANGLE = float(np.radians(30.0))
