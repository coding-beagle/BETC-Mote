"""
pose_utils.py
=============
Higher-level pose helpers that build on math_utils:
  - retarget()               maps human wrist position to robot workspace
  - compute_wrist_quaternion()  derives wrist orientation from arm landmarks
"""

from math import sqrt

import numpy as np

from math_utils import (
    vec_add,
    vec_length,
    vec_normalize,
    vec_scale,
    vec_sub,
    remap_axes,
)


def retarget(
    human_shoulder,
    human_wrist,
    robot_shoulder,
    robot_arm_length,
    human_scale: float = 1.0,
):
    """
    Map the human wrist vector (relative to shoulder) into the robot workspace.

    The output position is clamped to robot_arm_length so the IK target never
    leaves the reachable sphere.
    """
    human_vec = vec_sub(human_wrist, human_shoulder)
    human_length = vec_length(human_vec)
    if human_length < 1e-6:
        return None
    scaled_length = min(human_length * human_scale, robot_arm_length)
    direction = vec_scale(human_vec, 1.0 / human_length)
    remapped = remap_axes(direction)
    return vec_add(robot_shoulder, vec_scale(remapped, scaled_length))


def compute_wrist_quaternion(human_shoulder, human_elbow, human_wrist):
    """
    Build a quaternion (xyzw) representing the wrist orientation derived from
    the forearm and upper-arm vectors.

    Returns None if the geometry is degenerate.
    """
    forearm = vec_sub(human_wrist, human_elbow)
    upper = vec_sub(human_elbow, human_shoulder)

    fwd = vec_normalize(remap_axes(forearm))
    up_raw = vec_normalize(remap_axes(upper))
    if fwd is None or up_raw is None:
        return None

    fwd_np = np.array(fwd)
    up_np = np.array(up_raw)
    up_np = up_np - np.dot(up_np, fwd_np) * fwd_np  # Gram-Schmidt
    up_len = np.linalg.norm(up_np)
    if up_len < 1e-6:
        return None
    up_np /= up_len

    right_np = np.cross(fwd_np, up_np)
    R = np.column_stack([right_np, up_np, fwd_np])

    # Rotation matrix → quaternion
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return [x, y, z, w]
