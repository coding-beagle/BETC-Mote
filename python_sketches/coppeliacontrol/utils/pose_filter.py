"""
pose_filter.py
==============
Exponential moving average (EMA) for position and SLERP for quaternion.

    alpha = 1.0  →  no filtering (pass-through)
    alpha = 0.0  →  completely frozen
    Typical range: 0.1 (very smooth, ~6-frame lag) to 0.4 (responsive).
"""

import numpy as np


class PoseFilter:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self._pos = None  # list[3] | None
        self._quat = None  # list[4] xyzw | None

    # ── position ──────────────────────────────────────────────────────────────
    def update_pos(self, new_pos):
        if new_pos is None:
            return self._pos
        if self._pos is None:
            self._pos = list(new_pos)
        else:
            a = self.alpha
            self._pos = [a * n + (1 - a) * p for n, p in zip(new_pos, self._pos)]
        return self._pos

    # ── quaternion ────────────────────────────────────────────────────────────
    def update_quat(self, new_quat):
        """SLERP from current to new_quat by alpha."""
        if new_quat is None:
            return self._quat
        if self._quat is None:
            self._quat = list(new_quat)
            return self._quat

        q0 = np.array(self._quat)  # xyzw
        q1 = np.array(new_quat)

        if np.dot(q0, q1) < 0:  # ensure shortest path
            q1 = -q1

        result = self._slerp(q0, q1, self.alpha)
        self._quat = result.tolist()
        return self._quat

    # ── internal ──────────────────────────────────────────────────────────────
    @staticmethod
    def _slerp(q0, q1, t):
        dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
        if abs(dot) > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_t0 = np.sin(theta_0)
        return (np.sin(theta_0 - theta) / sin_t0) * q0 + (np.sin(theta) / sin_t0) * q1
