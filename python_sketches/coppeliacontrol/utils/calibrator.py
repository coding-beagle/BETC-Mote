"""
calibrator.py
=============
Tracks the user's maximum arm-reach over time and derives a scale factor
that maps human reach to the robot's arm length.
"""

from .math_utils import vec_length, vec_sub


class ArmCalibrator:
    DECAY_RATE = 0.002
    MIN_SAMPLES = 30
    MIN_REACH_M = 0.15

    def __init__(self, robot_arm_length: float):
        self.robot_arm_length = robot_arm_length
        self._max_reach = 0.0
        self._samples = 0

    def update(self, human_shoulder, human_elbow, human_wrist, dt: float):
        reach = vec_length(vec_sub(human_wrist, human_shoulder))
        if reach < self.MIN_REACH_M:
            return
        self._samples += 1
        if reach > self._max_reach:
            self._max_reach = reach
        else:
            self._max_reach -= self._max_reach * self.DECAY_RATE * dt

    # ── properties ────────────────────────────────────────────────────────────
    @property
    def calibrated(self) -> bool:
        return self._samples >= self.MIN_SAMPLES and self._max_reach > self.MIN_REACH_M

    @property
    def scale(self) -> float:
        if not self.calibrated:
            return 1.0
        return self.robot_arm_length / self._max_reach

    @property
    def human_max_reach(self) -> float:
        return self._max_reach
