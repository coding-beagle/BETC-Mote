"""
YuMi Pose Control — Multi-Camera Threaded Version
===================================================
Each camera runs MediaPipe pose detection in its own thread.
The main loop picks the best available pose (primary camera first,
falling back to secondary cameras if tracking is lost) and feeds it
to CoppeliaSim IK.

Experiment Modes
----------------
Press  R  to start / restart the Reach experiment
Press  T  to start / restart the Transport experiment
Press  Q  to quit

Configuration
-------------
CAMERA_INDICES  : list of OpenCV camera indices to open
PRIMARY_CAMERA  : index within CAMERA_INDICES to use as primary source
TILE_WIDTH      : width (px) of each camera tile in the display window
"""

import csv
import datetime
import threading
from math import pi, sqrt

import cv2
import mediapipe as mp
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from reach_experiment import Experiment, TransportExperiment

# ── camera configuration ──────────────────────────────────────────────────────
CAMERA_INDICES = [0, 1]  # OpenCV device indices; edit to match your setup
PRIMARY_CAMERA = 0  # which index in CAMERA_INDICES is the preferred source
TILE_WIDTH = 640  # display width of the primary (HUD) camera tile
SECONDARY_TILE_WIDTH = 240
HUD_CAMERA = 0  # index within CAMERA_INDICES that gets the experiment HUD

# ── constants ─────────────────────────────────────────────────────────────────
DEG_TO_RAD = pi / 180

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16

ROBOT_ARM_LENGTH = 0.21492 + 0.24129

# ── gripper signal ────────────────────────────────────────────────────────────
GRIPPER_SIGNAL = "gripper_close"  # matches joystick controller: 1=close, 0=open

# ── hand gesture configuration ────────────────────────────────────────────────
# Finger curl threshold: fraction of max curl distance at which a finger counts
# as "closed".  Lower = more sensitive (easier to trigger closed).
FINGER_CURL_THRESHOLD = 1.2
# How many of the 4 non-thumb fingers must be curled to call the hand "closed"
FINGER_CLOSED_COUNT = 3
# Hysteresis: once open/closed, require this many consecutive frames of the
# opposite state before flipping (prevents jitter).
GRIPPER_DEBOUNCE_FRAMES = 4

# ── pose smoothing ────────────────────────────────────────────────────────────
# EMA alpha for position and quaternion (0 = frozen, 1 = no filtering).
# Lower values = smoother but laggier.  0.15–0.25 is a good starting range.
POSE_SMOOTH_ALPHA = 0.2

# ── reach experiment configuration ────────────────────────────────────────────
EXP_N_TRIALS = 10
EXP_RADIUS = 0.05
EXP_DWELL_TIME = 0.5
EXP_TIMEOUT = 20.0
EXP_MIN_REACH = 0.7
EXP_MAX_REACH = 0.9
EXP_MIN_ELEVATION = -35.0
EXP_MAX_ELEVATION = 50.0
EXP_AZ_MIN = 20.0
EXP_AZ_MAX = 110.0
EXP_SEED = None

# ── transport experiment configuration ────────────────────────────────────────
TRANSPORT_N_TRIALS = 10
TRANSPORT_PICK_RADIUS = 0.06
TRANSPORT_DROP_RADIUS = 0.06
TRANSPORT_TIMEOUT = 30.0
TRANSPORT_MIN_REACH = 0.85
TRANSPORT_MAX_REACH = 0.9
TRANSPORT_MIN_ELEV = -35.0
TRANSPORT_MAX_ELEV = 50.0
TRANSPORT_AZ_MIN = 50.0
TRANSPORT_AZ_MAX = 90.0
TRANSPORT_SEED = None

# ── experiment mode ────────────────────────────────────────────────────────────
MODE_REACH = "reach"
MODE_TRANSPORT = "transport"
MODE_SELECT = "select"  # lobby / menu screen


# ── vector helpers ────────────────────────────────────────────────────────────
def vec3(lm):
    return [lm.x, lm.y, lm.z]


def vec_sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def vec_add(a, b):
    return [a[i] + b[i] for i in range(3)]


def vec_scale(v, s):
    return [v[i] * s for i in range(3)]


def vec_length(v):
    return sum(x**2 for x in v) ** 0.5


def vec_normalize(v):
    l = vec_length(v)
    return vec_scale(v, 1.0 / l) if l > 1e-6 else None


def remap_axes(v):
    return [-v[2], v[0], -v[1]]


# ── pose low-pass filter ──────────────────────────────────────────────────────
class PoseFilter:
    """
    Exponential moving average (EMA) for position + SLERP for quaternion.

    Both use the same alpha:
        smoothed = alpha * new + (1 - alpha) * previous

    For quaternions this is done via SLERP so the interpolation always travels
    the short way around the 4D sphere and never flips sign mid-motion.

    alpha = 1.0  →  no filtering (pass-through)
    alpha = 0.0  →  completely frozen
    Typical range: 0.1 (very smooth, ~6-frame lag) to 0.4 (responsive).
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self._pos = None  # list[3] | None
        self._quat = None  # list[4] xyzw | None

    def update_pos(self, new_pos):
        if new_pos is None:
            return self._pos
        if self._pos is None:
            self._pos = list(new_pos)
        else:
            a = self.alpha
            self._pos = [a * n + (1 - a) * p for n, p in zip(new_pos, self._pos)]
        return self._pos

    def update_quat(self, new_quat):
        """SLERP from current to new_quat by alpha."""
        if new_quat is None:
            return self._quat
        if self._quat is None:
            self._quat = list(new_quat)
            return self._quat

        q0 = np.array(self._quat)  # xyzw
        q1 = np.array(new_quat)

        # Ensure shortest path
        if np.dot(q0, q1) < 0:
            q1 = -q1

        result = self._slerp(q0, q1, self.alpha)
        self._quat = result.tolist()
        return self._quat

    @staticmethod
    def _slerp(q0, q1, t):
        dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
        if abs(dot) > 0.9995:
            # Quaternions are nearly identical — lerp + normalise
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_t0 = np.sin(theta_0)
        return (np.sin(theta_0 - theta) / sin_t0) * q0 + (np.sin(theta) / sin_t0) * q1


# ── arm-length calibration ────────────────────────────────────────────────────
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


# ── pose helpers ──────────────────────────────────────────────────────────────
def retarget(
    human_shoulder,
    human_wrist,
    robot_shoulder,
    robot_arm_length,
    human_scale: float = 1.0,
):
    human_vec = vec_sub(human_wrist, human_shoulder)
    human_length = vec_length(human_vec)
    if human_length < 1e-6:
        return None
    scaled_length = min(human_length * human_scale, robot_arm_length)
    direction = vec_scale(human_vec, 1.0 / human_length)
    remapped = remap_axes(direction)
    return vec_add(robot_shoulder, vec_scale(remapped, scaled_length))


def compute_wrist_quaternion(human_shoulder, human_elbow, human_wrist):
    forearm = vec_sub(human_wrist, human_elbow)
    upper = vec_sub(human_elbow, human_shoulder)

    fwd = vec_normalize(remap_axes(forearm))
    if fwd is None:
        return None
    up_raw = vec_normalize(remap_axes(upper))
    if up_raw is None:
        return None

    fwd_np = np.array(fwd)
    up_np = np.array(up_raw)
    up_np = up_np - np.dot(up_np, fwd_np) * fwd_np
    up_len = np.linalg.norm(up_np)
    if up_len < 1e-6:
        return None
    up_np /= up_len

    right_np = np.cross(fwd_np, up_np)
    R = np.column_stack([right_np, up_np, fwd_np])

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


# ── OpenCV HUD helpers ────────────────────────────────────────────────────────
def _cv_col(r, g, b):
    return (b, g, r)


def draw_mode_select_hud(frame):
    """Overlay a mode-selection menu on the HUD camera frame."""
    H, W = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (10, 10, 18), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        ("YuMi Pose Control", 0.9, (210, 210, 210), 2),
        ("Select experiment mode:", 0.65, (160, 160, 160), 1),
        ("", 0.0, (0, 0, 0), 0),
        ("[R]  Reach Experiment", 0.7, _cv_col(100, 220, 130), 2),
        ("[T]  Transport Experiment", 0.7, _cv_col(100, 180, 255), 2),
        ("", 0.0, (0, 0, 0), 0),
        ("Transport: open hand = gripper open", 0.45, (140, 140, 140), 1),
        ("          closed fist = gripper closed", 0.45, (140, 140, 140), 1),
        ("", 0.0, (0, 0, 0), 0),
        ("[Q]  Quit", 0.5, (130, 130, 130), 1),
    ]
    y = H // 2 - 120
    for text, scale, col, thick in lines:
        if not text:
            y += 14
            continue
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(
            frame,
            text,
            (W // 2 - tw // 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            col,
            thick,
        )
        y += th + 16


def draw_experiment_hud(frame, experiment, wrist_pos, dt, mode):
    """Unified HUD for both Reach and Transport experiments."""
    H, W = frame.shape[:2]

    # ── mode badge (top-left corner) ──────────────────────────────────────────
    if mode == MODE_REACH:
        badge_txt = "REACH"
        badge_col = _cv_col(100, 220, 130)
    else:
        badge_txt = "TRANSPORT"
        badge_col = _cv_col(100, 180, 255)

    cv2.rectangle(frame, (0, 0), (130, 28), (22, 22, 28), -1)
    cv2.putText(frame, badge_txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_col, 1)

    # ── switch hint ───────────────────────────────────────────────────────────
    hint = "[R] Reach  [T] Transport  [Q] Quit"
    cv2.putText(
        frame, hint, (W - 330, H - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1
    )

    active = experiment._active

    # ── progress bar (bottom) ─────────────────────────────────────────────────
    total = len(experiment._trial_defs)
    done = len(experiment.results)
    bar_y = H - 12
    cv2.rectangle(frame, (10, bar_y), (W - 10, bar_y + 8), (45, 45, 55), -1)
    if total and done:
        fill_x = 10 + int((W - 20) * done / total)
        cv2.rectangle(
            frame, (10, bar_y), (fill_x, bar_y + 8), _cv_col(100, 220, 130), -1
        )
    prog = f"Trial {min(done+1, total)} / {total}"
    cv2.putText(
        frame, prog, (10, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1
    )

    # ── finished overlay ──────────────────────────────────────────────────────
    if experiment.finished:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (18, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        n_ok = sum(1 for r in experiment.results if r["result"] == "success")
        cv2.putText(
            frame,
            "EXPERIMENT COMPLETE",
            (W // 2 - 160, H // 2 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            _cv_col(100, 220, 130),
            2,
        )
        cv2.putText(
            frame,
            f"{n_ok} / {total}  trials succeeded",
            (W // 2 - 120, H // 2 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (210, 210, 210),
            1,
        )
        cv2.putText(
            frame,
            "Press [R] or [T] to restart",
            (W // 2 - 140, H // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (130, 130, 130),
            1,
        )
        return

    if active is None:
        return

    # ── REACH-specific HUD ───────────────────────────────────────────────────
    if mode == MODE_REACH:
        dist = active.distance_to(wrist_pos)
        ratio = min(1.0, dist / (active.radius * 6))

        def lerp(a, b, t):
            t = max(0.0, min(1.0, t))
            return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

        C_HOT = (100, 220, 130)
        C_WARM = (220, 190, 60)
        C_IDLE = (100, 160, 220)
        hud_rgb = (
            lerp(C_HOT, C_WARM, ratio / 0.4)
            if ratio < 0.4
            else lerp(C_WARM, C_IDLE, (ratio - 0.4) / 0.6)
        )
        hud_col = _cv_col(*hud_rgb)

        if active._flash_t > 0:
            flash_rgb = (
                (100, 220, 130) if active._result == "success" else (220, 80, 80)
            )
            alpha = active._flash_t / 0.6
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), _cv_col(*flash_rgb), -1)
            cv2.addWeighted(overlay, alpha * 0.45, frame, 1 - alpha * 0.45, 0, frame)
            msg = "TARGET REACHED" if active._result == "success" else "TIMED OUT"
            cv2.putText(
                frame,
                msg,
                (W // 2 - 110, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 255),
                2,
            )
            return

        px, py = W - 270, 30
        cv2.rectangle(frame, (px - 8, py - 4), (W - 8, py + 115), (22, 22, 28), -1)
        cv2.rectangle(frame, (px - 8, py - 4), (W - 8, py + 115), hud_col, 1)
        cv2.putText(
            frame,
            active.label,
            (px, py + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
        )
        cv2.putText(
            frame,
            f"dist  {dist*100:.1f} cm",
            (px, py + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            hud_col,
            1,
        )

        bar_x, bar_y2 = px, py + 52
        bar_w = 250
        fill = int(bar_w * (1.0 - ratio))
        cv2.rectangle(
            frame, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + 8), (55, 55, 55), -1
        )
        if fill > 0:
            cv2.rectangle(
                frame, (bar_x, bar_y2), (bar_x + fill, bar_y2 + 8), hud_col, -1
            )

        if active._inside and active.dwell_fraction > 0:
            cx, cy = W - 35, py + 88
            r = 18
            cv2.circle(frame, (cx, cy), r, (55, 55, 55), 2)
            cv2.ellipse(
                frame,
                (cx, cy),
                (r, r),
                -90,
                0,
                int(360 * active.dwell_fraction),
                _cv_col(100, 220, 130),
                2,
            )
            cv2.putText(
                frame,
                f"{int(active.dwell_fraction*100)}%",
                (cx - 14, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                _cv_col(100, 220, 130),
                1,
            )

        if active.timeout > 0:
            if not active._started:
                cv2.putText(
                    frame,
                    "move to start timer",
                    (px, py + 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    _cv_col(160, 140, 60),
                    1,
                )
            else:
                tr = active.time_remaining
                t_col = _cv_col(220, 80, 80) if tr < 3.0 else (130, 130, 130)
                cv2.putText(
                    frame,
                    f"time  {tr:.1f}s",
                    (px, py + 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    t_col,
                    1,
                )

    # ── TRANSPORT-specific HUD ───────────────────────────────────────────────
    else:
        _PHASE_APPROACH = "approach"
        _PHASE_GRIP = "carry"  # matches internal phase strings from reach_experiment
        _PHASE_CARRY = "carry"
        _PHASE_PLACE = "place"

        # Phase label map (mirrors reach_experiment._PHASE_LABELS)
        PHASE_LABELS = {
            "approach": "1. Move wrist to cube  (open hand)",
            "grip": "2. Close fist to grip",
            "carry": "3. Carry cube to drop zone",
            "place": "4. Open hand to release",
            "done": "Done!",
        }
        phase = getattr(active, "_phase", "approach")
        phase_text = PHASE_LABELS.get(phase, phase)

        C_ORANGE = _cv_col(230, 130, 40)
        C_HOT = _cv_col(100, 220, 130)
        phase_col = C_ORANGE if phase in ("approach", "grip") else C_HOT

        if active._flash_t > 0 and active._result:
            flash_rgb = (
                (100, 220, 130) if active._result == "success" else (220, 80, 80)
            )
            alpha = active._flash_t / 0.8
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), _cv_col(*flash_rgb), -1)
            cv2.addWeighted(overlay, alpha * 0.45, frame, 1 - alpha * 0.45, 0, frame)
            msg = "DELIVERED!" if active._result == "success" else "TIMED OUT"
            cv2.putText(
                frame,
                msg,
                (W // 2 - 100, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 255),
                2,
            )
            return

        px, py = W - 290, 30
        panel_h = 140
        cv2.rectangle(frame, (px - 8, py - 4), (W - 8, py + panel_h), (22, 22, 28), -1)
        cv2.rectangle(frame, (px - 8, py - 4), (W - 8, py + panel_h), phase_col, 1)

        cv2.putText(
            frame,
            active.label,
            (px, py + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
        )
        cv2.putText(
            frame,
            phase_text,
            (px, py + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            phase_col,
            1,
        )

        # Distances
        import math

        cur_cube = getattr(active, "_current_cube_pos", active.cube_pos)
        d_cube = math.sqrt(sum((wrist_pos[i] - cur_cube[i]) ** 2 for i in range(3)))
        d_drop = math.sqrt(
            sum((wrist_pos[i] - active.drop_pos[i]) ** 2 for i in range(3))
        )

        if phase in ("approach", "grip"):
            d_txt = f"cube  {d_cube*100:.1f} cm"
            d_col = C_HOT if d_cube <= active.pick_radius else C_ORANGE
        else:
            d_txt = f"drop  {d_drop*100:.1f} cm"
            d_col = C_HOT if d_drop <= active.drop_radius else _cv_col(220, 190, 60)
        cv2.putText(
            frame, d_txt, (px, py + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, d_col, 1
        )

        gripped = getattr(active, "_gripped", False)
        g_txt = (
            "hand: CLOSED  (open to release)"
            if gripped
            else "hand: OPEN  (close fist to grip)"
        )
        g_col = C_HOT if gripped else _cv_col(100, 160, 220)
        cv2.putText(
            frame, g_txt, (px, py + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, g_col, 1
        )

        # Phase dots
        phases = ["approach", "grip", "carry", "place"]
        phase_idx = phases.index(phase) if phase in phases else 0
        dot_y = py + 108
        for i, ph in enumerate(phases):
            dot_x = px + i * 62 + 15
            done_ph = phase_idx > i
            active_ph = phase_idx == i
            col = (
                _cv_col(100, 220, 130)
                if done_ph
                else (phase_col if active_ph else (55, 55, 55))
            )
            cv2.circle(frame, (dot_x, dot_y), 7, col, -1)
            cv2.putText(
                frame,
                str(i + 1),
                (dot_x - 4, dot_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (20, 20, 20) if (done_ph or active_ph) else (100, 100, 100),
                1,
            )
            if i < len(phases) - 1:
                line_col = _cv_col(100, 220, 130) if done_ph else (55, 55, 55)
                cv2.line(frame, (dot_x + 7, dot_y), (dot_x + 55, dot_y), line_col, 2)

        tr = active.time_remaining
        t_col = _cv_col(220, 80, 80) if tr < 5.0 else (130, 130, 130)
        cv2.putText(
            frame,
            f"time  {tr:.1f}s",
            (px, py + 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            t_col,
            1,
        )


# ── Hand gesture classifier ───────────────────────────────────────────────────
_FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky"]
_MCP_IDS = [5, 9, 13, 17]
_TIP_IDS = [8, 12, 16, 20]
_WRIST_ID = 0


def compute_finger_curls(hand_landmarks):
    """
    Return a list of 4 curl ratios (tip_dist / mcp_dist) for Index→Pinky.

    ratio < FINGER_CURL_THRESHOLD  →  finger is curled / closed
    ratio >= FINGER_CURL_THRESHOLD →  finger is extended / open

    Values are clamped to [0, 2] for display purposes.
    Returns None if landmarks are invalid.
    """
    lm = hand_landmarks.landmark

    def dist3(a, b):
        return (
            (lm[a].x - lm[b].x) ** 2
            + (lm[a].y - lm[b].y) ** 2
            + (lm[a].z - lm[b].z) ** 2
        ) ** 0.5

    ratios = []
    for mcp, tip in zip(_MCP_IDS, _TIP_IDS):
        d_mcp = dist3(_WRIST_ID, mcp)
        if d_mcp < 1e-6:
            ratios.append(1.0)  # neutral if degenerate
        else:
            ratios.append(min(2.0, dist3(_WRIST_ID, tip) / d_mcp))
    return ratios


def classify_hand_open(hand_landmarks) -> bool:
    """True = open hand, False = closed fist."""
    ratios = compute_finger_curls(hand_landmarks)
    curled = sum(1 for r in ratios if r < FINGER_CURL_THRESHOLD)
    return curled < FINGER_CLOSED_COUNT


def draw_curl_meter(frame, curl_ratios, origin_xy, label_prefix=""):
    """
    Draw a compact per-finger curl meter at origin_xy (top-left of panel).

    Each finger gets a horizontal bar:
      - Full bar width = ratio of 2.0 (fully extended beyond MCP)
      - Threshold line marks FINGER_CURL_THRESHOLD
      - Bar colour: green (extended/open) → red (curled/closed)
      - A small label shows the finger name and raw ratio

    origin_xy : (x, y) top-left corner of the panel
    """
    if curl_ratios is None:
        return

    ox, oy = origin_xy
    BAR_W = 110  # max bar width in pixels
    BAR_H = 9
    ROW_STEP = 17
    MAX_RATIO = 2.0  # ratio that fills the full bar

    # Panel background
    panel_h = ROW_STEP * 4 + 10
    cv2.rectangle(
        frame, (ox - 4, oy - 4), (ox + BAR_W + 74, oy + panel_h), (22, 22, 28), -1
    )
    cv2.rectangle(
        frame, (ox - 4, oy - 4), (ox + BAR_W + 74, oy + panel_h), (55, 55, 65), 1
    )

    if label_prefix:
        cv2.putText(
            frame,
            label_prefix,
            (ox, oy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (110, 110, 110),
            1,
        )

    thresh_x = ox + int(BAR_W * FINGER_CURL_THRESHOLD / MAX_RATIO)

    for i, (name, ratio) in enumerate(zip(_FINGER_NAMES, curl_ratios)):
        by = oy + i * ROW_STEP

        # Background track
        cv2.rectangle(frame, (ox, by), (ox + BAR_W, by + BAR_H), (45, 45, 45), -1)

        # Filled portion — colour shifts green→red as ratio drops below threshold
        fill_w = max(1, int(BAR_W * min(ratio, MAX_RATIO) / MAX_RATIO))
        t = max(
            0.0,
            min(
                1.0,
                1.0
                - (ratio - FINGER_CURL_THRESHOLD)
                / max(1e-6, 1.0 - FINGER_CURL_THRESHOLD),
            ),
        )
        r_ch = int(40 + t * (220 - 40))
        g_ch = int(200 - t * (200 - 60))
        bar_col = (40, g_ch, r_ch)  # BGR
        cv2.rectangle(frame, (ox, by), (ox + fill_w, by + BAR_H), bar_col, -1)

        # Threshold line
        cv2.line(
            frame, (thresh_x, by - 1), (thresh_x, by + BAR_H + 1), (200, 200, 200), 1
        )

        # Label + value
        is_curled = ratio < FINGER_CURL_THRESHOLD
        lbl_col = (60, 60, 220) if is_curled else (60, 200, 100)  # BGR
        cv2.putText(
            frame,
            f"{name[0]}  {ratio:.2f}{'  curl' if is_curled else ''}",
            (ox + BAR_W + 4, by + BAR_H - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            lbl_col,
            1,
        )


# ── Camera thread ─────────────────────────────────────────────────────────────
class CameraThread(threading.Thread):
    """
    Captures frames and runs both MediaPipe Pose and MediaPipe Hands in a
    background thread.

    Public attributes (protected by self.lock):
        frame           : latest BGR frame (or None)
        world_landmarks : latest pose_world_landmarks (or None)
        pose_landmarks  : latest pose_landmarks for drawing (or None)
        tracking        : True when a pose is detected
        hand_landmarks  : list of detected hand landmark sets (may be empty)
        hand_open       : bool — True = open hand detected, False = closed fist
                          (None if no hand visible)
    """

    def __init__(self, cam_index: int, cam_id: int):
        super().__init__(daemon=True, name=f"CamThread-{cam_id}")
        self.cam_index = cam_index
        self.cam_id = cam_id
        self.lock = threading.Lock()
        self.frame = None
        self.world_landmarks = None
        self.pose_landmarks = None
        self.tracking = False
        self.hand_landmarks = []  # list of hand landmark objects
        self.hand_open = None  # None = no hand; True = open; False = closed
        self.hand_curl_ratios = None  # list of 4 floats (Index→Pinky) or None
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        pose = mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,  # fastest model
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        cap = cv2.VideoCapture(self.cam_index + cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(
                f"[CamThread-{self.cam_id}] WARNING: could not open camera {self.cam_index}"
            )
            return
        print(f"[CamThread-{self.cam_id}] Camera {self.cam_index} opened.")

        # Debounce state (per thread)
        _pending_open = None
        _pending_frames = 0

        try:
            while not self._stop_event.is_set():
                ret, bgr = cap.read()
                if not ret:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb)
                hand_results = hands.process(rgb)

                # ── hand state with debounce ──────────────────────────────────
                raw_hand_lms = []
                raw_open = None
                raw_curls = None
                if hand_results.multi_hand_landmarks:
                    raw_hand_lms = hand_results.multi_hand_landmarks
                    raw_curls = compute_finger_curls(raw_hand_lms[0])
                    raw_open = (
                        sum(1 for r in raw_curls if r < FINGER_CURL_THRESHOLD)
                        < FINGER_CLOSED_COUNT
                    )

                # Debounce: only flip state after GRIPPER_DEBOUNCE_FRAMES
                # consecutive frames of the new state
                if raw_open is None:
                    _pending_open = None
                    _pending_frames = 0
                    debounced_open = None
                else:
                    if raw_open == _pending_open:
                        _pending_frames += 1
                    else:
                        _pending_open = raw_open
                        _pending_frames = 1
                    debounced_open = (
                        raw_open
                        if _pending_frames >= GRIPPER_DEBOUNCE_FRAMES
                        else None  # still warming up — keep previous value
                    )

                with self.lock:
                    self.frame = bgr
                    self.world_landmarks = pose_results.pose_world_landmarks
                    self.pose_landmarks = pose_results.pose_landmarks
                    self.tracking = pose_results.pose_world_landmarks is not None
                    self.hand_landmarks = raw_hand_lms
                    self.hand_curl_ratios = (
                        raw_curls  # always latest, no debounce needed
                    )
                    # Only update hand_open once debounce threshold is met
                    if debounced_open is not None:
                        self.hand_open = debounced_open
                    elif raw_open is None:
                        self.hand_open = None  # hand lost entirely
        finally:
            cap.release()
            pose.close()
            hands.close()
            print(f"[CamThread-{self.cam_id}] Camera {self.cam_index} released.")


def read_camera(cam_thread: CameraThread):
    with cam_thread.lock:
        return (
            cam_thread.frame.copy() if cam_thread.frame is not None else None,
            cam_thread.world_landmarks,
            cam_thread.pose_landmarks,
            cam_thread.tracking,
            list(cam_thread.hand_landmarks),
            cam_thread.hand_open,
            list(cam_thread.hand_curl_ratios) if cam_thread.hand_curl_ratios else None,
        )


def tile_frames(frames: list, widths: list) -> np.ndarray:
    resized = []
    for f, tw in zip(frames, widths):
        if f is None:
            f = np.zeros((360, tw, 3), dtype=np.uint8)
        h, w = f.shape[:2]
        th = max(1, int(h * tw / w))
        resized.append(cv2.resize(f, (tw, th)))
    max_h = max(r.shape[0] for r in resized)
    padded = []
    for r in resized:
        dh = max_h - r.shape[0]
        if dh:
            r = np.vstack([r, np.zeros((dh, r.shape[1], 3), dtype=np.uint8)])
        padded.append(r)
    return np.hstack(padded)


# ── experiment factory helpers ────────────────────────────────────────────────
def make_reach_experiment(sim, robot_shoulder_world):
    return Experiment.from_hemisphere(
        sim,
        shoulder_pos=robot_shoulder_world,
        arm_length=ROBOT_ARM_LENGTH,
        n_trials=EXP_N_TRIALS,
        radius=EXP_RADIUS,
        dwell_time=EXP_DWELL_TIME,
        timeout=EXP_TIMEOUT,
        min_reach=EXP_MIN_REACH,
        max_reach=EXP_MAX_REACH,
        min_elevation=EXP_MIN_ELEVATION,
        max_elevation=EXP_MAX_ELEVATION,
        az_min=EXP_AZ_MIN,
        az_max=EXP_AZ_MAX,
        seed=EXP_SEED,
    )


def make_transport_experiment(sim, robot_shoulder_world, start_pos=None):
    return TransportExperiment.from_random(
        sim,
        shoulder_pos=robot_shoulder_world,
        arm_length=ROBOT_ARM_LENGTH,
        n_trials=TRANSPORT_N_TRIALS,
        pick_radius=TRANSPORT_PICK_RADIUS,
        drop_radius=TRANSPORT_DROP_RADIUS,
        timeout=TRANSPORT_TIMEOUT,
        min_reach=TRANSPORT_MIN_REACH,
        max_reach=TRANSPORT_MAX_REACH,
        min_elevation=TRANSPORT_MIN_ELEV,
        max_elevation=TRANSPORT_MAX_ELEV,
        az_min=TRANSPORT_AZ_MIN,
        az_max=TRANSPORT_AZ_MAX,
        seed=TRANSPORT_SEED,
        start_pos=start_pos,
    )


def save_results(experiment, mode):
    """Persist experiment results to a timestamped CSV."""
    results = experiment.results
    if not results:
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"poseEstimation{mode.capitalize()}Results/{mode}_results_{ts}.csv"

    if mode == MODE_REACH:
        fieldnames = [
            "trial",
            "label",
            "result",
            "duration_s",
            "target_x",
            "target_y",
            "target_z",
        ]
        trial_defs = {i + 1: t for i, t in enumerate(experiment._trial_defs)}
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                pos = trial_defs.get(r["trial"], {}).get("pos", [None, None, None])
                writer.writerow(
                    {
                        "trial": r["trial"],
                        "label": r["label"],
                        "result": r["result"],
                        "duration_s": round(r["duration"], 3),
                        "target_x": round(pos[0], 4) if pos[0] is not None else "",
                        "target_y": round(pos[1], 4) if pos[1] is not None else "",
                        "target_z": round(pos[2], 4) if pos[2] is not None else "",
                    }
                )
    else:
        fieldnames = [
            "trial",
            "label",
            "result",
            "duration_s",
            "cube_x",
            "cube_y",
            "cube_z",
            "drop_x",
            "drop_y",
            "drop_z",
            "start_x",
            "start_y",
            "start_z",  # ← add
            "dist_start_to_cube",
            "dist_start_to_drop",  # ← add
            "phase_approach_s",
            "phase_grip_s",  # ← add
            "phase_carry_s",
            "phase_place_s",  # ← add
        ]
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                cp = r.get("cube_pos", [None, None, None])
                dp = r.get("drop_pos", [None, None, None])
                sp = r.get("phase_splits", {})
                spos = r.get("start_pos") or [None, None, None]
                writer.writerow(
                    {
                        "trial": r["trial"],
                        "label": r["label"],
                        "result": r["result"],
                        "duration_s": round(r["duration"], 3),
                        "cube_x": round(cp[0], 4) if cp[0] is not None else "",
                        "cube_y": round(cp[1], 4) if cp[1] is not None else "",
                        "cube_z": round(cp[2], 4) if cp[2] is not None else "",
                        "drop_x": round(dp[0], 4) if dp[0] is not None else "",
                        "drop_y": round(dp[1], 4) if dp[1] is not None else "",
                        "drop_z": round(dp[2], 4) if dp[2] is not None else "",
                        "start_x": round(spos[0], 4) if spos[0] is not None else "",
                        "start_y": round(spos[1], 4) if spos[1] is not None else "",
                        "start_z": round(spos[2], 4) if spos[2] is not None else "",
                        "dist_start_to_cube": (
                            round(r["dist_start_to_cube"], 4)
                            if r.get("dist_start_to_cube") is not None
                            else ""
                        ),
                        "dist_start_to_drop": (
                            round(r["dist_start_to_drop"], 4)
                            if r.get("dist_start_to_drop") is not None
                            else ""
                        ),
                        "phase_approach_s": round(sp.get("approach", 0.0), 3),
                        "phase_grip_s": round(sp.get("grip", 0.0), 3),
                        "phase_carry_s": round(sp.get("carry", 0.0), 3),
                        "phase_place_s": round(sp.get("place", 0.0), 3),
                    }
                )
    print(f"Results saved to {filename}")
    print(experiment.summary())


# ── CoppeliaSim setup ─────────────────────────────────────────────────────────
print("Connecting to CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require("sim")
simIK = client.require("simIK")

rightShoulderAbduct = sim.getObject("/rightJoint1")
rightWristLink = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2"
    "/rightJoint3/rightLink3/rightJoint4/rightLink4"
    "/rightJoint5/rightLink5/rightJoint6/rightLink6/rightJoint7/rightLink7"
)

rightGripperObject = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2"
    "/rightJoint3/rightLink3/rightJoint4/rightLink4"
    "/rightJoint5/rightLink5/rightJoint6/rightLink6/rightJoint7/rightLink7/rightConnector/YuMiGripper/centerJoint/leftFinger"
)

target = sim.createDummy(0.02)
sim.setObjectAlias(target, "WristTarget")

robot_shoulder_world = sim.getObjectPosition(rightShoulderAbduct, -1)
print(f"Robot shoulder origin: {robot_shoulder_world}")

ikEnv = simIK.createEnvironment()
ikGroupUndamped = simIK.createGroup(ikEnv)
simIK.setGroupCalculation(ikEnv, ikGroupUndamped, simIK.method_pseudo_inverse, 0, 6)
simIK.addElementFromScene(
    ikEnv,
    ikGroupUndamped,
    rightShoulderAbduct,
    rightWristLink,
    target,
    simIK.constraint_position,
)

ikGroupDamped = simIK.createGroup(ikEnv)
simIK.setGroupCalculation(
    ikEnv, ikGroupDamped, simIK.method_damped_least_squares, 1, 99
)
simIK.addElementFromScene(
    ikEnv,
    ikGroupDamped,
    rightShoulderAbduct,
    rightWristLink,
    target,
    simIK.constraint_position,
)

# ── start camera threads ──────────────────────────────────────────────────────
cam_threads = [CameraThread(idx, i) for i, idx in enumerate(CAMERA_INDICES)]
for ct in cam_threads:
    ct.start()

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

try:
    print("Starting simulation...")
    sim.setStepping(True)
    sim.startSimulation()
    sim.setInt32Signal(GRIPPER_SIGNAL, 0)  # start open
    print("Simulation started OK")

    robot_shoulder_world = sim.getObjectPosition(rightShoulderAbduct, -1)

    calibrator = ArmCalibrator(ROBOT_ARM_LENGTH)
    pose_filter = PoseFilter(alpha=POSE_SMOOTH_ALPHA)
    wrist_pos = list(robot_shoulder_world)
    prev_time = cv2.getTickCount()

    # ── experiment state ──────────────────────────────────────────────────────
    current_mode = MODE_SELECT  # start at the menu
    experiment = None
    summary_printed = False
    gripper_open = True  # default: hand open

    print("Press [R] for Reach experiment, [T] for Transport, [Q] to quit.")
    print("Stretch your arm fully to calibrate reach mapping.")
    print("Open hand = gripper open | Closed fist = gripper closed.")

    # ── main loop ─────────────────────────────────────────────────────────────
    while True:
        now = cv2.getTickCount()
        dt = (now - prev_time) / cv2.getTickFrequency()
        prev_time = now

        snapshots = [read_camera(ct) for ct in cam_threads]

        # ── gripper state: OR gate across all cameras ─────────────────────────
        hand_states = [snap[5] for snap in snapshots if snap[5] is not None]
        if hand_states:
            new_gripper_open = all(hand_states)
            if new_gripper_open != gripper_open:
                gripper_open = new_gripper_open
                sim.setInt32Signal(GRIPPER_SIGNAL, 0 if gripper_open else 1)

        # ── pose fusion ───────────────────────────────────────────────────────
        ordered = [PRIMARY_CAMERA] + [
            i for i in range(len(cam_threads)) if i != PRIMARY_CAMERA
        ]

        target_pos = None
        target_quat = None
        source_idx = None

        for ci in ordered:
            frame, wl_world, wl_img, tracking, hand_lms, hand_open_ci, _curl = (
                snapshots[ci]
            )
            if not tracking:
                continue
            wl = wl_world.landmark
            hs = vec3(wl[RIGHT_SHOULDER])
            he = vec3(wl[RIGHT_ELBOW])
            hw = vec3(wl[RIGHT_WRIST])

            if ci == source_idx or source_idx is None:
                calibrator.update(hs, he, hw, dt)

            target_pos = retarget(
                hs,
                hw,
                robot_shoulder_world,
                ROBOT_ARM_LENGTH,
                human_scale=calibrator.scale,
            )
            target_quat = compute_wrist_quaternion(hs, he, hw)
            source_idx = ci
            break

        # ── smooth pose before sending to IK ─────────────────────────────────
        target_pos = pose_filter.update_pos(target_pos)
        target_quat = pose_filter.update_quat(target_quat)

        # ── IK ────────────────────────────────────────────────────────────────
        if target_pos:
            sim.setObjectPosition(target, target_pos)
        if target_quat:
            sim.setObjectQuaternion(target, target_quat)
        if target_pos or target_quat:
            res, *_ = simIK.handleGroup(ikEnv, ikGroupUndamped, {"syncWorlds": True})
            if res != simIK.result_success:
                simIK.handleGroup(ikEnv, ikGroupDamped, {"syncWorlds": True})

        sim.step()
        wrist_pos = sim.getObjectPosition(rightGripperObject, -1)

        # ── experiment update ─────────────────────────────────────────────────
        if experiment is not None and current_mode != MODE_SELECT:
            if current_mode == MODE_REACH:
                experiment.update(wrist_pos, dt)
            else:
                # gripper_open comes from hand gesture detection (open palm = True)
                experiment.update(wrist_pos, gripper_open, dt)

            if experiment.finished and not summary_printed:
                print(experiment.summary())
                save_results(experiment, current_mode)
                summary_printed = True

        # ── render ────────────────────────────────────────────────────────────
        display_frames = []
        tile_widths = []

        mp_hands = mp.solutions.hands
        mp_draw_hand = mp.solutions.drawing_utils
        hand_draw_spec = mp_draw_hand.DrawingSpec(
            color=(255, 200, 0), thickness=1, circle_radius=2
        )
        hand_conn_spec = mp_draw_hand.DrawingSpec(color=(200, 150, 0), thickness=1)

        for ci, (
            frame,
            wl_world,
            wl_img,
            tracking,
            hand_lms,
            hand_open_ci,
            curl_ratios,
        ) in enumerate(snapshots):
            is_hud_cam = ci == HUD_CAMERA
            tw = TILE_WIDTH if is_hud_cam else SECONDARY_TILE_WIDTH

            if frame is None:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)

            # Pose skeleton
            if wl_img is not None:
                mp_draw.draw_landmarks(
                    frame,
                    wl_img,
                    mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3
                    ),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
                )

            # Hand skeleton overlay
            for hl in hand_lms:
                mp_draw_hand.draw_landmarks(
                    frame,
                    hl,
                    mp_hands.HAND_CONNECTIONS,
                    hand_draw_spec,
                    hand_conn_spec,
                )

            # Curl meter — bottom-left on HUD cam, top-left on secondary cams
            if curl_ratios is not None:
                H_f = frame.shape[0]
                if is_hud_cam:
                    meter_origin = (10, H_f - 120)
                    draw_curl_meter(
                        frame, curl_ratios, meter_origin, label_prefix="finger curl"
                    )
                else:
                    draw_curl_meter(frame, curl_ratios, (4, 30), label_prefix="curl")

            is_active = ci == source_idx
            label_txt = f"Cam {ci}" + (" [ACTIVE]" if is_active else "")
            status_col = (0, 255, 0) if tracking else (0, 0, 255)

            if is_hud_cam:
                cv2.putText(
                    frame,
                    label_txt,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_col,
                    2,
                )
                cv2.putText(
                    frame,
                    "IK: tracking" if tracking else "IK: no pose",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    status_col,
                    2,
                )
                if target_pos and is_active:
                    cv2.putText(
                        frame,
                        f"Target: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})",
                        (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                    )
                # Calibration status
                if calibrator.calibrated:
                    cal_txt = f"Reach cal: {calibrator.human_max_reach*100:.1f} cm  scale={calibrator.scale:.2f}"
                    cal_col = (0, 220, 100)
                else:
                    cal_txt = (
                        f"Reach cal: stretch arm to calibrate "
                        f"({calibrator._samples}/{calibrator.MIN_SAMPLES})"
                    )
                    cal_col = (0, 180, 220)
                cv2.putText(
                    frame,
                    cal_txt,
                    (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    cal_col,
                    1,
                )

                # Gripper state indicator — OR gate across all cameras
                seeing = [i for i, snap in enumerate(snapshots) if snap[5] is not None]
                cam_note = (
                    f"  (cam {','.join(str(i) for i in seeing)})" if seeing else ""
                )
                if not hand_states:
                    g_txt = "HAND: not detected"
                    g_col = (80, 80, 80)
                elif gripper_open:
                    g_txt = f"HAND: OPEN  [gripper open]{cam_note}"
                    g_col = _cv_col(100, 220, 130)
                else:
                    g_txt = f"HAND: CLOSED  [gripper closed]{cam_note}"
                    g_col = _cv_col(100, 180, 255)
                cv2.putText(
                    frame, g_txt, (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.45, g_col, 1
                )

                # Experiment overlay
                if current_mode == MODE_SELECT:
                    draw_mode_select_hud(frame)
                elif experiment is not None:
                    draw_experiment_hud(frame, experiment, wrist_pos, dt, current_mode)
            else:
                cv2.putText(
                    frame,
                    label_txt,
                    (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    status_col,
                    1,
                )

            display_frames.append(frame)
            tile_widths.append(tw)

        combined = tile_frames(display_frames, tile_widths)
        cv2.imshow("YuMi Pose Control — Multi-Camera", combined)

        # ── key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            # Save current results before switching
            if experiment is not None and experiment.results:
                save_results(experiment, current_mode)
            print("Starting Reach experiment...")
            experiment = make_reach_experiment(sim, robot_shoulder_world)
            current_mode = MODE_REACH
            summary_printed = False
            print(f"  {EXP_N_TRIALS} targets placed on reachable hemisphere.")
        elif key == ord("t"):
            if experiment is not None and experiment.results:
                save_results(experiment, current_mode)
            print("Starting Transport experiment...")
            experiment = make_transport_experiment(
                sim, robot_shoulder_world, start_pos=wrist_pos
            )
            current_mode = MODE_TRANSPORT
            summary_printed = False
            print(f"  {TRANSPORT_N_TRIALS} pick-and-place tasks created.")

except KeyboardInterrupt:
    print("Interrupted.")

except Exception as e:
    import traceback

    print("\n── FATAL ERROR ──────────────────────────────")
    traceback.print_exc()
    print("─────────────────────────────────────────────\n")

finally:
    for ct in cam_threads:
        ct.stop()
    for ct in cam_threads:
        ct.join(timeout=2.0)

    cv2.destroyAllWindows()
    sim.clearInt32Signal(GRIPPER_SIGNAL)
    sim.stopSimulation()
    print("Simulation stopped.")

    # Save any remaining results
    if "experiment" in dir() and experiment is not None and experiment.results:
        if not summary_printed:
            save_results(experiment, current_mode)
    else:
        print("No results to save.")
