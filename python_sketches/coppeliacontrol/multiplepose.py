"""
YuMi Multi-Camera Pose Control  ·  threaded edition
────────────────────────────────────────────────────
Each camera runs in its own daemon thread and deposits its latest frame +
computed joint-angle dict into a shared CameraSlot.  The main thread reads
all slots, averages angles across cameras that are actively tracking, drives
the robot, steps the simulation, and renders:

  • One small window per camera  (raw feed + skeleton + "CAM N" label)
  • One rich experiment window   (experiment HUD  +  arc-gauge angle panel)

Configuration
─────────────
Edit CAMERA_INDICES to add / remove cameras.  Any index that cannot be opened
is skipped with a warning.
"""

from __future__ import annotations

import csv
import datetime
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from reach_experiment import Experiment

# ── constants ─────────────────────────────────────────────────────────────────
RADIAN_TO_DEGREES = 180 / math.pi
DEGREES_TO_RADIANS = math.pi / 180

RIGHT_HIP = 24
LEFT_HIP = 23
RIGHT_SMALL_FINGY = 18
RIGHT_BIG_FINGY = 20
RIGHT_SHOULDER = 12
LEFT_SHOULDER = 11
RIGHT_ELBOW = 14
RIGHT_WRIST = 16

ROBOT_ARM_LENGTH = 0.21492 + 0.24129

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

# ── camera configuration ──────────────────────────────────────────────────────
# List the camera indices you want to use.  e.g. [0, 1, 2] for three cameras.
CAMERA_INDICES: list[int] = [0, 1]

# Experiment window dimensions
EXP_WIN_W = 1100
EXP_WIN_H = 680
GAUGE_PANEL_W = 380  # right-side gauge panel width


# ═══════════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _v(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z])


def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b - a


def _mag(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))


def _angle_vv(a: np.ndarray, b: np.ndarray, degrees=False) -> float:
    cos = np.dot(a, b) / (_mag(a) * _mag(b))
    th = math.acos(np.clip(cos, -1.0, 1.0))
    return th * RADIAN_TO_DEGREES if degrees else th


def _angle_3pts(a, b, c, degrees=False) -> float:
    return _angle_vv(_vec(b, a), _vec(b, c), degrees)


def _mid(a, b) -> np.ndarray:
    return (a + b) / 2


def _normal(a, b, c, unit=True) -> np.ndarray:
    n = np.cross(_vec(a, b), _vec(a, c))
    return n / _mag(n) if unit else n


def calc_joint_angles(data: dict) -> dict:
    hl, hr = data["HipL"], data["HipR"]
    el, wr = data["Elbow"], data["Wrist"]
    ix, pk = data["Index"], data["Pinky"]
    rs, ls = data["ShoulderR"], data["ShoulderL"]

    # Shoulder
    bp_n = _normal(ls, rs, _mid(hl, hr))
    ua = _vec(rs, el)
    rv = _vec(ls, rs)
    dv = np.cross(bp_n, rv)
    sf = math.atan2(
        np.dot(ua, bp_n), math.sqrt(np.dot(ua, rv) ** 2 + np.dot(ua, dv) ** 2)
    )
    sa = math.atan2(np.dot(ua, rv), np.dot(ua, dv))

    # Roll
    uan = ua / _mag(ua)
    hn = _normal(wr, ix, pk)
    hnp = hn - np.dot(hn, uan) * uan
    hnp /= _mag(hnp)
    brp = rv - np.dot(rv, uan) * uan
    brp /= _mag(brp)
    roll = _angle_vv(hnp, brp, True)

    # Wrist deviation
    fp_n = _normal(rs, el, wr)
    fv = _vec(el, wr)
    hv = _vec(wr, ix)
    hip = hv - np.dot(hv, fp_n) * fp_n
    wd = _angle_vv(fv, hip, True)

    return {
        "shoulder_flexion": sf * RADIAN_TO_DEGREES,
        "shoulder_abduction": sa * RADIAN_TO_DEGREES,
        "elbow_flexion": _angle_3pts(rs, el, wr) * RADIAN_TO_DEGREES,
        "wrist_flexion": _angle_3pts(el, wr, ix) * RADIAN_TO_DEGREES,
        "roll": roll,
        "wrist_deviation": wd,
    }


def avg_angle_dicts(dicts: list[dict]) -> dict:
    if not dicts:
        return {}
    return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}


# ═══════════════════════════════════════════════════════════════════════════════
#  Thread-safe camera slot
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CameraSlot:
    cam_idx: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    frame: Optional[np.ndarray] = None  # annotated BGR frame
    angles: Optional[dict] = None  # joint angles, or None if no pose
    tracking: bool = False
    alive: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera worker thread
# ═══════════════════════════════════════════════════════════════════════════════


def camera_worker(slot: CameraSlot, stop_event: threading.Event) -> None:
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(slot.cam_idx + cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[CAM {slot.cam_idx}] WARNING: could not open — thread exiting.")
        with slot.lock:
            slot.alive = False
        return

    pose = mp_pose.Pose(
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            angles = None
            tracking = False

            if results.pose_world_landmarks:
                tracking = True
                wl = results.pose_world_landmarks.landmark
                pose_data = {
                    "ShoulderR": _v(wl[RIGHT_SHOULDER]),
                    "ShoulderL": _v(wl[LEFT_SHOULDER]),
                    "HipR": _v(wl[RIGHT_HIP]),
                    "HipL": _v(wl[LEFT_HIP]),
                    "Index": _v(wl[RIGHT_BIG_FINGY]),
                    "Pinky": _v(wl[RIGHT_SMALL_FINGY]),
                    "Elbow": _v(wl[RIGHT_ELBOW]),
                    "Wrist": _v(wl[RIGHT_WRIST]),
                }
                try:
                    angles = calc_joint_angles(pose_data)
                except Exception:
                    angles = None

            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3
                    ),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
                )

            # Camera label banner
            status = "TRACKING" if tracking else "NO POSE"
            col = (0, 230, 80) if tracking else (0, 60, 220)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (10, 10, 15), -1)
            cv2.putText(
                frame,
                f"CAM {slot.cam_idx}  {status}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                col,
                2,
            )

            with slot.lock:
                slot.frame = frame.copy()
                slot.angles = angles
                slot.tracking = tracking

    finally:
        cap.release()
        pose.close()
        with slot.lock:
            slot.alive = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Angle gauge panel
# ═══════════════════════════════════════════════════════════════════════════════

# (key, display_label, range_min, range_max, neutral)
GAUGE_DEFS = [
    ("shoulder_flexion", "Shoulder Flex", -60, 180, 0),
    ("shoulder_abduction", "Shoulder Abduct", -30, 130, 0),
    ("elbow_flexion", "Elbow Flex", 0, 180, 90),
    ("wrist_flexion", "Wrist Flex", 0, 180, 90),
    ("roll", "Arm Roll", 0, 180, 90),
    ("wrist_deviation", "Wrist Dev", 0, 90, 0),
]

# Colour palette (BGR throughout)
C_BG = (14, 14, 20)
C_PANEL_BG = (22, 22, 30)
C_BORDER = (50, 50, 70)
C_LABEL = (180, 180, 200)
C_VALUE = (240, 240, 255)
C_TRACK_BG = (40, 40, 55)
C_ARC_HOT = (80, 200, 100)
C_ARC_COLD = (100, 130, 220)
C_MULTI = (50, 200, 230)
C_NEUTRAL = (80, 80, 100)
C_TITLE = (120, 200, 160)


def _cv_col(r, g, b):
    return (b, g, r)


def _lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def draw_arc_gauge(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    value_deg: float,
    min_deg: float,
    max_deg: float,
    neutral_deg: float,
    label: str,
    n_cams: int = 1,
) -> None:
    """Draw a top-semicircle arc gauge at (cx, cy)."""
    span = max_deg - min_deg
    frac = float(np.clip((value_deg - min_deg) / span, 0.0, 1.0))
    neutral_frac = float(np.clip((neutral_deg - min_deg) / span, 0.0, 1.0))

    # OpenCV convention: 180° = left, 360° = right (top semicircle sweep)
    g_start, g_end = 180, 360

    # Track
    cv2.ellipse(
        canvas,
        (cx, cy),
        (radius, radius),
        0,
        g_start,
        g_end,
        C_TRACK_BG,
        6,
        cv2.LINE_AA,
    )

    # Filled arc
    filled_end = int(g_start + frac * (g_end - g_start))
    arc_col = _lerp(C_ARC_HOT, C_ARC_COLD, frac)
    if n_cams > 1:
        arc_col = _lerp(arc_col, C_MULTI, 0.35)
    if filled_end > g_start:
        cv2.ellipse(
            canvas,
            (cx, cy),
            (radius, radius),
            0,
            g_start,
            filled_end,
            arc_col,
            6,
            cv2.LINE_AA,
        )

    # Neutral tick mark
    na_rad = math.radians(g_start + neutral_frac * (g_end - g_start))
    cv2.line(
        canvas,
        (
            int(cx + (radius - 10) * math.cos(na_rad)),
            int(cy + (radius - 10) * math.sin(na_rad)),
        ),
        (
            int(cx + (radius + 10) * math.cos(na_rad)),
            int(cy + (radius + 10) * math.sin(na_rad)),
        ),
        C_NEUTRAL,
        2,
        cv2.LINE_AA,
    )

    # Needle tip
    nn_rad = math.radians(g_start + frac * (g_end - g_start))
    nx = int(cx + radius * math.cos(nn_rad))
    ny = int(cy + radius * math.sin(nn_rad))
    cv2.circle(canvas, (nx, ny), 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (nx, ny), 5, arc_col, 1, cv2.LINE_AA)

    # Label (below arc midpoint)
    lw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
    cv2.putText(
        canvas,
        label,
        (cx - lw // 2, cy + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        C_LABEL,
        1,
        cv2.LINE_AA,
    )

    # Value (above arc)
    val_str = f"{value_deg:+.1f}\u00b0"
    vw = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    cv2.putText(
        canvas,
        val_str,
        (cx - vw // 2, cy - radius - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        C_VALUE,
        1,
        cv2.LINE_AA,
    )


def build_angle_panel(
    angles: dict, panel_w: int, panel_h: int, n_cams: int
) -> np.ndarray:
    panel = np.full((panel_h, panel_w, 3), C_PANEL_BG, dtype=np.uint8)

    title = "JOINT ANGLES"
    tw = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
    cv2.putText(
        panel,
        title,
        (panel_w // 2 - tw // 2, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        C_TITLE,
        2,
        cv2.LINE_AA,
    )

    sub = (
        f"avg of {n_cams} camera{'s' if n_cams!=1 else ''}"
        if n_cams
        else "no pose detected"
    )
    sw = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
    cv2.putText(
        panel,
        sub,
        (panel_w // 2 - sw // 2, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        C_MULTI if n_cams > 1 else C_NEUTRAL,
        1,
        cv2.LINE_AA,
    )

    cv2.line(panel, (16, 54), (panel_w - 16, 54), C_BORDER, 1)

    if not angles:
        msg = "Waiting for pose..."
        mw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        cv2.putText(
            panel,
            msg,
            (panel_w // 2 - mw // 2, panel_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            C_NEUTRAL,
            1,
            cv2.LINE_AA,
        )
        return panel

    cols, rows = 2, 3
    cell_w = panel_w // cols
    cell_h = (panel_h - 60) // rows
    radius = min(cell_w, cell_h) // 2 - 22

    for i, (key, label, mn, mx, neutral) in enumerate(GAUGE_DEFS):
        col = i % cols
        row = i // cols
        cx = cell_w * col + cell_w // 2
        cy = 60 + cell_h * row + cell_h // 2 + 8
        draw_arc_gauge(
            panel, cx, cy, radius, angles.get(key, 0.0), mn, mx, neutral, label, n_cams
        )

    return panel


# ═══════════════════════════════════════════════════════════════════════════════
#  Experiment HUD
# ═══════════════════════════════════════════════════════════════════════════════


def draw_experiment_hud(frame: np.ndarray, experiment, wrist_pos) -> None:
    H, W = frame.shape[:2]
    active = experiment._active
    total = len(experiment._trial_defs)
    done = len(experiment.results)

    # Progress bar
    bar_y = H - 14
    cv2.rectangle(frame, (10, bar_y), (W - 10, bar_y + 8), (45, 45, 55), -1)
    if total and done:
        cv2.rectangle(
            frame,
            (10, bar_y),
            (10 + int((W - 20) * done / total), bar_y + 8),
            _cv_col(100, 220, 130),
            -1,
        )
    cv2.putText(
        frame,
        f"Trial {min(done+1,total)} / {total}",
        (10, bar_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (130, 130, 130),
        1,
    )

    # Finished overlay
    if experiment.finished:
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (W, H), (18, 10, 10), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
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
        return

    if active is None:
        return

    dist = active.distance_to(wrist_pos)
    ratio = min(1.0, dist / (active.radius * 6))

    hud_rgb = (
        _lerp(_cv_col(100, 220, 130), _cv_col(220, 190, 60), ratio / 0.4)
        if ratio < 0.4
        else _lerp(_cv_col(220, 190, 60), _cv_col(100, 160, 220), (ratio - 0.4) / 0.6)
    )

    # Flash overlay on result
    if active._flash_t > 0:
        fr = (100, 220, 130) if active._result == "success" else (220, 80, 80)
        alpha = active._flash_t / 0.6
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (W, H), _cv_col(*fr), -1)
        cv2.addWeighted(ov, alpha * 0.45, frame, 1 - alpha * 0.45, 0, frame)
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

    # Info panel
    px, py = 10, 52  # below title strip
    hud_col = hud_rgb  # already BGR from _lerp of _cv_col results
    cv2.rectangle(frame, (px - 4, py - 4), (px + 270, py + 120), (22, 22, 28), -1)
    cv2.rectangle(frame, (px - 4, py - 4), (px + 270, py + 120), hud_col, 1)
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

    bw = 255
    fill = int(bw * (1.0 - ratio))
    cv2.rectangle(frame, (px, py + 50), (px + bw, py + 58), (55, 55, 55), -1)
    if fill > 0:
        cv2.rectangle(frame, (px, py + 50), (px + fill, py + 58), hud_col, -1)

    # Dwell arc
    if active._inside and active.dwell_fraction > 0:
        ccx, ccy = px + 240, py + 88
        r = 18
        cv2.circle(frame, (ccx, ccy), r, (55, 55, 55), 2)
        cv2.ellipse(
            frame,
            (ccx, ccy),
            (r, r),
            -90,
            0,
            int(360 * active.dwell_fraction),
            _cv_col(100, 220, 130),
            2,
        )
        pct = f"{int(active.dwell_fraction*100)}%"
        cv2.putText(
            frame,
            pct,
            (ccx - 14, ccy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            _cv_col(100, 220, 130),
            1,
        )

    # Timer
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    # ── CoppeliaSim ──────────────────────────────────────────────────────────
    print("Connecting to CoppeliaSim...")
    client = RemoteAPIClient()
    sim = client.require("sim")
    print("Connected.")

    rShoulderAbduct = sim.getObject("/rightJoint1")
    rShoulderFlex = sim.getObject("/rightJoint1/rightLink1/rightJoint2")
    rForearmRoll = sim.getObject(
        "/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3"
    )
    rElbowFlex = sim.getObject(
        "/rightJoint1/rightLink1/rightJoint2/rightLink2"
        "/rightJoint3/rightLink3/rightJoint4/"
    )
    rWristLink = sim.getObject(
        "/rightJoint1/rightLink1/rightJoint2/rightLink2"
        "/rightJoint3/rightLink3/rightJoint4/rightLink4"
        "/rightJoint5/rightLink5/rightJoint6/rightLink6"
        "/rightJoint7/rightLink7"
    )

    sim.setStepping(True)
    sim.startSimulation()
    print("Simulation started.")

    shoulder_world = sim.getObjectPosition(rShoulderAbduct, -1)
    experiment = Experiment.from_hemisphere(
        sim,
        shoulder_pos=shoulder_world,
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
    print(f"Experiment: {EXP_N_TRIALS} targets.")

    # ── Launch camera threads ─────────────────────────────────────────────────
    stop_event = threading.Event()
    slots: list[CameraSlot] = []
    threads: list[threading.Thread] = []

    for idx in CAMERA_INDICES:
        slot = CameraSlot(cam_idx=idx)
        t = threading.Thread(
            target=camera_worker,
            args=(slot, stop_event),
            daemon=True,
            name=f"cam-{idx}",
        )
        t.start()
        slots.append(slot)
        threads.append(t)
        print(f"Camera thread started for index {idx}.")

    time.sleep(0.8)  # let threads open their cameras

    # ── Create windows ────────────────────────────────────────────────────────
    EXP_WIN = "YuMi — Experiment"
    cv2.namedWindow(EXP_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(EXP_WIN, EXP_WIN_W, EXP_WIN_H)

    cam_wins: dict[int, str] = {}
    for slot in slots:
        name = f"YuMi — CAM {slot.cam_idx}"
        cam_wins[slot.cam_idx] = name
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 640, 400)

    wrist_pos = list(shoulder_world)
    prev_tick = cv2.getTickCount()
    summary_printed = False

    try:
        while True:
            now = cv2.getTickCount()
            dt = (now - prev_tick) / cv2.getTickFrequency()
            prev_tick = now

            # ── Collect slot data ─────────────────────────────────────────────
            angle_dicts: list[dict] = []
            cam_frames: list[tuple[int, np.ndarray]] = []

            for slot in slots:
                with slot.lock:
                    frame = slot.frame
                    angles = slot.angles
                with slot.lock:
                    pass  # lightweight double-check (frame already copied in thread)

                if frame is not None:
                    cam_frames.append((slot.cam_idx, frame.copy()))
                if angles is not None:
                    angle_dicts.append(angles)

            # ── Average and drive robot ───────────────────────────────────────
            avg = avg_angle_dicts(angle_dicts)
            if avg:
                sim.setJointTargetPosition(
                    rShoulderFlex, (avg["shoulder_flexion"] - 90) * DEGREES_TO_RADIANS
                )
                sim.setJointTargetPosition(
                    rForearmRoll, (-avg["roll"] + 90) * DEGREES_TO_RADIANS
                )
                sim.setJointTargetPosition(
                    rShoulderAbduct,
                    (130 - avg["shoulder_abduction"]) * DEGREES_TO_RADIANS,
                )
                sim.setJointTargetPosition(
                    rElbowFlex, (90 - avg["elbow_flexion"]) * DEGREES_TO_RADIANS
                )

            sim.step()
            wrist_pos = sim.getObjectPosition(rWristLink, -1)
            experiment.update(wrist_pos, dt)

            if experiment.finished and not summary_printed:
                print(experiment.summary())
                summary_printed = True

            # ── Individual camera windows ─────────────────────────────────────
            for cam_idx, frame in cam_frames:
                win = cam_wins.get(cam_idx)
                if win:
                    cv2.imshow(win, frame)

            # ── Experiment window ─────────────────────────────────────────────
            hud_w = EXP_WIN_W - GAUGE_PANEL_W
            hud = np.full((EXP_WIN_H, hud_w, 3), C_BG, dtype=np.uint8)

            # Title strip
            cv2.rectangle(hud, (0, 0), (hud_w, 46), (18, 22, 28), -1)
            cv2.putText(
                hud,
                "YuMi REACH EXPERIMENT",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                C_TITLE,
                2,
                cv2.LINE_AA,
            )
            n_det = len(angle_dicts)
            det_str = f"cameras active: {n_det}/{len(slots)}"
            det_col = C_MULTI if n_det > 1 else (130, 130, 130)
            cv2.putText(
                hud,
                det_str,
                (hud_w - 220, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                det_col,
                1,
            )
            cv2.line(hud, (0, 46), (hud_w, 46), C_BORDER, 1)

            draw_experiment_hud(hud, experiment, wrist_pos)

            # Gauge panel (right side)
            gauge = build_angle_panel(avg, GAUGE_PANEL_W, EXP_WIN_H, n_det)
            cv2.line(gauge, (0, 0), (0, EXP_WIN_H), C_BORDER, 2)

            cv2.imshow(EXP_WIN, np.hstack([hud, gauge]))

            # ── Key ───────────────────────────────────────────────────────────
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
        cv2.destroyAllWindows()
        sim.stopSimulation()
        print("Simulation stopped.")

        # Save CSV
        results = experiment.results if experiment else []
        if results:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"poseEstimationReachResults/reach_results_{ts}.csv"
            fields = [
                "trial",
                "label",
                "result",
                "duration_s",
                "target_x",
                "target_y",
                "target_z",
            ]
            tdefs = {i + 1: t for i, t in enumerate(experiment._trial_defs)}
            with open(fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in results:
                    pos = tdefs.get(r["trial"], {}).get("pos", [None, None, None])
                    w.writerow(
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
            print(f"Results saved → {fname}")
            print(experiment.summary())
        else:
            print("No results to save.")


if __name__ == "__main__":
    main()
