"""
YuMi Pose Control — Multi-Camera Threaded Version
===================================================
Each camera runs MediaPipe pose detection in its own thread.
The main loop picks the best available pose (primary camera first,
falling back to secondary cameras if tracking is lost) and feeds it
to CoppeliaSim IK.

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

from reach_experiment import Experiment

# ── camera configuration ──────────────────────────────────────────────────────
CAMERA_INDICES = [0, 1]  # OpenCV device indices; edit to match your setup
PRIMARY_CAMERA = 0  # which index in CAMERA_INDICES is the preferred source
TILE_WIDTH = 640  # display width of the primary (HUD) camera tile
SECONDARY_TILE_WIDTH = 240  # display width of all other camera tiles
HUD_CAMERA = 0  # index within CAMERA_INDICES that gets the experiment HUD

# ── constants ─────────────────────────────────────────────────────────────────
DEG_TO_RAD = pi / 180

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16

ROBOT_ARM_LENGTH = 0.21492 + 0.24129

# ── experiment configuration ───────────────────────────────────────────────────
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
    return [-v[2], v[0], v[1]]


# ── body-relative normalisation ───────────────────────────────────────────────

LEFT_SHOULDER = 11
RIGHT_SHOULDER_LM = 12  # same as RIGHT_SHOULDER constant already defined
LEFT_HIP = 23
RIGHT_HIP = 24


def body_frame(wl):
    """
    Build a 3×3 rotation matrix (columns = right, up, forward) from
    the hip/shoulder quad so that arm vectors are camera-independent.
    Returns None if any landmark is missing or degenerate.
    """
    ls = vec3(wl[LEFT_SHOULDER])
    rs = vec3(wl[RIGHT_SHOULDER_LM])
    lh = vec3(wl[LEFT_HIP])
    rh = vec3(wl[RIGHT_HIP])

    right = vec_normalize(vec_sub(rs, ls))  # left→right shoulder
    if right is None:
        return None

    shoulder_mid = vec_scale(vec_add(ls, rs), 0.5)
    hip_mid = vec_scale(vec_add(lh, rh), 0.5)
    up = vec_normalize(vec_sub(shoulder_mid, hip_mid))  # hips→shoulders
    if up is None:
        return None

    right_np = np.array(right)
    up_np = np.array(up)

    # Re-orthogonalise: remove any right component from up
    up_np = up_np - np.dot(up_np, right_np) * right_np
    up_np /= np.linalg.norm(up_np)

    fwd_np = np.cross(right_np, up_np)  # points out of the person's chest

    # Columns: right, up, forward
    return np.column_stack([right_np, up_np, fwd_np])


def to_body_frame(v, R):
    """
    Project world-space vector v into the body frame defined by rotation matrix R.
    R columns are (right, up, forward), so R.T rotates world→body.
    """
    return (R.T @ np.array(v)).tolist()


# ── arm-length calibration ────────────────────────────────────────────────────
class ArmCalibrator:
    """
    Estimates the human's max arm reach from live pose data and exposes a scale
    factor so that human_max_reach maps to robot_arm_length.

    Strategy
    --------
    - Each frame we measure shoulder→elbow + elbow→wrist (matching the two robot
      segments) as the *anatomical* arm length, and shoulder→wrist as the
      *reach* length (straight-line extension).
    - We keep a rolling maximum of the reach length over a decay window so the
      calibration updates when the user stretches further, and slowly decays
      back down if they consistently reach less (e.g. after a break).
    - Until enough samples are collected the scale defaults to 1.0 (old
      normalised behaviour) so motion is never blocked at startup.
    """

    # How quickly the max decays toward the observed mean (fraction per second)
    DECAY_RATE = 0.002
    # Minimum samples before we trust the calibration
    MIN_SAMPLES = 30
    # Hard floor: human reach must be at least this many metres to count
    MIN_REACH_M = 0.15

    def __init__(self, robot_arm_length: float):
        self.robot_arm_length = robot_arm_length
        self._max_reach = 0.0
        self._samples = 0

    def update(self, human_shoulder, human_elbow, human_wrist, dt: float):
        """Call once per frame with raw MediaPipe world landmarks."""
        reach = vec_length(vec_sub(human_wrist, human_shoulder))
        if reach < self.MIN_REACH_M:
            return
        self._samples += 1
        if reach > self._max_reach:
            self._max_reach = reach
        else:
            # Gentle decay so calibration adapts downward over time
            self._max_reach -= self._max_reach * self.DECAY_RATE * dt

    @property
    def calibrated(self) -> bool:
        return self._samples >= self.MIN_SAMPLES and self._max_reach > self.MIN_REACH_M

    @property
    def scale(self) -> float:
        """robot_arm_length / human_max_reach  (or 1.0 before calibration)."""
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
    """
    Map human wrist to robot world space.

    With human_scale == 1.0 (default / uncalibrated) this normalises to the
    full robot arm length — identical to the old behaviour.

    With a calibrated human_scale (= robot_arm_length / human_max_reach) the
    reach is preserved proportionally: a half-extended human arm moves the
    robot to half its reach, a fully extended arm reaches the robot's maximum.
    """
    human_vec = vec_sub(human_wrist, human_shoulder)
    human_length = vec_length(human_vec)
    if human_length < 1e-6:
        return None
    # Scale the raw human reach into robot space, clamped to robot arm length
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


def draw_experiment_hud(frame, experiment, wrist_pos, dt):
    H, W = frame.shape[:2]
    active = experiment._active

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
        return

    if active is None:
        return

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
        flash_rgb = (100, 220, 130) if active._result == "success" else (220, 80, 80)
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

    px, py = W - 270, 10
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
    cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + 8), (55, 55, 55), -1)
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + fill, bar_y2 + 8), hud_col, -1)

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


# ── Camera thread ─────────────────────────────────────────────────────────────
class CameraThread(threading.Thread):
    """
    Captures frames and runs MediaPipe pose estimation in a background thread.

    Public attributes (protected by self.lock):
        frame          : latest BGR frame (or None)
        world_landmarks: latest pose_world_landmarks (or None)
        pose_landmarks : latest pose_landmarks for drawing (or None)
        tracking       : True when a pose is detected
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
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
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

        try:
            while not self._stop_event.is_set():
                ret, bgr = cap.read()
                if not ret:
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                with self.lock:
                    self.frame = bgr
                    self.world_landmarks = results.pose_world_landmarks
                    self.pose_landmarks = results.pose_landmarks
                    self.tracking = results.pose_world_landmarks is not None
        finally:
            cap.release()
            pose.close()
            print(f"[CamThread-{self.cam_id}] Camera {self.cam_index} released.")


def read_camera(cam_thread: CameraThread):
    """Return a thread-safe snapshot of (frame, world_landmarks, pose_landmarks, tracking)."""
    with cam_thread.lock:
        return (
            cam_thread.frame.copy() if cam_thread.frame is not None else None,
            cam_thread.world_landmarks,
            cam_thread.pose_landmarks,
            cam_thread.tracking,
        )


def tile_frames(frames: list, widths: list) -> np.ndarray:
    """
    Horizontally concatenate BGR frames, each resized to its corresponding width.
    `widths` must have the same length as `frames`.
    """
    resized = []
    for f, tw in zip(frames, widths):
        if f is None:
            f = np.zeros((360, tw, 3), dtype=np.uint8)
        h, w = f.shape[:2]
        th = max(1, int(h * tw / w))
        resized.append(cv2.resize(f, (tw, th)))
    # Pad all tiles to the same height
    max_h = max(r.shape[0] for r in resized)
    padded = []
    for r in resized:
        dh = max_h - r.shape[0]
        if dh:
            r = np.vstack([r, np.zeros((dh, r.shape[1], 3), dtype=np.uint8)])
        padded.append(r)
    return np.hstack(padded)


# ── CoppeliaSim setup ─────────────────────────────────────────────────────────
print("Connecting to CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require("sim")
simIK = client.require("simIK")

rightShoulderAbduct = sim.getObject("/rightJoint1")

# Elbow tip link — the last link before the wrist joints begin
rightElbowLink = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2"
    "/rightJoint3/rightLink3/rightJoint4/rightLink4"
)

# Wrist tip link — unchanged
rightWristLink = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2"
    "/rightJoint3/rightLink3/rightJoint4/rightLink4"
    "/rightJoint5/rightLink5/rightJoint6/rightLink6"
)

# Two dummy targets
elbowTarget = sim.createDummy(0.02)
sim.setObjectAlias(elbowTarget, "ElbowTarget")

wristTarget = sim.createDummy(0.02)
sim.setObjectAlias(wristTarget, "WristTarget")

robot_shoulder_world = sim.getObjectPosition(rightShoulderAbduct, -1)
print(f"Robot shoulder origin: {robot_shoulder_world}")

ikEnv = simIK.createEnvironment()

# ── Elbow IK groups ────────────────────────────────────────────────────────────
ikGroupElbowUndamped = simIK.createGroup(ikEnv)
simIK.setGroupCalculation(
    ikEnv, ikGroupElbowUndamped, simIK.method_pseudo_inverse, 0, 6
)
simIK.addElementFromScene(
    ikEnv,
    ikGroupElbowUndamped,
    rightShoulderAbduct,
    rightElbowLink,
    elbowTarget,
    simIK.constraint_position,
)

ikGroupElbowDamped = simIK.createGroup(ikEnv)
simIK.setGroupCalculation(
    ikEnv, ikGroupElbowDamped, simIK.method_damped_least_squares, 1, 99
)
simIK.addElementFromScene(
    ikEnv,
    ikGroupElbowDamped,
    rightShoulderAbduct,
    rightElbowLink,
    elbowTarget,
    simIK.constraint_position,
)

# ── Wrist IK groups ────────────────────────────────────────────────────────────
ikGroupWristUndamped = simIK.createGroup(ikEnv)
simIK.setGroupCalculation(
    ikEnv, ikGroupWristUndamped, simIK.method_pseudo_inverse, 0, 6
)
simIK.addElementFromScene(
    ikEnv,
    ikGroupWristUndamped,
    rightShoulderAbduct,
    rightWristLink,
    wristTarget,
    simIK.constraint_position,
)

ikGroupWristDamped = simIK.createGroup(ikEnv)
simIK.setGroupCalculation(
    ikEnv, ikGroupWristDamped, simIK.method_damped_least_squares, 1, 99
)
simIK.addElementFromScene(
    ikEnv,
    ikGroupWristDamped,
    rightShoulderAbduct,
    rightWristLink,
    wristTarget,
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
    print("Simulation started OK")

    robot_shoulder_world = sim.getObjectPosition(rightShoulderAbduct, -1)
    experiment = Experiment.from_hemisphere(
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
    print(
        f"Experiment created — {EXP_N_TRIALS} targets placed on reachable hemisphere."
    )
    for i, t in enumerate(experiment._trial_defs):
        print(f"  {i+1}. {t['pos']}")

    calibrator = ArmCalibrator(ROBOT_ARM_LENGTH)
    print(f"Running with {len(cam_threads)} camera(s) — press Q to quit.")
    print(
        "Stretch your arm fully to calibrate reach mapping (takes ~1 s of full extension)."
    )

    wrist_pos = list(robot_shoulder_world)
    prev_time = cv2.getTickCount()

    # ── main loop ─────────────────────────────────────────────────────────────
    while True:
        now = cv2.getTickCount()
        dt = (now - prev_time) / cv2.getTickFrequency()
        prev_time = now

        # -- collect latest frame + landmarks from each camera -----------------
        snapshots = [read_camera(ct) for ct in cam_threads]

        # -- pose fusion: primary first, then fallbacks ------------------------
        ordered = [PRIMARY_CAMERA] + [
            i for i in range(len(cam_threads)) if i != PRIMARY_CAMERA
        ]

        target_pos = None
        target_quat = None
        source_idx = None  # which camera provided the pose this frame

        for ci in ordered:
            frame, wl_world, wl_img, tracking = snapshots[ci]
            if not tracking:
                continue
            wl = wl_world.landmark

            R = body_frame(wl)
            if R is None:
                continue  # degenerate pose, try next camera

            # Raw world landmarks
            hs_w = vec3(wl[RIGHT_SHOULDER])
            he_w = vec3(wl[RIGHT_ELBOW])
            hw_w = vec3(wl[RIGHT_WRIST])

            # Express elbow and wrist relative to shoulder, in body frame
            # (shoulder becomes the origin, so camera translation drops out too)
            hs = [0.0, 0.0, 0.0]
            he = to_body_frame(vec_sub(he_w, hs_w), R)
            hw = to_body_frame(vec_sub(hw_w, hs_w), R)

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

        # -- IK ----------------------------------------------------------------
        # -- IK ----------------------------------------------------------------
        if target_pos:
            # target_pos here is the wrist; compute elbow position separately
            elbow_pos = retarget(
                hs,
                he,  # human elbow instead of wrist
                robot_shoulder_world,
                ROBOT_ARM_LENGTH * 0.47,  # ~proportion of upper arm to total arm length
                human_scale=calibrator.scale,
            )

            # Elbow IK pass
            if elbow_pos:
                sim.setObjectPosition(elbowTarget, elbow_pos)
                res, *_ = simIK.handleGroup(
                    ikEnv, ikGroupElbowUndamped, {"syncWorlds": True}
                )
                if res != simIK.result_success:
                    simIK.handleGroup(ikEnv, ikGroupElbowDamped, {"syncWorlds": True})

            # Wrist IK pass — runs on the joint config left by the elbow pass
            sim.setObjectPosition(wristTarget, target_pos)
            if target_quat:
                sim.setObjectQuaternion(wristTarget, target_quat)
            res, *_ = simIK.handleGroup(
                ikEnv, ikGroupWristUndamped, {"syncWorlds": True}
            )
            if res != simIK.result_success:
                simIK.handleGroup(ikEnv, ikGroupWristDamped, {"syncWorlds": True})

        sim.step()

        wrist_pos = sim.getObjectPosition(rightWristLink, -1)
        experiment.update(wrist_pos, dt)

        if experiment.finished and not getattr(experiment, "_summary_printed", False):
            print(experiment.summary())
            experiment._summary_printed = True

        # -- annotate frames & tile -------------------------------------------
        display_frames = []
        tile_widths = []
        for ci, (frame, wl_world, wl_img, tracking) in enumerate(snapshots):
            is_hud_cam = ci == HUD_CAMERA
            tw = TILE_WIDTH if is_hud_cam else SECONDARY_TILE_WIDTH

            if frame is None:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)

            # Skeleton overlay
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

            # Camera label + tracking status
            is_active = ci == source_idx
            label_txt = f"Cam {ci}" + (" [ACTIVE]" if is_active else "")
            status_col = (0, 255, 0) if tracking else (0, 0, 255)

            if is_hud_cam:
                # Full annotations on the HUD camera
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
                    cal_txt = f"Reach cal: stretch arm to calibrate ({calibrator._samples}/{calibrator.MIN_SAMPLES})"
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
                draw_experiment_hud(frame, experiment, wrist_pos, dt)
            else:
                # Minimal label only on secondary cameras (text is scaled to fit)
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
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    for ct in cam_threads:
        ct.stop()
    for ct in cam_threads:
        ct.join(timeout=2.0)

    cv2.destroyAllWindows()
    sim.stopSimulation()
    print("Simulation stopped.")

    # -- save results ----------------------------------------------------------
    exp_results = experiment.results if "experiment" in dir() else []
    if exp_results:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"poseEstimationReachResults/reach_results_{ts}.csv"
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
            for r in exp_results:
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
        print(f"Results saved to {filename}")
        print(experiment.summary())
    else:
        print("No results to save.")
