from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import csv
import cv2
import datetime
import math
import mediapipe as mp
import numpy as np

# ── NEW: experiment module ────────────────────────────────────────────────────
from reach_experiment import Experiment

RADIAN_TO_DEGREES = 180 / (math.pi)
DEGREES_TO_RADIANS = (math.pi) / 180

RIGHT_HIP = 24
LEFT_HIP = 23
RIGHT_SMALL_FINGY = 18
RIGHT_BIG_FINGY = 20
RIGHT_SHOULDER = 12
LEFT_SHOULDER = 11
RIGHT_ELBOW = 14
RIGHT_WRIST = 16

# ── NEW: experiment configuration ────────────────────────────────────────────
ROBOT_ARM_LENGTH = 0.21492 + 0.24129

EXP_N_TRIALS = 10  # number of targets
EXP_RADIUS = 0.05  # success zone radius in metres
EXP_DWELL_TIME = 0.5  # seconds to hold inside zone
EXP_TIMEOUT = 20.0  # seconds per trial before fail
EXP_MIN_REACH = 0.7  # nearest target (fraction of arm length)
EXP_MAX_REACH = 0.9  # furthest target (fraction of arm length)
EXP_MIN_ELEVATION = -35.0  # degrees – allow slightly below horizontal
EXP_MAX_ELEVATION = 50.0  # degrees – cap well before overhead singularity
EXP_AZ_MIN = 20.0  # degrees – quarter-sphere spread around centre
EXP_AZ_MAX = 110.0  # degrees
EXP_SEED = None  # set an int for reproducible target placement


# ── geometry helpers ──────────────────────────────────────────────────────────
def landmark_to_pos_vec(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z])


def vector_between_two_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]])


def magnitude(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))


def angle_between_vectors(a: np.ndarray, b: np.ndarray, degrees=False) -> float:
    dot = np.dot(a, b)
    cos_theta = dot / (magnitude(a) * magnitude(b))
    theta = math.acos(np.clip(cos_theta, -1.0, 1.0))
    return theta * RADIAN_TO_DEGREES if degrees else theta


def angle_between_three_points(a, b, c, degrees=False) -> float:
    return angle_between_vectors(
        vector_between_two_points(b, a),
        vector_between_two_points(b, c),
        degrees,
    )


def midpoint(a, b) -> np.ndarray:
    return np.array([(a[i] + b[i]) / 2 for i in range(len(a))])


def normal_vector_of_plane_on_three_points(a, b, c, unit_vec=True) -> np.ndarray:
    vec_a = vector_between_two_points(a, b)
    vec_b = vector_between_two_points(a, c)
    normal = np.cross(vec_a, vec_b)
    return normal / magnitude(normal) if unit_vec else normal


def calc_wrist_deviation(shoulder_pos, elbow_pos, wrist_pos, hand_pos, degrees=False):
    flexion_plane_normal = normal_vector_of_plane_on_three_points(
        shoulder_pos, elbow_pos, wrist_pos
    )
    forearm_vec = vector_between_two_points(elbow_pos, wrist_pos)
    hand_vec = vector_between_two_points(wrist_pos, hand_pos)
    hand_in_plane = (
        hand_vec - np.dot(hand_vec, flexion_plane_normal) * flexion_plane_normal
    )
    return angle_between_vectors(forearm_vec, hand_in_plane, degrees)


def calc_wrist_flex(elbow_pos, wrist_pos, hand_pos):
    return angle_between_three_points(elbow_pos, wrist_pos, hand_pos)


def calc_elbow_flex(shoulder_pos, elbow_pos, wrist_pos):
    return angle_between_three_points(shoulder_pos, elbow_pos, wrist_pos)


def calc_shoulder(left_shoulder, right_shoulder, hip_left, hip_right, elbow):
    midpoint_hip = midpoint(hip_left, hip_right)
    body_plane_normal = normal_vector_of_plane_on_three_points(
        left_shoulder, right_shoulder, midpoint_hip
    )
    right_upper_arm_vector = vector_between_two_points(right_shoulder, elbow)
    right_vector = vector_between_two_points(left_shoulder, right_shoulder)
    down_vector = np.cross(body_plane_normal, right_vector)

    A_dot_n = np.dot(right_upper_arm_vector, body_plane_normal)
    A_dot_down = np.dot(right_upper_arm_vector, down_vector)
    A_dot_right = np.dot(right_upper_arm_vector, right_vector)

    shoulder_flexion = math.atan2(A_dot_n, math.sqrt(A_dot_right**2 + A_dot_down**2))
    shoulder_abduction = math.atan2(A_dot_right, A_dot_down)
    return [shoulder_flexion, shoulder_abduction]


def calc_upper_arm_roll(
    left_shoulder,
    right_shoulder,
    elbow_pos,
    wrist_pos,
    index_mcp,
    pinky_mcp,
    degrees=False,
):
    upper_arm_vec = vector_between_two_points(right_shoulder, elbow_pos)
    upper_arm_norm = upper_arm_vec / magnitude(upper_arm_vec)
    hand_normal = normal_vector_of_plane_on_three_points(
        wrist_pos, index_mcp, pinky_mcp
    )
    hand_normal_projected = (
        hand_normal - np.dot(hand_normal, upper_arm_norm) * upper_arm_norm
    )
    hand_normal_projected /= magnitude(hand_normal_projected)
    body_right = vector_between_two_points(left_shoulder, right_shoulder)
    body_right_projected = (
        body_right - np.dot(body_right, upper_arm_norm) * upper_arm_norm
    )
    body_right_projected /= magnitude(body_right_projected)
    return angle_between_vectors(hand_normal_projected, body_right_projected, degrees)


def calc_joint_angles_from_data_dict(in_data):
    output = {}
    hip_left = in_data["HipL"]
    hip_right = in_data["HipR"]
    elbow = in_data["Elbow"]
    wrist = in_data["Wrist"]
    index = in_data["Index"]
    pinky = in_data["Pinky"]
    shoulder = in_data["ShoulderR"]
    left_shoulder = in_data["ShoulderL"]

    output["elbow_flexion"] = (
        calc_elbow_flex(shoulder, elbow, wrist) * RADIAN_TO_DEGREES
    )
    output["wrist_flexion"] = calc_wrist_flex(elbow, wrist, index) * RADIAN_TO_DEGREES
    shoulder_flex, shoulder_abduct = calc_shoulder(
        left_shoulder, shoulder, hip_left, hip_right, elbow
    )
    output["roll"] = calc_upper_arm_roll(
        left_shoulder, shoulder, elbow, wrist, index, pinky, True
    )
    output["wrist_deviation"] = calc_wrist_deviation(
        shoulder, elbow, wrist, index, True
    )
    output["shoulder_flexion"] = shoulder_flex * RADIAN_TO_DEGREES
    output["shoulder_abduction"] = shoulder_abduct * RADIAN_TO_DEGREES
    return output


# ── NEW: OpenCV HUD helpers ───────────────────────────────────────────────────
def _cv_col(r, g, b):
    """Convert RGB tuple to BGR for OpenCV."""
    return (b, g, r)


def draw_experiment_hud(frame, experiment, wrist_pos, dt):
    """
    Draw the reach-experiment HUD directly onto an OpenCV BGR frame.
    Called once per frame in place of experiment.draw() (which needs pygame).
    """
    H, W = frame.shape[:2]
    active = experiment._active

    # ── progress bar along bottom ─────────────────────────────────────────────
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
        return

    if active is None:
        return

    # ── distance and colour ───────────────────────────────────────────────────
    dist = active.distance_to(wrist_pos)
    ratio = min(1.0, dist / (active.radius * 6))

    def lerp(a, b, t):
        t = max(0.0, min(1.0, t))
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

    C_HOT = (100, 220, 130)
    C_WARM = (220, 190, 60)
    C_IDLE = (100, 160, 220)
    if ratio < 0.4:
        hud_rgb = lerp(C_HOT, C_WARM, ratio / 0.4)
    else:
        hud_rgb = lerp(C_WARM, C_IDLE, (ratio - 0.4) / 0.6)
    hud_col = _cv_col(*hud_rgb)

    # ── flash overlay on result ───────────────────────────────────────────────
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

    # ── info panel (top-right) ────────────────────────────────────────────────
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

    # distance bar
    bar_x, bar_y2 = px, py + 52
    bar_w = 250
    fill = int(bar_w * (1.0 - ratio))
    cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + 8), (55, 55, 55), -1)
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + fill, bar_y2 + 8), hud_col, -1)

    # dwell arc (drawn as filled wedge approximation)
    if active._inside and active.dwell_fraction > 0:
        cx, cy = W - 35, py + 88
        r = 18
        cv2.circle(frame, (cx, cy), r, (55, 55, 55), 2)
        angle_end = int(360 * active.dwell_fraction)
        cv2.ellipse(
            frame, (cx, cy), (r, r), -90, 0, angle_end, _cv_col(100, 220, 130), 2
        )
        pct = f"{int(active.dwell_fraction*100)}%"
        cv2.putText(
            frame,
            pct,
            (cx - 14, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            _cv_col(100, 220, 130),
            1,
        )

    # timer / waiting message
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

print("Entering Try Block")

try:
    print("Starting simulation...")
    sim.setStepping(True)
    sim.startSimulation()
    print("Simulation started OK")

    # ── NEW: create experiment after sim starts ───────────────────────────────
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

    # ── MediaPipe setup ───────────────────────────────────────────────────────
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Running — press Q to quit.")

    wrist_pos = list(robot_shoulder_world)  # initialised; updated each step
    prev_time = cv2.getTickCount()

    # ── main loop ─────────────────────────────────────────────────────────────
    while True:
        # dt for experiment timing
        now = cv2.getTickCount()
        dt = (now - prev_time) / cv2.getTickFrequency()
        prev_time = now

        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        tracking = False
        roll_angle = 45

        if results.pose_world_landmarks:
            tracking = True
            wl = results.pose_world_landmarks.landmark

            pose_data = {
                "ShoulderR": landmark_to_pos_vec(wl[RIGHT_SHOULDER]),
                "ShoulderL": landmark_to_pos_vec(wl[LEFT_SHOULDER]),
                "HipR": landmark_to_pos_vec(wl[RIGHT_HIP]),
                "HipL": landmark_to_pos_vec(wl[LEFT_HIP]),
                "Index": landmark_to_pos_vec(wl[RIGHT_BIG_FINGY]),
                "Pinky": landmark_to_pos_vec(wl[RIGHT_SMALL_FINGY]),
                "Elbow": landmark_to_pos_vec(wl[RIGHT_ELBOW]),
                "Wrist": landmark_to_pos_vec(wl[RIGHT_WRIST]),
            }

            angles = calc_joint_angles_from_data_dict(pose_data)

            sim.setJointTargetPosition(
                rightShoulderFlex,
                (angles["shoulder_flexion"] - 90) * DEGREES_TO_RADIANS,
            )
            sim.setJointTargetPosition(
                rightForearmRoll, roll_angle * DEGREES_TO_RADIANS
            )
            sim.setJointTargetPosition(
                rightShoulderAbduct,
                (130 - angles["shoulder_abduction"]) * DEGREES_TO_RADIANS,
            )
            sim.setJointTargetPosition(
                rightElbowFlex,
                (90 - angles["elbow_flexion"]) * DEGREES_TO_RADIANS,
            )

        sim.step()

        # ── NEW: read wrist and update experiment ─────────────────────────────
        wrist_pos = sim.getObjectPosition(rightWristLink, -1)
        experiment.update(wrist_pos, dt)

        if experiment.finished and not getattr(experiment, "_summary_printed", False):
            print(experiment.summary())
            experiment._summary_printed = True

        # ── visualisation ─────────────────────────────────────────────────────
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

        status_color = (0, 255, 0) if tracking else (0, 0, 255)
        cv2.putText(
            frame,
            "tracking" if tracking else "no pose",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        # ── NEW: draw experiment HUD onto frame ───────────────────────────────
        draw_experiment_hud(frame, experiment, wrist_pos, dt)

        cv2.imshow("YuMi Pose Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted.")
except Exception as ex:
    print("Failed because of exception:")
    import traceback

    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    sim.stopSimulation()
    print("Simulation stopped.")

    # ── NEW: save results to CSV ──────────────────────────────────────────────
    results = experiment.results if "experiment" in dir() else []
    if results:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reach_results_{ts}.csv"
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
        print(f"Results saved to {filename}")
        print(experiment.summary())
    else:
        print("No results to save.")
