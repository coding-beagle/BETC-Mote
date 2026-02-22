from math import pi, sqrt
import cv2
import mediapipe as mp
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ── constants ────────────────────────────────────────────────────────────────
DEG_TO_RAD = pi / 180

# MediaPipe landmark indices
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16

ROBOT_ARM_LENGTH = 0.21492 + 0.24129


# ── vector helpers ───────────────────────────────────────────────────────────
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
    """Remap MediaPipe axes to CoppeliaSim axes."""
    return [
        -v[2],  # CoppeliaSim x ← MediaPipe -z
        v[0],  # CoppeliaSim y ← MediaPipe  x
        -v[1],  # CoppeliaSim z ← MediaPipe -y
    ]


# ── pose helpers ─────────────────────────────────────────────────────────────
def retarget(human_shoulder, human_wrist, robot_shoulder, robot_arm_length):
    """Map human wrist position to robot world space."""
    human_vec = vec_sub(human_wrist, human_shoulder)
    human_length = vec_length(human_vec)
    if human_length < 1e-6:
        return None
    normalised = vec_scale(human_vec, 1.0 / human_length)
    remapped = remap_axes(normalised)
    return vec_add(robot_shoulder, vec_scale(remapped, robot_arm_length))


def compute_wrist_quaternion(human_shoulder, human_elbow, human_wrist):
    """
    Derive wrist pitch + yaw from the upper arm and forearm vectors.

    Strategy:
      - forward axis (Z): direction the forearm points (elbow → wrist)
      - up axis (Y):      component of upper arm perpendicular to forearm
                          (encodes the elbow hinge plane, giving pitch)
      - right axis (X):   cross product of the two, giving yaw

    Roll is zeroed out via Gram-Schmidt so only pitch + yaw are driven.
    Returns a CoppeliaSim quaternion [x, y, z, w] or None if degenerate.
    """
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

    # Gram-Schmidt: remove component of up parallel to fwd → zeroes out roll
    up_np = up_np - np.dot(up_np, fwd_np) * fwd_np
    up_len = np.linalg.norm(up_np)
    if up_len < 1e-6:
        return None
    up_np /= up_len

    right_np = np.cross(fwd_np, up_np)

    # Build rotation matrix: columns are right, up, forward
    # CoppeliaSim convention: X=right, Y=up, Z=forward
    R = np.column_stack([right_np, up_np, fwd_np])

    # Rotation matrix → quaternion (Shepperd method)
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

    return [x, y, z, w]  # CoppeliaSim quaternion order


# ── CoppeliaSim setup ────────────────────────────────────────────────────────
print("Connecting to CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require("sim")
simIK = client.require("simIK")

rightShoulderAbduct = sim.getObject("/rightJoint1")
rightWristLink = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2"
    "/rightJoint3/rightLink3/rightJoint4/rightLink4"
    "/rightJoint5/rightLink5/rightJoint6/rightLink6"
)

# Create a target dummy — IK drives rightWristLink to match this
target = sim.createDummy(0.02)
sim.setObjectAlias(target, "WristTarget")

robot_shoulder_world = sim.getObjectPosition(rightShoulderAbduct, -1)
print(f"Robot shoulder origin: {robot_shoulder_world}")

# IK environment
ikEnv = simIK.createEnvironment()

# Undamped group — position only (fast, accurate when reachable)
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

# Damped fallback — full pose constraint drives position + pitch + yaw
# Roll is zeroed out in compute_wrist_quaternion so only pitch/yaw are applied
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
    simIK.constraint_pose,
)

try:
    print("Starting simulation...")
    sim.setStepping(True)
    sim.startSimulation()
    print("Simulation started OK")

    # ── MediaPipe setup ──────────────────────────────────────────────────────
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

    # ── main loop ────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        target_pos = None
        target_quat = None

        if results.pose_world_landmarks:
            wl = results.pose_world_landmarks.landmark

            human_shoulder = vec3(wl[RIGHT_SHOULDER])
            human_elbow = vec3(wl[RIGHT_ELBOW])
            human_wrist = vec3(wl[RIGHT_WRIST])

            target_pos = retarget(
                human_shoulder,
                human_wrist,
                robot_shoulder_world,
                ROBOT_ARM_LENGTH,
            )
            target_quat = compute_wrist_quaternion(
                human_shoulder,
                human_elbow,
                human_wrist,
            )

        if target_pos:
            sim.setObjectPosition(target, target_pos)

        if target_quat:
            sim.setObjectQuaternion(target, target_quat)

        if target_pos or target_quat:
            # Try fast undamped solve first (position only)
            res, *_ = simIK.handleGroup(ikEnv, ikGroupUndamped, {"syncWorlds": True})
            if res != simIK.result_success:
                # Fall back to damped solve which also applies orientation
                simIK.handleGroup(ikEnv, ikGroupDamped, {"syncWorlds": True})

        sim.step()

        # ── visualisation ────────────────────────────────────────────────────
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

        status_color = (0, 255, 0) if target_pos else (0, 0, 255)
        cv2.putText(
            frame,
            "IK: tracking" if target_pos else "IK: no pose",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        if target_pos:
            cv2.putText(
                frame,
                f"Target: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.imshow("YuMi Pose Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    sim.stopSimulation()
    print("Simulation stopped.")
