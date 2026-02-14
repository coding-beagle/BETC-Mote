from math import pi
import cv2
import mediapipe as mp
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ── constants ────────────────────────────────────────────────────────────────
DEG_TO_RAD = pi / 180

# MediaPipe landmark indices
RIGHT_SHOULDER = 12
RIGHT_WRIST = 16

# YuMi right arm length in metres (upper + lower arm)
ROBOT_ARM_LENGTH = 0.21492 + 0.24129


# ── helpers ──────────────────────────────────────────────────────────────────
def vec3(lm):
    """Extract [x, y, z] from a MediaPipe landmark."""
    return [lm.x, lm.y, lm.z]


def vec_sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def vec_add(a, b):
    return [a[i] + b[i] for i in range(3)]


def vec_scale(v, s):
    return [v[i] * s for i in range(3)]


def vec_length(v):
    return sum(x**2 for x in v) ** 0.5


def retarget(human_shoulder, human_wrist, robot_shoulder, robot_arm_length):
    human_vec = vec_sub(human_wrist, human_shoulder)
    human_length = vec_length(human_vec)
    if human_length < 1e-6:
        return None

    normalised = vec_scale(human_vec, 1.0 / human_length)

    # Remap MediaPipe axes → CoppeliaSim axes
    # MediaPipe: x=right, y=down, z=toward camera
    # CoppeliaSim: x=forward, y=left, z=up
    remapped = [
        normalised[2],  # CoppeliaSim x ← MediaPipe -z (depth becomes forward)
        normalised[0],  # CoppeliaSim y ← MediaPipe -x (right becomes left)
        -normalised[1],  # CoppeliaSim z ← MediaPipe  y (down, so negate if up is wrong)
    ]

    robot_vec = vec_scale(remapped, robot_arm_length)
    return vec_add(robot_shoulder, robot_vec)


# ── CoppeliaSim setup ────────────────────────────────────────────────────────
print("Connecting to CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require("sim")
simIK = client.require("simIK")

# Object handles
rightShoulderAbduct = sim.getObject("/rightJoint1")
rightWristLink = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2"
    "/rightJoint3/rightLink3/rightJoint4/rightLink4"
    "/rightJoint5/rightLink5"
)
# Target dummy — a small sphere/dummy in the scene the IK solver drives toward.
# Create one programmatically so the scene doesn't need manual setup.
target = sim.createDummy(0.02)
sim.setObjectAlias(target, "WristTarget")

# IK environment
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

robot_shoulder_world = sim.getObjectPosition(rightShoulderAbduct, -1)
print(f"Robot shoulder origin: {robot_shoulder_world}")

try:
    target = sim.createDummy(0.02)
    sim.setObjectAlias(target, "WristTarget")
    print("Dummy created OK")

    ikEnv = simIK.createEnvironment()
    print("IK env created OK")

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
    print("Undamped IK group created OK")

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
    print("Damped IK group created OK")

except Exception as e:
    print(f"Setup failed: {e}")
    raise  # re-raise so you see the full traceback

print("Starting simulation...")


# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
print("mp_pose successful")
mp_draw = mp.solutions.drawing_utils
print("mp_draw successful")
pose = mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

sim.setStepping(True)
sim.startSimulation()
print("Simulation started OK")

print("Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")
print("Webcam opened OK")

try:
    while True:
        ret, frame = cap.read()
        print(f"Frame read: ret={ret}, frame shape={frame.shape if ret else 'N/A'}")
        break  # just test one frame for now
except Exception as e:
    print(f"Loop error: {e}")
    raise

# ── main loop ────────────────────────────────────────────────────────────────
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame, retrying...")
            continue

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        target_pos = None

        if results.pose_world_landmarks:
            wl = results.pose_world_landmarks.landmark

            human_shoulder = vec3(wl[RIGHT_SHOULDER])
            human_wrist = vec3(wl[RIGHT_WRIST])

            target_pos = retarget(
                human_shoulder,
                human_wrist,
                robot_shoulder_world,
                ROBOT_ARM_LENGTH,
            )

        if target_pos:
            sim.setObjectPosition(target, target_pos)

            res, *_ = simIK.handleGroup(ikEnv, ikGroupUndamped, {"syncWorlds": True})
            if res != simIK.result_success:
                simIK.handleGroup(ikEnv, ikGroupDamped, {"syncWorlds": True})

        sim.step()

        # ── visualisation ────────────────────────────────────────────────────
        # Draw skeleton on the BGR frame
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

        # Overlay IK status
        status_color = (0, 255, 0) if target_pos else (0, 0, 255)
        status_text = "IK: tracking" if target_pos else "IK: no pose"
        cv2.putText(
            frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
        )

        if target_pos:
            coord_text = f"Target: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})"
            cv2.putText(
                frame,
                coord_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.imshow("YuMi Pose Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit requested.")
            break

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    sim.stopSimulation()
    print("Simulation stopped.")
