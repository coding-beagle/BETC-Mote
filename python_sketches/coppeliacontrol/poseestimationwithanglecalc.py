from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import cv2
import math
import mediapipe as mp
import numpy as np

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


def landmark_to_pos_vec(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z])


def vector_between_two_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return a vector object between two ThreeDPoints
    """

    return np.array(
        [
            b[0] - a[0],
            b[1] - a[1],
            b[2] - a[2],
        ]
    )


def magnitude(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))


def angle_between_vectors(a: np.ndarray, b: np.ndarray, degrees=False) -> float:
    dot = np.dot(a, b)

    mag_a = magnitude(a)
    mag_b = magnitude(b)

    cos_thetha = dot / (mag_a * mag_b)

    thetha = math.acos(cos_thetha)

    return thetha * RADIAN_TO_DEGREES if degrees else thetha


def angle_between_three_points(a, b, c, degrees=False) -> float:
    vec_a = vector_between_two_points(b, a)
    vec_b = vector_between_two_points(b, c)

    return angle_between_vectors(vec_a, vec_b, degrees)


def midpoint(a, b) -> np.ndarray:
    output = []
    for i in range(len(a)):
        output.append((a[i] + b[i]) / 2)

    return np.array(output)


def normal_vector_of_plane_on_three_points(a, b, c, unit_vec=True) -> np.ndarray:
    vec_a = vector_between_two_points(a, b)
    vec_b = vector_between_two_points(a, c)

    normal = np.cross(vec_a, vec_b)

    return normal * 1 / (magnitude(normal)) if unit_vec else normal


def transform_vector_in_relation_to_body_plane(
    a, body_normal, right_vector, up_vector
) -> np.ndarray:
    vec_forward = np.dot(a, body_normal)
    vec_right = np.dot(a, right_vector)
    vec_up = np.dot(a, up_vector)

    return np.array([vec_forward, vec_right, vec_up])


def invert_z(a) -> np.ndarray:
    return np.array([a[0], a[1], -a[2]])


def calc_wrist_deviation(shoulder_pos, elbow_pos, wrist_pos, hand_pos, degrees=False):
    # Define the flexion plane using shoulder, elbow, wrist
    flexion_plane_normal = normal_vector_of_plane_on_three_points(
        shoulder_pos, elbow_pos, wrist_pos
    )

    # Forearm axis
    forearm_vec = vector_between_two_points(elbow_pos, wrist_pos)

    # Hand vector
    hand_vec = vector_between_two_points(wrist_pos, hand_pos)

    # Project hand_vec onto the flexion plane
    # (subtract the component along the plane normal)
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
    # Upper arm axis (normalized)
    upper_arm_vec = vector_between_two_points(right_shoulder, elbow_pos)
    upper_arm_norm = upper_arm_vec / magnitude(upper_arm_vec)

    # Hand plane normal as proxy for arm roll
    hand_normal = normal_vector_of_plane_on_three_points(
        wrist_pos, index_mcp, pinky_mcp
    )

    # Project hand normal onto plane perpendicular to upper arm axis
    # (removes any component along the arm axis itself)
    hand_normal_projected = (
        hand_normal - np.dot(hand_normal, upper_arm_norm) * upper_arm_norm
    )
    hand_normal_projected = hand_normal_projected / magnitude(hand_normal_projected)

    # Reference vector: body right, also projected onto same plane
    body_right = vector_between_two_points(left_shoulder, right_shoulder)
    body_right_projected = (
        body_right - np.dot(body_right, upper_arm_norm) * upper_arm_norm
    )
    body_right_projected = body_right_projected / magnitude(body_right_projected)

    return angle_between_vectors(hand_normal_projected, body_right_projected, degrees)


def calc_joint_angles_from_data_dict(in_data):
    output = {}

    hip_left = in_data["HipL"]
    hip_right = in_data["HipR"]
    # hand = in_data["Hand"]
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


print("Connecting to CoppeliaSim...")
client = RemoteAPIClient()
print("Sucessfully connected")
sim = client.require("sim")
simIK = client.require("simIK")

rightShoulderAbduct = sim.getObject("/rightJoint1")
rightShoulderFlex = sim.getObject("/rightJoint1/rightLink1/rightJoint2")
rightForearmRoll = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3"
)
rightElbowFlex = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3/rightLink3/rightJoint4/"
)
rightWristDeviation = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3/rightLink3/rightJoint4/rightLink4/rightJoint5"
)
rightWristFlex = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3/rightLink3/rightJoint4/rightLink4/rightJoint5/rightLink5/rightJoint6"
)

print("Entering Try Block")

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
        target_pos = False

        if results.pose_world_landmarks:
            target_pos = True
            wl = results.pose_world_landmarks.landmark

            pose_data = {}

            # landmark_to_pos_vec(wl[RIGHT_SHOULDER])

            pose_data["ShoulderR"] = landmark_to_pos_vec(wl[RIGHT_SHOULDER])
            pose_data["ShoulderL"] = landmark_to_pos_vec(wl[LEFT_SHOULDER])
            pose_data["HipR"] = landmark_to_pos_vec(wl[RIGHT_HIP])
            pose_data["HipL"] = landmark_to_pos_vec(wl[LEFT_HIP])
            pose_data["Index"] = landmark_to_pos_vec(wl[RIGHT_BIG_FINGY])
            pose_data["Pinky"] = landmark_to_pos_vec(wl[RIGHT_SMALL_FINGY])
            pose_data["Elbow"] = landmark_to_pos_vec(wl[RIGHT_ELBOW])
            pose_data["Wrist"] = landmark_to_pos_vec(wl[RIGHT_WRIST])

            angles = calc_joint_angles_from_data_dict(pose_data)
            print(angles["roll"])

            sim.setJointTargetPosition(
                rightShoulderFlex,
                (angles["shoulder_flexion"] - 90) * DEGREES_TO_RADIANS,
            )
            sim.setJointTargetPosition(
                rightForearmRoll, (-angles["roll"] - 90) * DEGREES_TO_RADIANS
            )
            # sim.setJointTargetPosition(rightForearmRoll, (90) * DEGREES_TO_RADIANS)

            sim.setJointTargetPosition(
                rightShoulderAbduct,
                (130 - angles["shoulder_abduction"]) * DEGREES_TO_RADIANS,
            )
            sim.setJointTargetPosition(
                rightElbowFlex, (90 - angles["elbow_flexion"]) * DEGREES_TO_RADIANS
            )

            sim.setJointTargetPosition(
                rightWristDeviation, (angles["wrist_deviation"]) * DEGREES_TO_RADIANS
            )

            sim.setJointTargetPosition(
                rightWristFlex, (90 - angles["wrist_flexion"]) * DEGREES_TO_RADIANS
            )

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

        # if target_pos:
        #     cv2.putText(
        #         frame,
        #         f"Target: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})",
        #         (10, 60),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (255, 255, 255),
        #         1,
        #     )

        cv2.imshow("YuMi Pose Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted.")
except Exception as ex:
    print("Failed because of exception:")
    print(ex)
finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    sim.stopSimulation()
    print("Simulation stopped.")
