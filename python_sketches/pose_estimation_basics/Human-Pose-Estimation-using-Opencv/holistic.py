import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find rotation matrix that aligns vec1 to vec2
    """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s == 0:
        return np.eye(3)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))
    return rotation_matrix


def rotation_matrix_to_euler(R):
    """
    Convert rotation matrix to Euler angles (XYZ convention)
    Returns roll, pitch, yaw
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def calculate_limb_orientation(
    proximal, distal, body_reference_up, body_reference_forward
):
    """
    Calculate full 3D orientation of a limb segment
    """
    limb_vector = np.array(
        [distal.x - proximal.x, distal.y - proximal.y, distal.z - proximal.z]
    )

    # Default reference (global coordinate system)
    default_limb = np.array([0, 1, 0])  # Pointing down

    # Calculate rotation from default to actual limb orientation
    R = rotation_matrix_from_vectors(default_limb, limb_vector)

    # Convert to Euler angles
    roll, pitch, yaw = rotation_matrix_to_euler(R)

    return roll, pitch, yaw


def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_angle_3d(a, b, c):
    """
    Calculate 3D angle at point b given three landmarks a, b, c
    Returns angle in degrees
    """
    a = np.array([a.x, a.y, a.z])  # Include z
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    # Vectors
    ba = a - b
    bc = c - b

    # Angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(
        np.clip(cosine_angle, -1.0, 1.0)
    )  # Clip to avoid numerical errors

    return np.degrees(angle)


cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract positions and angles
        if results.pose_landmarks:
            landmarks = results.pose_world_landmarks.landmark

            landmarks_2d = results.pose_landmarks.landmark

            # Get specific joint positions
            left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST]
            left_hip = landmarks[mp_holistic.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE]

            left_elbow_2d = landmarks_2d[mp_holistic.PoseLandmark.LEFT_ELBOW]
            left_shoulder_2d = landmarks_2d[mp_holistic.PoseLandmark.LEFT_SHOULDER]

            # Calculate angles
            left_elbow_angle = calculate_angle_3d(left_shoulder, left_elbow, left_wrist)
            left_shoulder_angle = calculate_angle_3d(
                left_hip, left_shoulder, left_elbow
            )
            # left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

            # Print results
            print(f"Left Elbow Angle: {left_elbow_angle:.2f}°")
            # print(f"Left Shoulder Angle: {left_shoulder_angle:.2f}°")
            # print(f"Left Knee Angle: {left_knee_angle:.2f}°")

            # Display angles on image
            h, w, _ = image.shape
            if left_elbow_angle:
                # print(left_elbow_angle)
                cv2.putText(
                    image,
                    f"Elbow: {left_elbow_angle:.1f}",
                    (int(left_elbow_2d.x * w), int(left_elbow_2d.y * h)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            if left_shoulder_angle:
                # print(left_shoulder_angle)
                cv2.putText(
                    image,
                    f"Shoulder: {left_shoulder_angle:.1f}",
                    (int(left_shoulder_2d.x * w), int(left_shoulder_2d.y * h)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        pose_connections = [
            conn
            for conn in mp_holistic.POSE_CONNECTIONS
            if conn[0] > 10 and conn[1] > 10
        ]

        body_landmarks = None

        # Only draw body landmarks (indices 11 and above)
        if results.pose_landmarks:
            # Create a copy of pose_landmarks with only body landmarks
            from mediapipe.framework.formats import landmark_pb2

            body_landmarks = landmark_pb2.NormalizedLandmarkList()
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                if i >= 11:  # Skip face landmarks (0-10)
                    body_landmarks.landmark.add().CopyFrom(landmark)
                else:
                    # Add invisible dummy landmarks to maintain indexing
                    dummy = body_landmarks.landmark.add()
                    dummy.x = 0
                    dummy.y = 0
                    dummy.z = 0
                    dummy.visibility = 0

        mp_drawing.draw_landmarks(image, body_landmarks, pose_connections)
        cv2.imshow("Whole body capture", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
