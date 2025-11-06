import numpy as np


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


def calculate_body_reference_frame(landmarks, mp_holistic):
    """
    Calculate body reference frame using hips and shoulders
    Returns: origin, forward_vector, up_vector, right_vector
    """
    # Get key body landmarks
    left_hip = landmarks[mp_holistic.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_holistic.PoseLandmark.RIGHT_HIP]
    left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

    # Hip center as origin
    hip_center = np.array(
        [
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2,
            (left_hip.z + right_hip.z) / 2,
        ]
    )

    # Shoulder center
    shoulder_center = np.array(
        [
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2,
            (left_shoulder.z + right_shoulder.z) / 2,
        ]
    )

    # Up vector: from hip center to shoulder center
    up_vector = shoulder_center - hip_center
    up_vector = up_vector / np.linalg.norm(up_vector)

    # Right vector: from left hip to right hip
    right_hip_pos = np.array([right_hip.x, right_hip.y, right_hip.z])
    left_hip_pos = np.array([left_hip.x, left_hip.y, left_hip.z])
    right_vector = right_hip_pos - left_hip_pos
    right_vector = right_vector / np.linalg.norm(right_vector)

    # Forward vector: cross product of right and up
    forward_vector = np.cross(right_vector, up_vector)
    forward_vector = forward_vector / np.linalg.norm(forward_vector)

    # Re-orthogonalize right vector
    right_vector = np.cross(up_vector, forward_vector)
    right_vector = right_vector / np.linalg.norm(right_vector)

    return hip_center, forward_vector, up_vector, right_vector


def calculate_limb_orientation_relative_to_body(
    proximal, distal, forward_vector, up_vector, right_vector
):
    """
    Calculate limb orientation relative to body reference frame
    """
    # Limb direction vector
    limb_vector = np.array(
        [distal.x - proximal.x, distal.y - proximal.y, distal.z - proximal.z]
    )
    limb_vector = limb_vector / np.linalg.norm(limb_vector)

    # Project limb onto body planes
    # Sagittal plane (forward-up): flexion/extension
    forward_component = np.dot(limb_vector, forward_vector)
    up_component = np.dot(limb_vector, up_vector)
    sagittal_angle = np.degrees(
        np.arctan2(forward_component, -up_component)
    )  # Flexion angle

    # Frontal plane (right-up): abduction/adduction
    right_component = np.dot(limb_vector, right_vector)
    frontal_angle = np.degrees(
        np.arctan2(right_component, -up_component)
    )  # Abduction angle

    # Transverse plane (forward-right): rotation
    transverse_angle = np.degrees(np.arctan2(right_component, forward_component))

    return {
        "flexion_extension": sagittal_angle,
        "abduction_adduction": frontal_angle,
        "rotation": transverse_angle,
    }
