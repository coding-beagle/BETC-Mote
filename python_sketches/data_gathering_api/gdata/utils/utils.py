# from .types import ThreeDPoint, Vector3D

import numpy as np
import math

RADIAN_TO_DEGREES = 180 / (math.pi)


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
