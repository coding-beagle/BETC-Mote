import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np

from dataclasses import dataclass


@dataclass
class JointPositions:
    image: np.ndarray
    joint_pos: dict
    successful: bool
    joint_pos2d: dict


def convert_landmark_dict_to_vec(landmark) -> np.ndarray:
    return np.array([landmark["x"], landmark["y"], landmark["z"]])


def landmark_to_dict(landmark):
    # print(landmark)
    land_dict = {
        "x": landmark.x,
        "y": landmark.y,
        "z": landmark.z,
        "visibility": landmark.visibility,  # if it exists
    }
    return convert_landmark_dict_to_vec(land_dict)


def return_all_relevant_joint_positions(
    image: np.ndarray, draw: bool = False
) -> JointPositions:
    """
    Wrapper for media pipe functionality to return all joint positions
    in a custom data format that makes things easy to work with :)
    """

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:

        results = holistic.process(image)
        output = JointPositions(
            image=None, joint_pos=None, joint_pos2d=None, successful=False
        )

        if results.pose_world_landmarks.landmark:
            output.successful = True
            landmarks = results.pose_world_landmarks.landmark
            landmarks_2d = results.pose_landmarks.landmark
            output.joint_pos = {
                "RIGHT_SHOULDER": landmark_to_dict(
                    landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                ),
                "RIGHT_WRIST": landmark_to_dict(
                    landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]
                ),
                "RIGHT_ELBOW": landmark_to_dict(
                    landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW]
                ),
                "LEFT_SHOULDER": landmark_to_dict(
                    landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                ),
                "LEFT_WRIST": landmark_to_dict(
                    landmarks[mp_holistic.PoseLandmark.LEFT_WRIST]
                ),
                "LEFT_ELBOW": landmark_to_dict(
                    landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW]
                ),
                "LEFT_HIP": landmark_to_dict(
                    landmarks[mp_holistic.PoseLandmark.LEFT_HIP]
                ),
                "RIGHT_HIP": landmark_to_dict(
                    landmarks[mp_holistic.PoseLandmark.RIGHT_HIP]
                ),
            }
            output.joint_pos2d = {
                "RIGHT_SHOULDER": landmarks_2d[mp_holistic.PoseLandmark.RIGHT_SHOULDER],
                "RIGHT_WRIST": landmarks_2d[mp_holistic.PoseLandmark.RIGHT_WRIST],
                "RIGHT_ELBOW": landmarks_2d[mp_holistic.PoseLandmark.RIGHT_ELBOW],
                "LEFT_SHOULDER": landmarks_2d[mp_holistic.PoseLandmark.LEFT_SHOULDER],
                "LEFT_WRIST": landmarks_2d[mp_holistic.PoseLandmark.LEFT_WRIST],
                "LEFT_ELBOW": landmarks_2d[mp_holistic.PoseLandmark.LEFT_ELBOW],
                "LEFT_HIP": landmarks_2d[mp_holistic.PoseLandmark.LEFT_HIP],
                "RIGHT_HIP": landmarks_2d[mp_holistic.PoseLandmark.RIGHT_HIP],
            }
            if draw:
                pose_connections = [
                    conn
                    for conn in mp_holistic.POSE_CONNECTIONS
                    if conn[0] > 10 and conn[1] > 10
                ]

                body_landmarks = None

                # Only draw body landmarks (indices 11 and above)
                if results.pose_landmarks:
                    # Create a copy of pose_landmarks with only body landmarks

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
                output.image = image

        return output


def convert_landmark_2d_to_pixel_coordinates(image_height, image_width, in_landmark):
    # print(dict(in_dict))
    return [int(in_landmark.x * image_width), int(in_landmark.y * image_height)]
