import mediapipe as mp
import cv2
import numpy as np

def return_all_joint_positions(image: np.ndarray) -> dict
    """
        Wrapper for media pipe functionality to return all joint positions
        in a custom data format that makes things easy to work with :)
    """
    
    mp_holistic = mp.solutions.holistic
    
    with mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
        
        results = holistic.process(image)
        