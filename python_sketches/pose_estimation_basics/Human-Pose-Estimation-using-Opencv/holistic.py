import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


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
            landmarks = results.pose_landmarks.landmark

            # Get specific joint positions
            left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST]
            left_hip = landmarks[mp_holistic.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE]

            # Calculate angles
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

            # Print results
            # print(f"Left Elbow Angle: {left_elbow_angle:.2f}°")
            # print(f"Left Shoulder Angle: {left_shoulder_angle:.2f}°")
            # print(f"Left Knee Angle: {left_knee_angle:.2f}°")

            # Display angles on image
            h, w, _ = image.shape
            if left_elbow_angle:
                print(left_elbow_angle)
                cv2.putText(
                    image,
                    f"Elbow: {left_elbow_angle:.1f}",
                    (int(left_elbow.x * w), int(left_elbow.y * h)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            if left_shoulder_angle:
                print(left_shoulder_angle)
                cv2.putText(
                    image,
                    f"Shoulder: {left_shoulder_angle:.1f}",
                    (int(left_shoulder.x * w), int(left_shoulder.y * h)),
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
