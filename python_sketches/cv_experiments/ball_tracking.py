import cv2 as cv
from matplotlib import pyplot
import numpy as np
import math

cam = cv.VideoCapture(0 + cv.CAP_DSHOW)

# Get the default frame width and height
frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*"mp4v")
out = cv.VideoWriter("output.mp4", fourcc, 20.0, (frame_width, frame_height))


def find_yellow_ball(frame: np.ndarray) -> tuple[int, int]:
    """Given a video frame, find the 2d coordinates of the yellow ball

    Args:
        frame np.ndarray: a frame containing red,green and blue values

    Returns:
        tuple[int,int]: (x,y) of where the center of the ball is
    """

    x = 0
    y = 0

    return (x, y)


def highlight_yellow_ball(frame: np.ndarray) -> np.ndarray:
    """Given a video frame, find the 2d coordinates of the yellow ball

    Args:
        frame np.ndarray: a frame containing red,green and blue values

    Returns:
        tuple[int,int]: (x,y) of where the center of the ball is
    """

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define range for yellow in HSV
    # Yellow hue is around 25-35 in OpenCV's HSV (0-180 scale)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create mask for yellow color
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    # Morphological operations to clean up the mask
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Reduce noise
    mask = cv.medianBlur(mask, 7)

    # Detect circles
    circles = cv.HoughCircles(
        mask,
        cv.HOUGH_GRADIENT,
        dp=2,
        minDist=1000,
        param1=150,
        param2=40,
        minRadius=20,
        maxRadius=100,
    )

    circles_out = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        circles_out.append([x, y, r])

    return mask, frame, circles_out


last_circle = []


def calc_distance_between_circles(circle_1, circle_2):
    return math.sqrt(
        (circle_1[0] - circle_2[0]) ** 2 + (circle_1[1] - circle_2[1]) ** 2
    )


while True:
    ret, frame = cam.read()

    # Display the captured frame
    cv.imshow("Camera", frame)

    gray, frame, circles = highlight_yellow_ball(frame)
    cv.imshow("gray", gray)

    if len(circles) > 0:
        x, y, r = circles[0]
        cv.circle(frame, (x, y), r, (0, 255, 0), 2)  # Circle outline
        cv.circle(frame, (x, y), 2, (0, 0, 255), 3)  # Center point
        last_circle = [x, y, r]
    elif len(last_circle) > 0:
        cv.circle(
            frame, (last_circle[0], last_circle[1]), last_circle[2], (0, 0, 255), 2
        )  # Circle outline
        cv.circle(
            frame, (last_circle[0], last_circle[1]), 2, (0, 0, 255), 3
        )  # Center point

    cv.imshow("YellowBall", frame)

    # Press 'q' to exit the loop
    if cv.waitKey(1) == ord("q"):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv.destroyAllWindows()
