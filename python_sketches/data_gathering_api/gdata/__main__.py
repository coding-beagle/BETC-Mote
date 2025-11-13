import click
import os
import cv2
import numpy as np
from math import floor
import mediapipe as mp
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons
import csv

from .utils.utils import *
from .mediapipe_wrapper.mediapipe_utils import *


@click.group()
def cli():
    """GData - Bulk Video Editor + Pose Analysis CLI tool"""
    pass


OUT_FORMAT = ".mp4"


def resize_and_write(
    video_path: str,
    outfile: str,
    width: int,
    height: int,
    fps: int,
    rotate: bool = False,
) -> None:
    cap = cv2.VideoCapture(video_path)

    if rotate:
        output_width, output_height = height, width  # swapped after 90 deg rotation
    else:
        output_width, output_height = width, height

    output = cv2.VideoWriter(
        outfile, cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height)
    )

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.resize(image, (width, height))
        if rotate:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        output.write(image)

    cap.release()
    output.release()


def trim_video_file_and_write(video_path, out_path, start, end):
    video = cv2.VideoCapture(video_path)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = floor(float(start) * fps)
    end_frame = min(floor(float(end) * fps), frames)

    output_width, output_height = width, height

    output = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        int(fps),
        (output_width, output_height),
    )

    # Seek to start frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame = start_frame
    while video.isOpened() and frame <= end_frame:
        success, image = video.read()
        if not success:
            break

        output.write(image)
        frame += 1

    video.release()
    output.release()


def process_one_video(video_path, out_csv_path, draw=False):
    click.echo(f"Processing {video_path}, will write to {out_csv_path}")

    # Check if video file exists
    import os

    if not os.path.exists(video_path):
        click.echo(f"ERROR: Video file not found: {video_path}")
        return

    click.echo("Video file exists, attempting to open...")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            click.echo("ERROR: Failed to open video capture")
            return
        click.echo("Video capture opened successfully")
    except Exception as e:
        click.echo(f"ERROR opening video: {e}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_width, output_height = width, height

    fps = cap.get(cv2.CAP_PROP_FPS)
    output = None

    out_csv_name = out_csv_path
    frame = 0

    mp_drawing = None
    if draw:
        mp_drawing = mp.solutions.drawing_utils
        click.echo(
            f"Creating drawing output to {video_path.split('.MP4')[0] + '_drawn.mp4'}"
        )
        output = cv2.VideoWriter(
            video_path.split(".MP4")[0] + "_drawn.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            int(fps),
            (output_height, output_width),
        )

    mp_holistic = mp.solutions.holistic

    try:
        click.echo("About to create Holistic context...")
        with mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            click.echo("Successfully started holistic")
            with open(out_csv_name, "w") as f:
                click.echo("Successfully started csv writing")
                csv_writer = csv.writer(f, delimiter=";", lineterminator=";\n")
                csv_writer.writerow(["Frame", "Elbow Angle"])
                while cap.isOpened():
                    success, image = cap.read()

                    if not success:
                        click.echo("REACHED END OF VIDEO")
                        break

                    if draw:
                        image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_CLOCKWISE)

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
                            from mediapipe.framework.formats import landmark_pb2

                            body_landmarks = landmark_pb2.NormalizedLandmarkList()
                            for i, landmark in enumerate(
                                results.pose_landmarks.landmark
                            ):
                                if i >= 11:  # Skip face landmarks (0-10)
                                    body_landmarks.landmark.add().CopyFrom(landmark)
                                else:
                                    # Add invisible dummy landmarks to maintain indexing
                                    dummy = body_landmarks.landmark.add()
                                    dummy.x = 0
                                    dummy.y = 0
                                    dummy.z = 0
                                    dummy.visibility = 0

                        mp_drawing.draw_landmarks(
                            image, body_landmarks, pose_connections
                        )

                    # Extract positions and angles
                    if results.pose_landmarks:
                        landmarks = results.pose_world_landmarks.landmark

                        # # Calculate body reference frame
                        hip_center, forward_vec, up_vec, right_vec = (
                            calculate_body_reference_frame(landmarks, mp_holistic)
                        )

                        # # Get specific joint positions
                        # left_shoulder = landmarks[
                        #     mp_holistic.PoseLandmark.RIGHT_SHOULDER
                        # ]
                        left_elbow = landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW]
                        left_wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]
                        # left_hip = landmarks[mp_holistic.PoseLandmark.RIGHT_HIP]

                        # # Calculate simple 3D angles
                        # left_elbow_angle = calculate_angle_3d(
                        #     left_shoulder, left_elbow, left_wrist
                        # )

                        # # Calculate upper arm orientation relative to body
                        # upper_arm_orientation = (
                        #     calculate_limb_orientation_relative_to_body(
                        #         left_shoulder,
                        #         left_elbow,
                        #         forward_vec,
                        #         up_vec,
                        #         right_vec,
                        #     )
                        # )

                        # Calculate forearm orientation relative to body
                        forearm_orientation = (
                            calculate_limb_orientation_relative_to_body(
                                left_elbow, left_wrist, forward_vec, up_vec, right_vec
                            )
                        )

                        # Print results
                        # print(f"Left Elbow Flexion: {left_elbow_angle:.2f}°")
                        # print(
                        #     f"Upper Arm - Flex/Ext: {upper_arm_orientation['flexion_extension']:.1f}°, "
                        #     f"Abd/Add: {upper_arm_orientation['abduction_adduction']:.1f}°, "
                        #     f"Rotation: {upper_arm_orientation['rotation']:.1f}°"
                        # )
                        if draw:
                            landmarks_2d = results.pose_landmarks.landmark

                            left_elbow_2d = landmarks_2d[
                                mp_holistic.PoseLandmark.RIGHT_ELBOW
                            ]
                            left_shoulder_2d = landmarks_2d[
                                mp_holistic.PoseLandmark.RIGHT_SHOULDER
                            ]
                            # Display on image
                            h, w, _ = image.shape
                            cv2.putText(
                                image,
                                f"Flex: {forearm_orientation['flexion_extension']:.1f}",
                                (
                                    int(left_elbow_2d.x * w),
                                    int(left_elbow_2d.y * h) - 20,
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                (255, 255, 0),
                                2,
                            )
                            output.write(image)

                        csv_writer.writerow(
                            [frame, forearm_orientation["flexion_extension"]]
                        )

                        # cv2.putText(
                        #     image,
                        #     f"Flex: {forearm_orientation['abduction_adduction']:.1f}",
                        #     (int(left_elbow_2d.x * w), int(left_elbow_2d.y * h)),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.4,
                        #     (255, 255, 0),
                        #     1,
                        # )

                        # cv2.putText(
                        #     image,
                        #     f"Flex: {upper_arm_orientation['flexion_extension']:.1f}",
                        #     (
                        #         int(left_shoulder_2d.x * w),
                        #         int(left_shoulder_2d.y * h) - 20,
                        #     ),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.4,
                        #     (255, 255, 0),
                        #     1,
                        # )

                        # cv2.putText(
                        #     image,
                        #     f"Abd: {upper_arm_orientation['abduction_adduction']:.1f}",
                        #     (int(left_shoulder_2d.x * w), int(left_shoulder_2d.y * h)),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.4,
                        #     (255, 255, 0),
                        #     1,
                        # )
                    else:
                        click.echo(f"No pose analysed in frame {frame}")
                    frame += 1
    except Exception as e:
        click.echo(f"ERROR creating Holistic: {e}")
        import traceback

        click.echo(traceback.format_exc())

    cap.release()
    output.release()


@cli.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument("outdirectory", type=click.Path(dir_okay=True, file_okay=False))
@click.option("--width", default=1920)
@click.option("--height", default=1080)
@click.option("--fps", default=60)
@click.option("--rotate", default=False)
def batch_resize(directory, outdirectory, width, height, fps, rotate):
    try:
        os.makedirs(outdirectory)
    except OSError:
        click.echo("Output directory exists, try another path!")
        exit(-1)

    click.echo(f"Resizing everything in {directory} to {width}, {height}")
    videos = []
    for video in os.listdir(directory):
        videos.append({"path": directory + video, "file_name": video})

    LEN_VIDS = len(videos)

    click.echo(f"Found {LEN_VIDS} video(s), including: {videos[0]['file_name']})")

    for index, video_to_resize in enumerate(videos):
        click.echo(f"Resizing {index} / {LEN_VIDS} files...")
        outfile_location: str = (
            outdirectory
            + "/"
            + video_to_resize["file_name"].split(".")[0]
            + "_resized"
            + OUT_FORMAT
        )
        resize_and_write(
            video_to_resize["path"], outfile_location, width, height, fps, rotate
        )

    click.echo("Files resized and/or rotated!")


@cli.command()
@click.argument("video", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    "outfile", type=click.Path(exists=False, file_okay=True, dir_okay=False)
)
@click.argument("start")
@click.argument("end")
def trim_video(video, outfile, start, end):
    # start and end are in seconds
    click.echo(f"Trimming video: {video} into {outfile}...")
    trim_video_file_and_write(video, outfile, start, end)
    click.echo(f"Done!")


@cli.command()
@click.argument(
    "video_path", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option("--draw", default=False, is_flag=True)
def process_one(video_path, draw):
    file_name = video_path.split("/")[-1]
    file_name_no_extension = file_name.split(".")[0]
    folder_path = video_path[: -len(file_name)]
    csv_path = f"{folder_path}{file_name_no_extension}.csv"
    click.echo(f"Processing {file_name_no_extension}")

    process_one_video(video_path, csv_path, draw)


@cli.command()
@click.argument(
    "file_path", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option(
    "--save", "-s", is_flag=True, help="Save plot to file instead of displaying"
)
@click.option("--start", type=int, default=0, help="Start frame (inclusive)")
@click.option(
    "--end", type=int, default=None, help="End frame (inclusive, None for last frame)"
)
def plot_csv(file_path, save, start, end):
    import csv
    import os

    frames = []
    elbow_angles = []

    click.echo(f"Reading file: {file_path}")

    with open(file_path, "r") as file:
        csv_reader = csv.reader(file, delimiter=";", lineterminator=";\n")
        header = next(csv_reader, None)
        click.echo(f"Header: {header}")

        for row in csv_reader:
            if row and len(row) >= 2:
                try:
                    frame = float(row[0])
                    angle = float(row[1])

                    # Apply frame filtering
                    if frame < start:
                        continue
                    if end is not None and frame > end:
                        continue

                    frames.append(frame)
                    elbow_angles.append(angle)
                except (ValueError, IndexError) as e:
                    click.echo(f"Skipping row {row}: {e}")
                    continue

    click.echo(f"Loaded {len(frames)} data points (frames {start} to {end or 'end'})")

    if len(frames) == 0:
        click.echo("ERROR: No data loaded!")
        return

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(frames, elbow_angles, marker="o", linestyle="-", linewidth=2, markersize=4)
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Elbow Angle", fontsize=12)

    # Update title to show trim range
    if start > 0 or end is not None:
        title = f"Elbow Angle vs Frame (frames {start}-{end or 'end'})"
    else:
        title = "Elbow Angle vs Frame"
    plt.title(title, fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        # Get absolute path and change extension
        output_path = os.path.splitext(os.path.abspath(file_path))[0] + "_plot.png"
        click.echo(f"Attempting to save to: {output_path}")
        try:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            click.echo(f"✓ Plot saved successfully to: {output_path}")
        except Exception as e:
            click.echo(f"ERROR saving plot: {e}")
    else:
        click.echo("Displaying plot...")
        plt.show(block=True)

    plt.close()


# def forward_kinematics_arm(
#     shoulder_pos,
#     upper_arm_length,
#     forearm_length,
#     shoulder_flexion,
#     shoulder_abduction,
#     shoulder_rotation,
#     elbow_flexion,
# ):
#     """
#     Calculate elbow and wrist positions using forward kinematics

#     Args:
#         shoulder_pos: 3D position of shoulder (x, y, z)
#         upper_arm_length: length of upper arm segment
#         forearm_length: length of forearm segment
#         shoulder_flexion: shoulder flexion angle (0-180°, 0=down, 90=forward, 180=up)
#         shoulder_abduction: shoulder abduction angle (0-180°, 0=down, 90=lateral)
#         shoulder_rotation: shoulder rotation angle (0-180°)
#         elbow_flexion: elbow flexion angle (0-180°, 0=straight, 180=fully bent)

#     Returns:
#         elbow_pos, wrist_pos: 3D coordinates
#     """
#     # Convert angles to radians and adjust coordinate system
#     sf = np.radians(shoulder_flexion - 90)
#     sa = np.radians(shoulder_abduction)
#     sr = np.radians(shoulder_rotation)
#     ef = np.radians(elbow_flexion - 180)

#     # Calculate elbow position relative to shoulder
#     elbow_dir = np.array([0, -1, 0])

#     # Apply flexion rotation (around x-axis)
#     flex_rot = np.array(
#         [[1, 0, 0], [0, np.cos(sf), -np.sin(sf)], [0, np.sin(sf), np.cos(sf)]]
#     )

#     # Apply abduction rotation (around z-axis)
#     abd_rot = np.array(
#         [[np.cos(sa), -np.sin(sa), 0], [np.sin(sa), np.cos(sa), 0], [0, 0, 1]]
#     )

#     # Combine rotations for upper arm
#     upper_arm_dir = abd_rot @ flex_rot @ elbow_dir
#     elbow_pos = shoulder_pos + upper_arm_length * upper_arm_dir

#     # Calculate forearm direction with elbow flexion
#     forearm_dir = flex_rot @ np.array([0, np.cos(ef), np.sin(ef)])
#     forearm_dir = abd_rot @ forearm_dir

#     wrist_pos = elbow_pos + forearm_length * forearm_dir

#     return elbow_pos, wrist_pos


# @cli.command()
# @click.option(
#     "--upper-arm-length", "-a", default=0.3, help="Upper arm length in meters"
# )
# @click.option("--forearm-length", "-b", default=0.3, help="Forearm length in meters")
# def pose_estimation_3d_with_plotting(upper_arm_length, forearm_length):
#     """
#     Process video with pose estimation and show dual 3D plots:
#     1. Actual joint positions from MediaPipe
#     2. Forward kinematics reconstruction from angles
#     """

#     # Initialize MediaPipe
#     mp_holistic = mp.solutions.holistic
#     mp_drawing = mp.solutions.drawing_utils

#     # Open video
#     cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
#     if not cap.isOpened():
#         click.echo("ERROR: Cannot open video")
#         return

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Setup matplotlib with 3 subplots
#     plt.ion()
#     fig = plt.figure(figsize=(18, 6))

#     # Left: video feed
#     ax_video = fig.add_subplot(131)
#     ax_video.axis("off")
#     ax_video.set_title("Video Feed", fontsize=14)

#     # Middle: actual joint positions
#     ax_actual = fig.add_subplot(132, projection="3d")
#     ax_actual.set_xlabel("X (m)")
#     ax_actual.set_ylabel("Y (m)")
#     ax_actual.set_zlabel("Z (m)")
#     ax_actual.set_title("Actual Joint Positions (MediaPipe)", fontsize=14)

#     # Right: forward kinematics
#     ax_fk = fig.add_subplot(133, projection="3d")
#     ax_fk.set_xlabel("X (m)")
#     ax_fk.set_ylabel("Y (m)")
#     ax_fk.set_zlabel("Z (m)")
#     ax_fk.set_title("Forward Kinematics Reconstruction", fontsize=14)

#     # Set consistent limits for both 3D plots
#     for ax in [ax_actual, ax_fk]:
#         ax.set_xlim([-0.6, 0.6])
#         ax.set_ylim([-0.6, 0.6])
#         ax.set_zlim([-0.8, 0.4])

#     # Initialize plot elements for actual joints
#     (actual_arm_line,) = ax_actual.plot(
#         [], [], [], "b-", linewidth=3, marker="o", markersize=8, label="Arm"
#     )
#     (actual_wrist_trail,) = ax_actual.plot(
#         [], [], [], "r-", linewidth=1, alpha=0.5, label="Wrist Trail"
#     )

#     # Initialize plot elements for FK
#     (fk_arm_line,) = ax_fk.plot(
#         [], [], [], "g-", linewidth=3, marker="o", markersize=8, label="FK Arm"
#     )
#     (fk_wrist_trail,) = ax_fk.plot(
#         [], [], [], "m-", linewidth=1, alpha=0.5, label="FK Wrist Trail"
#     )

#     # Add legends
#     ax_actual.legend(loc="upper right")
#     ax_fk.legend(loc="upper right")

#     # Trail storage
#     actual_wrist_positions = []
#     fk_wrist_positions = []
#     max_trail_length = 50

#     video_img = None

#     # Create checkbox for toggling FK plot
#     checkbox_ax = plt.axes([0.01, 0.5, 0.15, 0.15])
#     checkbox = CheckButtons(checkbox_ax, ["Show FK Plot", "Show Trails"], [True, True])

#     show_fk = [True]  # Use list to allow modification in nested function
#     show_trails = [True]

#     def toggle_visibility(label):
#         if label == "Show FK Plot":
#             show_fk[0] = not show_fk[0]
#             ax_fk.set_visible(show_fk[0])
#         elif label == "Show Trails":
#             show_trails[0] = not show_trails[0]
#             actual_wrist_trail.set_visible(show_trails[0])
#             fk_wrist_trail.set_visible(show_trails[0])
#         fig.canvas.draw_idle()

#     checkbox.on_clicked(toggle_visibility)

#     with mp_holistic.Holistic(
#         min_detection_confidence=0.5, min_tracking_confidence=0.5
#     ) as holistic:

#         frame_count = 0

#         while cap.isOpened():
#             success, image = cap.read()
#             if not success:
#                 click.echo("End of video")
#                 break

#             # Process image
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = holistic.process(image_rgb)

#             # Draw pose landmarks on video
#             if results.pose_landmarks:
#                 pose_connections = [
#                     conn
#                     for conn in mp_holistic.POSE_CONNECTIONS
#                     if conn[0] > 10 and conn[1] > 10
#                 ]
#                 mp_drawing.draw_landmarks(
#                     image_rgb, results.pose_landmarks, pose_connections
#                 )

#                 # Get 3D world landmarks
#                 landmarks = results.pose_world_landmarks.landmark

#                 # Get arm landmarks (RIGHT arm)
#                 shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
#                 elbow = landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW]
#                 wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]

#                 shoulder_pos = np.array([shoulder.x, shoulder.y, shoulder.z])
#                 elbow_pos_actual = np.array([elbow.x, elbow.y, elbow.z])
#                 wrist_pos_actual = np.array([wrist.x, wrist.y, wrist.z])

#                 # Calculate body reference frame
#                 from .utils.utils import (
#                     calculate_body_reference_frame,
#                     calculate_limb_orientation_relative_to_body,
#                 )

#                 hip_center, forward_vec, up_vec, right_vec = (
#                     calculate_body_reference_frame(landmarks, mp_holistic)
#                 )

#                 # Get upper arm and forearm orientations
#                 upper_arm_orientation = calculate_limb_orientation_relative_to_body(
#                     shoulder, elbow, forward_vec, up_vec, right_vec
#                 )

#                 forearm_orientation = calculate_limb_orientation_relative_to_body(
#                     elbow, wrist, forward_vec, up_vec, right_vec
#                 )

#                 # Extract angles
#                 shoulder_flexion = upper_arm_orientation["flexion_extension"]
#                 shoulder_abduction = upper_arm_orientation["abduction_adduction"]
#                 shoulder_rotation = upper_arm_orientation["rotation"]
#                 elbow_flexion = forearm_orientation["flexion_extension"]

#                 # Forward kinematics
#                 elbow_fk, wrist_fk = forward_kinematics_arm(
#                     shoulder_pos,
#                     upper_arm_length,
#                     forearm_length,
#                     shoulder_flexion,
#                     shoulder_abduction,
#                     shoulder_rotation,
#                     elbow_flexion,
#                 )

#                 # Update actual joint positions plot
#                 actual_x = [shoulder_pos[0], elbow_pos_actual[0], wrist_pos_actual[0]]
#                 actual_y = [shoulder_pos[1], elbow_pos_actual[1], wrist_pos_actual[1]]
#                 actual_z = [shoulder_pos[2], elbow_pos_actual[2], wrist_pos_actual[2]]

#                 actual_arm_line.set_data(actual_x, actual_y)
#                 actual_arm_line.set_3d_properties(actual_z)

#                 # Update FK plot
#                 fk_x = [shoulder_pos[0], elbow_fk[0], wrist_fk[0]]
#                 fk_y = [shoulder_pos[1], elbow_fk[1], wrist_fk[1]]
#                 fk_z = [shoulder_pos[2], elbow_fk[2], wrist_fk[2]]

#                 fk_arm_line.set_data(fk_x, fk_y)
#                 fk_arm_line.set_3d_properties(fk_z)

#                 # Update trails if enabled
#                 if show_trails[0]:
#                     # Actual wrist trail
#                     actual_wrist_positions.append(wrist_pos_actual)
#                     if len(actual_wrist_positions) > max_trail_length:
#                         actual_wrist_positions.pop(0)

#                     if len(actual_wrist_positions) > 1:
#                         trail = np.array(actual_wrist_positions)
#                         actual_wrist_trail.set_data(trail[:, 0], trail[:, 1])
#                         actual_wrist_trail.set_3d_properties(trail[:, 2])

#                     # FK wrist trail
#                     fk_wrist_positions.append(wrist_fk)
#                     if len(fk_wrist_positions) > max_trail_length:
#                         fk_wrist_positions.pop(0)

#                     if len(fk_wrist_positions) > 1:
#                         trail = np.array(fk_wrist_positions)
#                         fk_wrist_trail.set_data(trail[:, 0], trail[:, 1])
#                         fk_wrist_trail.set_3d_properties(trail[:, 2])

#                 # Add angle text to video
#                 cv2.putText(
#                     image_rgb,
#                     f"Shoulder Flex: {shoulder_flexion:.1f}deg",
#                     (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (255, 255, 0),
#                     2,
#                 )
#                 cv2.putText(
#                     image_rgb,
#                     f"Shoulder Abd: {shoulder_abduction:.1f}deg",
#                     (10, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (255, 255, 0),
#                     2,
#                 )
#                 cv2.putText(
#                     image_rgb,
#                     f"Elbow Flex: {elbow_flexion:.1f}deg",
#                     (10, 90),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (255, 255, 0),
#                     2,
#                 )

#                 # Calculate and display error between actual and FK
#                 wrist_error = np.linalg.norm(wrist_pos_actual - wrist_fk)
#                 cv2.putText(
#                     image_rgb,
#                     f"FK Error: {wrist_error*100:.1f}cm",
#                     (10, 120),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (0, 255, 255),
#                     2,
#                 )

#             # Update video display
#             if video_img is None:
#                 video_img = ax_video.imshow(image_rgb)
#             else:
#                 video_img.set_data(image_rgb)

#             # Update display
#             plt.pause(0.001)

#             frame_count += 1
#             if frame_count % 30 == 0:
#                 click.echo(f"Processed frame {frame_count}")

#             # Check for window close
#             if not plt.fignum_exists(fig.number):
#                 break

#     cap.release()
#     plt.ioff()
#     plt.show()
#     click.echo("Processing complete!")


@cli.command()
@click.argument(
    "file_path", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option("-s", "--show", is_flag=True, default=False)
def process_img(file_path, show):
    click.echo(f"Searching for file {file_path}")

    img = cv2.imread(file_path)
    img_copy = img.copy()

    positions = return_all_relevant_joint_positions(img, show)

    if show:
        img_resized = cv2.resize(img_copy, [500, 700])
        cv2.imshow("Found image", img_resized)

        resized_landmarks = cv2.resize(positions.image, [500, 700])
        left_shoulder = convert_landmark_2d_to_pixel_coordinates(
            500, 700, positions.joint_pos2d["LEFT_SHOULDER"]
        )
        cv2.circle(resized_landmarks, left_shoulder, 10, [255, 0, 255], 2)
        cv2.imshow("Landmark Positions", resized_landmarks)

        cv2.waitKey(0)
