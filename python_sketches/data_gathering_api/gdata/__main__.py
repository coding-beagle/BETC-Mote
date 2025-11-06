import click
import os
import cv2
import numpy as np
from math import floor
import mediapipe as mp
import csv
import matplotlib.pyplot as plt
import csv

from .utils.utils import *


@click.group()
def cli():
    """Example script."""
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
