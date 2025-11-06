import click
import os
import cv2
import numpy as np
from math import floor


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
