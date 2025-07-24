import datetime
import os
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
import base64

from config import VIDEO_PATH, WEEKDAYS

font = cv2.FONT_HERSHEY_SIMPLEX

# Global variables
points = []
drawing = False
frame_idx = 0
frame = None


def get_video_starttime(video_name):
    print("here is the vidd",  video_name)
    video_start_time = video_name.split("_")[1]
    return datetime.datetime.strptime(video_start_time, "%Y%m%d%H%M%S")


def calculate_frame_time(video_starttime, frame_number, fps):
    video_time = frame_number / fps
    frame_time = video_starttime + datetime.timedelta(seconds=video_time)
    wd = WEEKDAYS[frame_time.weekday()]
    frame_time_str = f"{frame_time.year}/{frame_time.month}/{frame_time.day} {wd} {frame_time.hour}:{frame_time.minute}:{frame_time.second}"
    return frame_time_str, frame_time


def draw_on_frame(frame, box, conf, id):
    x, y, w, h = box
    xi = x - (w // 2)
    yi = y - (h // 2)
    xf = x + (w // 2)
    yf = y + (h // 2)

    # temp_frame = frame.copy()
    cv2.rectangle(frame, (xi, yi), (xf, yf), color=(0, 0, 255), thickness=2, lineType=1)
    cv2.rectangle(
        frame,
        (xi - 10, yi - 50),
        (xf, yi - 10),
        color=(250, 250, 250),
        thickness=-1,
        lineType=1,
    )
    cv2.putText(
        frame, f"{id} ({conf})", (xi, yi - 20), font, 1, (0, 0, 255), 1, cv2.LINE_AA
    )
    # return temp_frame


def write_on_frame(
    frame, x, y, text: str, color=(0, 0, 255), thickness: int = 1, font_size: int = 1
):
    cv2.putText(
        frame, text, (x, y), font, thickness, (0, 0, 255), font_size, cv2.LINE_AA
    )


def point_is_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def load_roi(video_id):
    folder_path = f"roi/{video_id}"
    roi_list = []
    for fname in os.listdir(folder_path):
        polygon = {"name": fname.split(".")[0], "centroid": [0, 0], "polygon": []}
        with open(os.path.join(f"{folder_path}/{fname}")) as fp:
            for line in fp.readlines():
                polygon["polygon"].append([int(p) for p in line.split(",")])

            x_list = [p[0] for p in polygon["polygon"]]
            y_list = [p[1] for p in polygon["polygon"]]
            polygon["centroid"] = [
                int(np.mean([min(x_list), max(x_list)])),
                int(np.mean([min(y_list), max(y_list)])) - 50,
            ]

        roi_list.append(polygon)

    return roi_list


# Mouse callback function to capture points
def draw_polygon(event, x, y, flags, param):
    global points, drawing, frame
    # frame_copy = param.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        points.append((x, y))
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if len(points) > 0:
            last_point = None
            for point in points:
                if last_point is not None:
                    cv2.line(frame, last_point, point, (0, 255, 0), 2)

                last_point = point

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(points) > 1:
            last_point = None
            for point in points:
                if last_point is not None:
                    cv2.line(frame, last_point, point, (0, 255, 0), 2)

                last_point = point

    cv2.imshow("Video", frame)


def insert_previous_hour_min(end_time, freq=15, exclude_last=True):
    end_time = pd.to_datetime(end_time)  # Get the existing time
    if exclude_last:
        end_time -= datetime.timedelta(minutes=freq)
    start_time = pd.to_datetime("00:00")
    time_index = pd.date_range(
        start_time, end_time, freq=f"{freq}min"
    ).time
    return time_index


def image_to_base64(frame):
    try:
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame_base64 = base64.b64encode(buffer).decode("utf-8")
        frame_data = f"data:image/jpeg;base64,{frame_base64}"
        return frame_data
    except TypeError as e:
        print(e)
        return None


# Main function to load the video and set up the mouse callback
def main(video_path):
    global points, frame_idx, frame

    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to load video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow("Video")

    print(
        "Click on the video to create a polygon. Press 'c' to clear, 'q' to quit, 's' to save the frame with the polygon, 'n' for next frame, and 'p' for previous frame."
    )

    target_height = 900
    target_ratio = 1

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if frame.shape[1] > target_height:
            target_ratio = target_height / frame.shape[1]

        frame = cv2.resize(frame, dsize=None, fx=target_ratio, fy=target_ratio)

        cv2.setMouseCallback("Video", draw_polygon)

        if not ret:
            print("Reached end of video or failed to read frame.")
            break

        cv2.imshow("Video", frame)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):  # Quit the loop
            break
        elif key == ord("c"):  # Clear the points
            points = []
            cv2.imshow("Video", frame)
        elif key == ord("s"):  # Save the frame with the polygon
            if len(points) > 1:
                cv2.polylines(
                    frame,
                    [np.array(points, dtype=np.int32)],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )
                output_filename = f"output_frame_{frame_idx}.jpg"
                cv2.imwrite(output_filename, frame)
                with open(f"roi/{uuid4()}.csv", "w") as fp:
                    for p in points:
                        fp.writelines(
                            f"{int(p[0]/target_ratio)},{int(p[1]/target_ratio)}\n"
                        )

                print(f"Frame saved as '{output_filename}'.")
        elif key == ord("n"):  # Next frame
            if frame_idx < total_frames - 1:
                frame_idx += 1
                points = []  # Clear points for new frame
        elif key == ord("p"):  # Previous frame
            if frame_idx > 0:
                frame_idx -= 1
                points = []  # Clear points for new frame

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = f"{VIDEO_PATH}/ch03_20240522041053.mp4"
    main(video_path)
