import json
import os

import cv2
import pandas as pd
import tqdm

from config import DATA_PATH, VIDEO_PATH
from utils import (
    calculate_frame_time,
    draw_on_frame,
    get_video_starttime,
    load_roi,
    point_is_in_polygon,
)

DATA_PATH = "./data3"
VIDEO_PATH  = "."
def process_raw_data(video_name, visualize: bool = False, save_video: bool = False):
    if save_video:
        # size = [1920, 1080]
        size = [1280, 720]
        video_writer = cv2.VideoWriter(
            "filename.mp4", cv2.VideoWriter_fourcc(*"XVID"), 25, size
        )

    base_video_name = video_name.split(".")[0]

    video_start_time = get_video_starttime(base_video_name)
    video_path = os.path.join(f"{VIDEO_PATH}/{video_name}")
    cap = cv2.VideoCapture(video_path)
    frame_counts = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    conf_limit = 0.5

    try:
        polygons = load_roi(base_video_name.split("_")[0])
    except FileNotFoundError as e:
        polygons = []

    Analytics = {"frame_number": [], "frame_time": []}

    Frame_People = {
        "frame_number": [],
        "box_x": [],
        "box_y": [],
        "box_w": [],
        "box_h": [],
        "conf": [],
        "label": [],
        "user_id": [],
        "region": [],
    }

    Frame_Regoins = {"frame_number": [], "region_name": [], "people_count": []}

    Person_Analytics = {}

    font = cv2.FONT_HERSHEY_SIMPLEX
    data = pd.read_csv(f"{DATA_PATH}/{base_video_name}.csv")

    for row in tqdm.tqdm(data.iterrows(), total=len(data)):
        if visualize or save_video:
            ret, frame = cap.read()

        try:
            _, frame_number, frame_time, results = row[1]
        except:
            # frame_number, results = row[1]
            frame_number,frame_time,  results = row[1]

        results = json.loads(results.replace("'", '"'))

        roi_counter = {}
        total_visitors = 0

        for polygon in polygons:
            roi_counter[polygon["name"]] = 0

        for result in results:
            id = result["id"]
            x, y, w, h = result["box"]
            x, y, w, h = int(x), int(y), int(w), int(h)
            conf = result["conf"]
            try:
                label_id = result["label_id"]
            except:
                label_id = -1

            Frame_People["frame_number"].append(frame_number)
            Frame_People["box_x"].append(x)
            Frame_People["box_y"].append(y)
            Frame_People["box_w"].append(w)
            Frame_People["box_h"].append(h)
            Frame_People["conf"].append(conf)
            Frame_People["label"].append(label_id)
            Frame_People["user_id"].append(id)

            if visualize or save_video:
                draw_on_frame(frame, [x, y, w, h], conf, id)

            if id not in Person_Analytics:
                Person_Analytics[id] = {
                    "locations": [],  # Store locations where the person was seen
                    "total_time_seen": 0,  # Initialize total time seen
                }

            if conf > conf_limit:
                total_visitors += 1

                location = {
                    "box": [x, y, w, h],
                    "conf": conf,
                    "frame_number": frame_number,
                }

                Person_Analytics[id]["locations"].append(location)

            person_region = ""
            for polygon in polygons:
                if point_is_in_polygon([x, y], polygon["polygon"]):
                    roi_counter[polygon["name"]] += 1
                    person_region = polygon["name"].split(".")[0]
            Frame_People["region"].append(person_region)

        frame_time_str, frame_time = calculate_frame_time(
            video_start_time, frame_number, fps
        )

        for k, v in roi_counter.items():
            k = k.split(".")[0]
            Frame_Regoins["frame_number"].append(frame_number)
            Frame_Regoins["region_name"].append(k)
            Frame_Regoins["people_count"].append(v)

        if visualize or save_video:
            y_loc = 160
            for k, v in roi_counter.items():
                k = k.split(".")[0]
                cv2.putText(
                    frame,
                    f"{k}: {v}",
                    (10, y_loc),
                    font,
                    1,
                    (250, 250, 250),
                    2,
                    cv2.LINE_AA,
                )
                y_loc += 50

            k = "People"
            v = total_visitors
            cv2.putText(
                frame,
                f"{k}: {v}",
                (10, y_loc),
                font,
                1,
                (250, 250, 250),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"{frame_time}", (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA
            )

            if save_video:
                video_writer.write(frame)

            if visualize:
                cv2.imshow("frame", frame)

                key_pressed = cv2.waitKey(1)
                if key_pressed == ord("q"):
                    break

        Analytics["frame_number"].append(frame_number)
        Analytics["frame_time"].append(frame_time_str)

    if save_video:
        video_writer.release()

    # Calculate total time seen for each person
    for person_id, person_data in Person_Analytics.items():
        locations = person_data["locations"]
        total_time = 0
        if locations:
            total_time = len(locations) / fps
        Person_Analytics[person_id]["total_time_seen"] = total_time

    Analytics = pd.DataFrame(Analytics)
    Analytics.to_csv(f"{DATA_PATH}/{base_video_name}_analytics.csv", index=False)

    Frame_People = pd.DataFrame(Frame_People)
    Frame_People.to_csv(f"{DATA_PATH}/{base_video_name}_frame_objects.csv", index=False)

    Frame_Regoins = pd.DataFrame(Frame_Regoins)
    Frame_Regoins.to_csv(
        f"{DATA_PATH}/{base_video_name}_frame_regions.csv", index=False
    )

    with open(f"{DATA_PATH}/{base_video_name}_person_analytics.json", "w") as fp:
        json.dump(Person_Analytics, fp)


if __name__ == "__main__":
    # video_name = "ch16_20240522034137.mp4"
    video_name = 'ch03_20240522000001.mp4'
    # video_name = 'ch03_20240522000055.mp4'
    process_raw_data(video_name, visualize=True, save_video=True)
