import asyncio
import datetime
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

from config import DATA_PATH, VIDEO_PATH
from generate_heatmap import generate_heatmap_points
from analytics import IntervalLabelCounter, UniqueIDLabelCounter, UniqueIDIntervalLabelCounter
from object_recognition import ObjectRecognition
from objects_data import ObjectsHistoricData
from utils import (
    calculate_frame_time,
    draw_on_frame,
    get_video_starttime,
    load_roi,
)

if not os.path.exists("temp"):
    os.mkdir("temp")

with open(file="models/labels.json") as fp:
    model_labels = json.load(fp=fp)

cashier_recognition_model = ObjectRecognition("cashier_clf")

time_to_retain_sub_label = 3

heatmap_scale = 4

def channel_analytics(
    channel_name: str,
    start_frame=0,
    skip_frame=1,
    show_frame: bool = False,
    cashier_recognition: bool = False,
    display_region: bool = True,
):
    channel_files = [
        f for f in sorted(os.listdir(VIDEO_PATH)) if f.startswith(channel_name)
    ]

    print(channel_name)
    print(channel_files)

    print(start_frame, skip_frame, show_frame, cashier_recognition, display_region)

    target_fps = 25
    frame_delay = 1 / target_fps

    try:
        polygons = load_roi(channel_name.lower())
    except FileNotFoundError as e:
        print(e)
        polygons = []

    recog_models = []
    if cashier_recognition:
        recog_models.append(cashier_recognition_model)

    historic_data = ObjectsHistoricData(regions=polygons)

    cashier_time_missing = 0
    cashier_time_available = 0

    cashier_hourly_counter = IntervalLabelCounter(
        label="cashier",
        interval=60
    )

    regions_unique_id_counter = [
        UniqueIDLabelCounter(label=f"Region:{pgn['name']}", min_count=50)
        for pgn in polygons
    ]

    labels_unique_id_counter = [
        UniqueIDLabelCounter(label=_name, min_count=25)
        for _k, _name in cashier_recognition_model.names.items()
    ]

    up_direction_unique_counter = UniqueIDLabelCounter(label="Direction:Up")
    down_direction_unique_counter = UniqueIDLabelCounter(label="Direction:Down")

    down_unique_hourly_counter = UniqueIDIntervalLabelCounter(
        label="Direction:Down",
        interval=60
        )

    down_unique_15m_counter = UniqueIDIntervalLabelCounter(
        label="Direction:Down",
        interval=15
        )

    calculate_rate = 25 * 10
    last_calculation_frame = calculate_rate

    global_frame_number = 0

    # Initialize Heatmap data as list for faster appends
    heatmap_data = {"x": [], "y": [], "label": []}
    heatmap_filepath = f"temp/{channel_name}_heatmap.csv"

    heatmap_data = pd.DataFrame(heatmap_data)
    heatmap_data.to_csv(heatmap_filepath, index=False)

    SubLabelsIds = {}
    SubLabelsTimeout = 30

    frame_send_time = time.time()
    frame_send_delay = 0

    frame_save_time = 30*10
    last_frame_save_time = frame_save_time

    for video_name in channel_files:
        base_video_name = video_name.split(".")[0]
        video_start_time = get_video_starttime(base_video_name)
        video_path = os.path.join(f"{VIDEO_PATH}/{video_name}")

        # load video
        cap = cv2.VideoCapture(video_path)
        frame_counts = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        data = pd.read_csv(f"{DATA_PATH}/{base_video_name}.csv")

        current_video_frame_number = 0

        while True:
            timer_frame_start = time.time()
            time_stats = {}

            last_calculation_frame += 1

            ret, frame = cap.read()
            # print(time.time() - timer_frame_start)

            if not ret:
                print(video_name, "Error Ret")
                break

            frame_heatmap_data = {"x": [], "y": [], "label": []}

            if last_frame_save_time > frame_save_time:
                cv2.imwrite(
                    f"temp/{channel_name}_frame.jpg",
                    cv2.resize(
                    frame,
                    dsize=None,
                    fx=1/heatmap_scale,
                    fy=1/heatmap_scale)
                    )
                last_frame_save_time = 0
            else:
                last_frame_save_time += 1

            current_video_frame_number += 1
            global_frame_number += 1

            row = data.iloc[
                (data["frame_number"] - current_video_frame_number).abs().argsort()[:1]
            ].values

            if len(row) == 0:
                continue

            row = row[0]

            if len(row) == 4:
                _, frame_number, frame_time, results = row
            elif len(row) == 3:
                frame_number, frame_time, results = row
            else:
                frame_number, results = row

            if ((current_video_frame_number % skip_frame != 0)
                or (current_video_frame_number < start_frame)):
                skip_current_frame = True
            else:
                skip_current_frame = False

            frame_time_str, frame_time = calculate_frame_time(
                video_start_time, frame_number, fps
            )

            frame_data = {
                "frame_number": frame_number,
                "frame_time": frame_time_str,
                "objects_count": 0,
                "object_data": [],
                "regions_count": {},
                "labels_count": {},
            }

            results = json.loads(results.replace("'", '"'))

            for result in results:
                uid = result["id"]
                x, y, w, h = map(int, result["box"])
                conf = round(float(result["conf"]), 1)

                label_id = result.get("label_id", 0)

                obj_label = model_labels.get(str(label_id), label_id)

                if obj_label != "person":
                    continue

                sub_labels = result.get("sub_labels", [])
                if not sub_labels:
                    if uid in SubLabelsIds:
                        sub_labels = SubLabelsIds[uid]["labels"]

                        if SubLabelsIds[uid]["timeout"] > SubLabelsTimeout:
                            del SubLabelsIds[uid]
                        else:
                            SubLabelsIds[uid]["timeout"] += 1
                    else:
                        for obj_recog in recog_models:
                            xi, yi = x - (w // 2), y - (h // 2)
                            xf, yf = x + (w // 2), y + (h // 2)
                            object_image = frame[yi:yf, xi:xf, :]
                            _, object_label = obj_recog.recognize(object_image)
                            # print(object_label)
                            sub_labels.append(object_label[0])
                        SubLabelsIds[uid] = {"labels": sub_labels, "timeout": 0}
                else:
                    sub_labels = [sub_labels]

                if "not_human" in sub_labels:
                    continue

                object_regions = historic_data.insert(
                    uid,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    label=obj_label,
                    sub_labels=sub_labels,
                    time=frame_time,
                    frame_number=frame_number,
                )

                direction = ""
                if uid in historic_data.objects_list:
                    direction = historic_data.objects_list[uid].motion_direction(
                        region="entrance", min_dir_samples=10
                    )

                for region in object_regions:
                    frame_data["regions_count"][region] = 1 + frame_data[
                        "regions_count"
                    ].get(region, 0)

                for label in sub_labels:
                    frame_data["labels_count"][label] = 1 + frame_data[
                        "labels_count"
                    ].get(label, 0)
                    frame_heatmap_data["x"].append(int(x / heatmap_scale))
                    frame_heatmap_data["y"].append(int(y / heatmap_scale))
                    frame_heatmap_data["label"].append(label)

                if not skip_current_frame and show_frame:
                    draw_on_frame(
                        frame,
                        [x, y, w, h],
                        conf,
                        f"{direction} {uid} - {' '.join(sub_labels)}",
                    )

                object_regions = [f"Region:{region}" for region in object_regions]

                if direction == "D":
                    direction = "Down"
                if direction == "U":
                    direction = "Up"

                frame_data["object_data"].append(
                    {
                        "uid": uid,
                        "box": [x, y, w, h],
                        "labels": [obj_label]+sub_labels+object_regions+[f"Direction:{direction}"]
                    }
                )

            region_counter = {}
            for rc in regions_unique_id_counter:
                rc.update(frame_data["object_data"])
                region_counter[rc.label] = rc.count_greater_than_min

            labels_counters = {}
            for rc in labels_unique_id_counter:
                rc.update(frame_data["object_data"])
                labels_counters[rc.label] = rc.count_greater_than_min

            up_direction_unique_counter.update(frame_data["object_data"])
            down_direction_unique_counter.update(frame_data["object_data"])

            down_unique_hourly_counter.update(
                frame_time,
                frame_data["object_data"]
                )
            down_unique_15m_counter.update(
                frame_time,
                frame_data["object_data"]
                )

            # Save Heatmap
            frame_heatmap_data = pd.DataFrame(frame_heatmap_data)
            frame_heatmap_data.to_csv(heatmap_filepath, mode="a", index=False)

            if show_frame and display_region and not skip_current_frame:
                for polygon in polygons:
                    cv2.polylines(
                        frame,
                        [np.array(polygon["polygon"], dtype=np.int32)],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2,
                    )

            diff = (
                down_direction_unique_counter.count - up_direction_unique_counter.count
                )

            entrance_inside_count = 0 if diff < 0 else diff

            if not show_frame:
                frame = None

            if frame_data["labels_count"].get("cashier", 0) == 0:
                cashier_time_missing += 1
                cashier_hourly_counter.not_seen(frame_time)
            else:
                cashier_time_available += 1
                cashier_hourly_counter.seen(frame_time)

            cashier_time_missing_str = str(
                datetime.timedelta(seconds=int((cashier_time_missing) // fps))
            )
            cashier_time_available_str = str(
                datetime.timedelta(seconds=int((cashier_time_available) // fps))
            )

            if skip_current_frame:
                continue

            time_stats["End"] = time.time() - timer_frame_start

            h, w, _ = frame.shape
            target_height = int(720 / 2)
            target_width = int(1280 / 2)

            if h != target_height or w != target_width:
                frame = cv2.resize(frame, (target_width, target_height))

            output = {
                "frame": frame,
                "metadata": {
                    "frame_number": frame_number,
                    "total_frames": frame_counts,
                    "frame_time": frame_time_str,
                    "time_since_start": str(
                        datetime.timedelta(seconds=global_frame_number / fps)
                    ),
                    "time_cashier_missing": cashier_time_missing_str,
                    "time_cashier_available": cashier_time_available_str,
                    "output_dir_hourly_count": down_unique_hourly_counter.get_count()["Yaxis"],
                    "output_dir_fifteen_min_count": down_unique_15m_counter.get_count(),
                    "customers": frame_data["labels_count"].get("other", 0),
                    "historic_unique_region_count": region_counter,
                    "entrance_down_count": down_direction_unique_counter.count,
                    "entrance_inside_count": entrance_inside_count,
                    "cashier_time_count": cashier_hourly_counter.get_count(normalize=fps*60, format="2-axis"),
                },
            }

            frame_send_delay = time.time() - frame_send_time
            diff = frame_send_delay - frame_delay
            if diff < 0:
                time.sleep(diff * -1)
                yield output

            frame_send_time = time.time()

            # time_stats["Send"] = time.time() - timer_frame_start


async def live_channel_heatmap(
    channel_name: str, heatmap_delay=10, cashier_recognition=True, target_label=None
):

    print(channel_name)
    print("Heatmap", heatmap_delay)

    frame_filepath = f"temp/{channel_name}_frame.jpg"
    heatmap_filepath = f"temp/{channel_name}_heatmap.csv"

    while True:
        if os.path.exists(frame_filepath):
            frame = cv2.imread(frame_filepath)
            heatmap_data = pd.read_csv(heatmap_filepath)
        else:
            await asyncio.sleep(1)
            print("Data Not Found")
            continue

        if target_label is None:
            location_histgram = heatmap_data.groupby(["x", "y"]).count().reset_index()
        else:
            location_histgram = (
                heatmap_data[heatmap_data["label"] == target_label]
                .groupby(["x", "y"])
                .count()
                .reset_index()
            )

        location_histgram = location_histgram.values
        hm_points = []
        for temp in location_histgram:
            x, y, count = temp
            hm_points.append([[int(x), int(y)], int(count)])

        heatmap = generate_heatmap_points(frame, hm_points)

        h, w, _ = heatmap.shape
        target_height = int(720 / 2)
        target_width = int(1280 / 2)

        if h != target_height or w != target_width:
            heatmap = cv2.resize(heatmap, (target_width, target_height))

        yield heatmap

        await asyncio.sleep(heatmap_delay)


if __name__ == "__main__":
    import time

    def main(channel_name):
        start_time = time.time()
        for d in channel_analytics(
            channel_name=channel_name,
            start_frame=0,
            skip_frame=3,
            show_frame=True,
            cashier_recognition=True,
        ):
            # metadata = d["metadata"]
            frame = d["frame"]
            # print(metadata)

            cv2.imshow("frame", frame)
            cv2.waitKey(1)

            # start_time = time.time()

    # main("ch16")
    main("ch03")
