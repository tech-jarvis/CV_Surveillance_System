import asyncio
import datetime
import json
import os
import time
from typing import List

import cv2
import numpy as np
import pandas as pd

from config import DATA_PATH, VIDEO_PATH
from generate_heatmap import generate_heatmap_points
from live_analytics import LiveAnalytics
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

heatmap_data = {"x": [], "y": [], "label": []}

heatmap_data = pd.DataFrame(heatmap_data)
Recent_Data = {}


def channel_analytics(
    channel_name: str,
    start_frame=0,
    skip_frame=1,
    show_frame: bool = False,
    cashier_recognition: bool = False,
    display_region: bool = True,
):
    global heatmap_data, Recent_Data
    channel_files = [
        f for f in sorted(os.listdir(VIDEO_PATH)) if f.startswith(channel_name)
    ]

    print(VIDEO_PATH)
    print(channel_name)
    print(channel_files)

    print(start_frame, skip_frame, show_frame, cashier_recognition, display_region)

    Recent_Data[channel_name] = {}

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

    live_analytics = LiveAnalytics()

    calculate_rate = 25 * 10
    last_calculation_frame = calculate_rate

    global_frame_number = 0

    heatmap_data = {"x": [], "y": [], "label": []}

    heatmap_data = pd.DataFrame(heatmap_data)
    heatmap_filepath = f"temp/{channel_name}_heatmap.csv"
    heatmap_data.to_csv(heatmap_filepath, index=False)

    SubLabelsIds = {}
    SubLabelsTimeout = 30

    for video_name in channel_files:
        base_video_name = video_name.split(".")[0]
        video_start_time = get_video_starttime(base_video_name)
        video_path = os.path.join(f"{VIDEO_PATH}/{video_name}")

        # load video
        cap = cv2.VideoCapture(video_path)
        frame_counts = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        data = pd.read_csv(f"{DATA_PATH}/{base_video_name}.csv")

        print(fps)

        min_count = 0
        current_video_frame_number = 0

        last_analytics = {}

        SlowVideoSkipNextFrame = False

        while True:
            timer_frame_start = time.time()
            time_stats = {}

            last_calculation_frame += 1

            ret, frame = cap.read()

            frame_heatmap_data = {"x": [], "y": [], "label": []}

            if not ret:
                print(video_name)
                print("Error Ret")
                break

            Recent_Data[channel_name]["frame"] = frame.copy()

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

            if current_video_frame_number % skip_frame != 0:
                skip_current_frame = True
            else:
                skip_current_frame = False

            if current_video_frame_number < start_frame:
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
                "regions_count": {},
                "labels_count": {},
            }

            results = json.loads(results.replace("'", '"'))

            for result in results:
                uid = result["id"]
                x, y, w, h = result["box"]
                x, y, w, h = int(x), int(y), int(w), int(h)
                conf = round(float(result["conf"]), 1)

                label_id = result.get("label_id", 0)
                # print(label_id)
                if str.isdigit(str(label_id)):
                    obj_label = model_labels[f"{label_id}"]
                else:
                    obj_label = label_id

                if obj_label != "person":
                    continue

                sub_labels = result.get("sub_labels", [])
                if len(sub_labels) == 0:
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

                if isinstance(sub_labels, str):
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

                try:
                    for region in object_regions:
                        for _l in sub_labels:
                            live_analytics.add(
                                frame_number=frame_number,
                                frame_time=frame_time,
                                uid=uid,
                                x=x,
                                y=y,
                                w=w,
                                h=h,
                                region=region,
                                label=_l,
                                video_direction=direction,
                            )

                        frame_data["regions_count"][region] = 1 + frame_data[
                            "regions_count"
                        ].get(region, 0)

                except Exception as e:
                    print("Object Region", e)

                for label in sub_labels:
                    frame_data["labels_count"][label] = 1 + frame_data[
                        "labels_count"
                    ].get(label, 0)
                    frame_heatmap_data["x"].append(x)
                    frame_heatmap_data["y"].append(y)
                    frame_heatmap_data["label"].append(label)

                if not skip_current_frame and show_frame:
                    draw_on_frame(
                        frame,
                        [x, y, w, h],
                        conf,
                        f"{direction} {uid} - {' '.join(sub_labels)}",
                    )

            # print(channel_name, video_name, "Analytics")
            time_stats["Result Processed"] = time.time() - timer_frame_start

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

            if last_calculation_frame >= calculate_rate:
                last_calculation_frame = 0

                entrance_count = live_analytics.column_unqiue_uid_count(
                    "video_direction", skip_count=int(fps * 1)
                )

                entrance_down_count = 0
                entrance_up_count = 0
                entrance_inside_count = 0

                if isinstance(entrance_count, dict):
                    entrance_down_count = entrance_count.get("D", 0)
                    entrance_up_count = entrance_count.get("U", 0)

                    diff = entrance_down_count - entrance_up_count
                    if diff < min_count:
                        min_count = diff

                    entrance_inside_count = 0 if diff < 0 else diff

                time_stats["Entrance Count"] = time.time() - timer_frame_start

                ##
                output_dir_hourly_count = calculate_output_dir_hourly_count(
                    live_analytics, fps, frame_time
                )
                cashier_time_count = calculate_cashier_time(
                    live_analytics, fps, frame_time, global_frame_number
                )
                historic_unique_region_count = calculate_historic_unique_region_count(
                    live_analytics, fps
                )
                output_dir_15m_count = calculate_output_dir_15m_count(
                    live_analytics, fps, frame_time
                )
                ##

                last_analytics["entrance_down_count"] = entrance_down_count
                last_analytics["entrance_inside_count"] = entrance_inside_count
                last_analytics["output_dir_15m_count"] = output_dir_15m_count
                last_analytics["cashier_time_count"] = cashier_time_count
                last_analytics["historic_unique_region_count"] = (
                    historic_unique_region_count
                )
                last_analytics["output_dir_hourly_count"] = output_dir_hourly_count

            else:
                entrance_down_count = last_analytics["entrance_down_count"]
                entrance_inside_count = last_analytics["entrance_inside_count"]
                output_dir_15m_count = last_analytics["output_dir_15m_count"]
                cashier_time_count = last_analytics["cashier_time_count"]
                historic_unique_region_count = last_analytics[
                    "historic_unique_region_count"
                ]

            if not show_frame:
                frame = None

            if frame_data["labels_count"].get("cashier", 0) == 0:
                cashier_time_missing += 1
            else:
                cashier_time_available += 1

            cashier_time_missing_str = str(
                datetime.timedelta(seconds=int((cashier_time_missing) // fps))
            )
            cashier_time_available_str = str(
                datetime.timedelta(seconds=int((cashier_time_available) // fps))
            )

            if skip_current_frame:
                continue

            time_stats["End"] = time.time() - timer_frame_start
            # print(channel_name, video_name, "End")
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
                    "output_dir_hourly_count": output_dir_hourly_count,
                    "output_dir_fifteen_min_count": output_dir_15m_count,
                    "customers": frame_data["labels_count"].get("other", 0),
                    "historic_unique_region_count": historic_unique_region_count,
                    "entrance_down_count": entrance_down_count,
                    "entrance_inside_count": entrance_inside_count,
                    "cashier_time_count": cashier_time_count,
                },
            }

            if not SlowVideoSkipNextFrame:
                yield output
            else:
                SlowVideoSkipNextFrame = False

            time_stats["Send"] = time.time() - timer_frame_start
            # print(video_name, frame_number, time_stats)
            # print(frame.shape)

            if (time.time() - timer_frame_start) > 0.5:
                SlowVideoSkipNextFrame = True
            # print(channel_name, video_name, "Send")


def calculate_output_dir_hourly_count(live_analytics, fps, frame_time):
    label_time_hour_count = live_analytics.timely_count_day(
        current_time=frame_time,
        group_by="video_direction",
        group_filter="D",
        freq=60,
        skip_count=5,
    )

    hourly_count = {}
    if label_time_hour_count is not None:
        for row in label_time_hour_count.iterrows():
            row = row[1]
            hourly_count[row.iloc[1]] = row.iloc[2]

    output_dir_hourly_count = []
    for h in range(24):
        hn = f"{h:02}:00"
        if hn in hourly_count:
            output_dir_hourly_count.append(hourly_count[hn])
        else:
            output_dir_hourly_count.append(0)

    return output_dir_hourly_count


def calculate_cashier_time(live_analytics, fps, frame_time, global_frame_number):
    cashier_time_count = live_analytics.timely_apearance(
        current_time=frame_time,
        group_by="label",
        group_filter="cashier",
        freq=60,
        skip_count=fps,
    )
    cashier_time_count["frame_number"] = cashier_time_count["frame_number"].apply(
        lambda x: x / fps
    )

    cashier_time_count = cashier_time_count.rename(
        columns={"frame_number": "available"}
    )
    cashier_time_count = cashier_time_count.drop("frame_time", axis=1)
    cashier_time_count["unavailable"] = (
        global_frame_number / fps
    ) - cashier_time_count["available"]
    cashier_time_count["unavailable"] = cashier_time_count["unavailable"].apply(
        lambda x: round(x) if x > 0 else 0
    )
    cashier_time_count = cashier_time_count.fillna(0)

    cashier_time_count["unavailable"] = cashier_time_count["unavailable"].apply(
        lambda x: x / 60
    )
    cashier_time_count["available"] = cashier_time_count["available"].apply(
        lambda x: x / 60
    )

    return react_xy_graph_data(cashier_time_count)


def calculate_output_dir_15m_count(live_analytics, fps, frame_time):
    dir_15m_time_count = live_analytics.timely_count_day(
        current_time=frame_time,
        group_by="video_direction",
        group_filter="D",
        freq=15,
        skip_count=fps * 2,
    )

    return react_xy_graph_data(dir_15m_time_count)


def calculate_historic_unique_region_count(live_analytics, fps):
    return live_analytics.column_unqiue_uid_count("region", skip_count=fps * 2)


def react_xy_graph_data(data):
    output = {"Xaxis": [], "Yaxis": []}
    if data is not None:
        for row in data.iterrows():
            row = row[1]
            output["Xaxis"].append(row.iloc[1])
            output["Yaxis"].append(row.iloc[2])
    return output


async def live_channel_heatmap_old(
    channel_name: str, heatmap_delay=10, cashier_recognition=True, target_label=None
):
    channel_files = [
        f for f in sorted(os.listdir(VIDEO_PATH)) if f.startswith(channel_name)
    ]
    print(VIDEO_PATH)
    print(channel_name)
    print(channel_files)
    print("Heatmap", heatmap_delay)

    location_histgram = {}

    recog_models: List[ObjectRecognition] = []
    if cashier_recognition:
        recog_models.append(cashier_recognition_model)

    for video_name in channel_files:
        base_video_name = video_name.split(".")[0]
        video_path = os.path.join(f"{VIDEO_PATH}/{video_name}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        data = pd.read_csv(f"{DATA_PATH}/{base_video_name}.csv")

        indx = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            indx += 1

            row = data[data["frame_number"] == indx].values

            if len(row) == 0:
                continue

            row = row[0]
            if len(row) == 4:
                _, frame_number, frame_time, results = row
            elif len(row) == 3:
                frame_number, frame_time, results = row
            else:
                frame_number, results = row

            results = json.loads(results.replace("'", '"'))

            for result in results:
                uid = result["id"]
                x, y, w, h = result["box"]
                x, y, w, h = int(x), int(y), int(w), int(h)

                label_id = result.get("label_id", 0)

                if str.isdigit(str(label_id)):
                    obj_label = model_labels[f"{label_id}"]
                else:
                    obj_label = label_id

                if obj_label != "person":
                    continue

                sub_labels = result.get("sub_labels", [])
                if len(sub_labels) == 0:
                    for obj_recog in recog_models:
                        xi, yi = x - (w // 2), y - (h // 2)
                        xf, yf = x + (w // 2), y + (h // 2)
                        object_image = frame[yi:yf, xi:xf, :]
                        _, object_label = obj_recog.recognize(object_image)
                        # print(object_label)
                        sub_labels.append(object_label[0])

                if "not_human" in [_l[0] for _l in sub_labels]:
                    continue

                if target_label is None:
                    location_histgram[f"{x}_{y}"] = 1 + location_histgram.get(
                        f"{x}_{y}", 0
                    )
                elif target_label in sub_labels:
                    location_histgram[f"{x}_{y}"] = 1 + location_histgram.get(
                        f"{x}_{y}", 0
                    )

            if (indx % (heatmap_delay * fps)) == 0:
                hm_points = []
                for loc, count in location_histgram.items():
                    x, y = loc.split("_")
                    hm_points.append([[int(x), int(y)], count])

                heatmap = generate_heatmap_points(frame, hm_points)

                h, w, _ = heatmap.shape
                target_height = int(720 / 2)
                target_width = int(1280 / 2)

                if h != target_height or w != target_width:
                    heatmap = cv2.resize(heatmap, (target_width, target_height))

                yield heatmap

            await asyncio.sleep(1 / fps)


async def live_channel_heatmap(
    channel_name: str, heatmap_delay=10, cashier_recognition=True, target_label=None
):
    global heatmap_data, Recent_Data

    print(channel_name)
    print("Heatmap", heatmap_delay)

    while True:
        if channel_name in Recent_Data:
            frame = Recent_Data[channel_name]["frame"]
        else:
            await asyncio.sleep(1)
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

    def main(channel_name):
        for d in channel_analytics(
            channel_name=channel_name,
            start_frame=0,
            skip_frame=3,
            show_frame=True,
            cashier_recognition=True,
        ):
            metadata = d["metadata"]
            print(metadata)

            # frame = cv2.resize(d["frame"], dsize=None, fx=size, fy=size)

            # cv2.imshow("image", frame)
            # if cv2.waitKey(1) == ord("q"):
            #     break

    main("ch16")
