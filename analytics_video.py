import asyncio
import datetime
import json
import os
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


VIDEO_PATH = "."

with open(file="models/labels.json") as fp:
    model_labels = json.load(fp=fp)

cashier_recognition_model = ObjectRecognition("cashier_clf")

delay = 300


async def analytics(
    video_name: str,
    start_frame=0,
    skip_frame=1,
    show_frame: bool = False,
    cashier_recognition: bool = False,
    display_region: bool = True,
):
    base_video_name = video_name.split(".")[0]
    video_start_time = get_video_starttime(base_video_name)
    video_path = os.path.join(f"{VIDEO_PATH}/{video_name}")

    # load video
    cap = cv2.VideoCapture(video_path)
    frame_counts = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    data = pd.read_csv(f"{DATA_PATH}/{base_video_name}.csv")

    print(fps)

    try:
        polygons = load_roi(base_video_name.split("_")[0])
    except FileNotFoundError as e:
        print(e)
        polygons = []

    hourly_count = None

    recog_models = []
    if cashier_recognition:
        recog_models.append(cashier_recognition_model)

    historic_data = ObjectsHistoricData(regions=polygons)
    live_analytics = LiveAnalytics()

    cashier_time_missing = 0
    cashier_time_available = 0

    min_count = 0
    indx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        indx += 1

        # row = data[data["frame_number"] == indx].values
        row = data.iloc[(data["frame_number"] - indx).abs().argsort()[:1]].values

        if len(row) == 0:
            continue

        row = row[0]
        if len(row) == 4:
            _, frame_number, frame_time, results = row
        elif len(row) == 3:
            frame_number, frame_time, results = row
        else:
            frame_number, results = row

        if indx % skip_frame != 0:
            skip_current_frame = True
        else:
            skip_current_frame = False

        results = json.loads(results.replace("'", '"'))

        if indx < start_frame:
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

        for result in results:
            uid = result["id"]
            x, y, w, h = result["box"]
            x, y, w, h = int(x), int(y), int(w), int(h)
            conf = round(float(result["conf"]), 1)

            # if conf < 0.7:
            #     continue

            label_id = result.get("label_id", 0)
            # print(label_id)
            if str.isdigit(str(label_id)):
                obj_label = model_labels[f"{label_id}"]
            else:
                obj_label = label_id

            if obj_label != "person":
                continue

            sub_labels = []
            for obj_recog in recog_models:
                xi, yi = x - (w // 2), y - (h // 2)
                xf, yf = x + (w // 2), y + (h // 2)
                object_image = frame[yi:yf, xi:xf, :]
                _, object_label = obj_recog.recognize(object_image)
                # print(object_label)
                sub_labels.append(object_label[0])
                obj_recog.insert(uid, object_label[0])

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

            direction = historic_data.objects_list[uid].motion_direction(
                region="entrance", min_dir_samples=10
            )

            for r in object_regions:
                for _l in sub_labels:
                    live_analytics.add(
                        frame_number=frame_number,
                        frame_time=frame_time,
                        uid=uid,
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        region=r,
                        label=_l,
                        video_direction=direction,
                    )

            for region in object_regions:
                frame_data["regions_count"][region] = 1 + frame_data[
                    "regions_count"
                ].get(region, 0)

            for label in sub_labels:
                frame_data["labels_count"][label] = 1 + frame_data["labels_count"].get(
                    label, 0
                )

            if not skip_current_frame and show_frame:
                draw_on_frame(
                    frame,
                    [x, y, w, h],
                    conf,
                    f"{direction} {uid} - {' '.join(sub_labels)}",
                )

        entrance_count = live_analytics.column_unqiue_uid_count(
            "video_direction", skip_count=int(fps * 1)
        )

        if isinstance(entrance_count, dict):
            entrance_down_count = entrance_count.get("D", 0)
            entrance_up_count = entrance_count.get("U", 0)

            diff = entrance_down_count - entrance_up_count
            if diff < min_count:
                min_count = diff

            entrance_inside_count = 0 if diff < 0 else diff
        else:
            entrance_down_count = 0
            entrance_up_count = 0
            entrance_inside_count = 0

        # write_on_frame(frame, 10, 150, text=f"{entrance_down_count}")
        # print(frame_data)

        if frame_data["labels_count"].get("cashier", 0) == 0:
            cashier_time_missing += 1
        else:
            cashier_time_available += 1

        if show_frame and display_region and not skip_current_frame:
            for polygon in polygons:
                cv2.polylines(
                    frame,
                    [np.array(polygon["polygon"], dtype=np.int32)],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )

        if not show_frame:
            frame = None

        label_time_hour_count = live_analytics.timely_count_day(
            current_time=frame_time,
            group_by="video_direction",
            group_filter="D",
            freq=60,
            skip_count=fps,
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

        label_time_count = live_analytics.timely_count_day(
            current_time=frame_time,
            group_by="label",
            group_filter="other",
            freq=15,
            skip_count=fps * 2,
        )

        output_15m_count = react_xy_graph_data(label_time_count)

        cashier_time_count = live_analytics.timely_apearance(
            current_time=frame_time,
            group_by="label",
            group_filter="cashier",
            freq=60,
            skip_count=fps * 2,
        )
        cashier_time_count["frame_number"] = cashier_time_count["frame_number"].apply(
            lambda x: x / fps
        )
        cashier_time_count = react_xy_graph_data(cashier_time_count)

        dir_15m_time_count = live_analytics.timely_count_day(
            current_time=frame_time,
            group_by="video_direction",
            group_filter="D",
            freq=15,
            skip_count=fps * 2,
        )

        # dir_hourly_count = {}
        output_dir_15m_count = react_xy_graph_data(dir_15m_time_count)

        cashier_time_missing_str = str(
            datetime.timedelta(seconds=int((cashier_time_missing) // fps))
        )
        cashier_time_available_str = str(
            datetime.timedelta(seconds=int((cashier_time_available) // fps))
        )

        if skip_current_frame:
            await asyncio.sleep(1 / delay)
            continue

        yield {
            "frame": frame,
            "metadata": {
                "frame_number": frame_number,
                "total_frames": frame_counts,
                "frame_time": frame_time_str,
                "time_since_start": str(datetime.timedelta(seconds=indx / fps)),
                "total_people": historic_data.get_unique_ids(),  # useless
                "time_cashier_missing": cashier_time_missing_str,
                "time_cashier_available": cashier_time_available_str,
                "output_dir_hourly_count": output_dir_hourly_count,
                "output_fifteen_min_count": output_15m_count,
                "output_dir_fifteen_min_count": output_dir_15m_count,
                "total_visitors": historic_data.get_unique_visitors(
                    skip_count=fps
                ),  # redundant
                "customers": frame_data["labels_count"].get("other", 0),  # redundant
                "counts": frame_data["labels_count"].get("other", 0),
                "historic_unique_label_count": live_analytics.column_unqiue_uid_count(
                    "label", skip_count=fps * 2
                ),  # redundant
                "historic_unique_region_count": live_analytics.column_unqiue_uid_count(
                    "region", skip_count=fps * 2
                ),  # redundant
                "roi_counter": frame_data["regions_count"],  # renamed
                "sub_labels_counter": frame_data["labels_count"],
                # new keys
                "frame_regions_count": frame_data["regions_count"],
                "frame_labels_count": frame_data["labels_count"],
                "historic_region_unique_id_count": live_analytics.column_unqiue_uid_count(
                    "region", skip_count=fps * 2
                ),
                "historic_region_labels_unique_id_count": live_analytics.columns_unqiue_uid_count(
                    "label", "region", skip_count=fps
                ),
                "historic_region_total_count": live_analytics.column_occurance(
                    "region", fps=fps, convert_to_time=True
                ),
                "historic_label_unique_id_count": live_analytics.column_unqiue_uid_count(
                    "label", skip_count=fps * 2
                ),
                "historic_label_total_count": live_analytics.column_occurance(
                    "label", fps=fps, convert_to_time=True
                ),
                "entrance_down_count": entrance_down_count,
                "entrance_inside_count": entrance_inside_count,
                "cashier_time_count": cashier_time_count,
            },
        }
        await asyncio.sleep(1 / delay)


def react_xy_graph_data(data):
    output = {"Xaxis": [], "Yaxis": []}
    if data is not None:
        for row in data.iterrows():
            row = row[1]
            output["Xaxis"].append(row.iloc[1])
            output["Yaxis"].append(row.iloc[2])
    return output


async def live_heatmap(
    video_name: str, heatmap_delay=10, cashier_recognition=True, target_label=None
):
    base_video_name = video_name.split(".")[0]
    video_path = os.path.join(f"{VIDEO_PATH}/{video_name}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    data = pd.read_csv(f"{DATA_PATH}/{base_video_name}.csv")

    recog_models: List[ObjectRecognition] = []
    if cashier_recognition:
        recog_models.append(ObjectRecognition("cashier_clf"))

    location_histgram = {}

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

            if cashier_recognition:
                xi, yi = x - (w // 2), y - (h // 2)
                xf, yf = x + (w // 2), y + (h // 2)

                object_image = frame[yi:yf, xi:xf, :]

                sub_labels = []
                for obj_recog in recog_models:
                    _, object_label = obj_recog.recognize(object_image)
                    sub_labels.append(object_label[0])
                    obj_recog.insert(uid, object_label[0])
            else:
                sub_labels = ["other", 1]

            if "not_human" in [_l[0] for _l in sub_labels]:
                continue

            if target_label is None:
                location_histgram[f"{x}_{y}"] = 1 + location_histgram.get(f"{x}_{y}", 0)
            elif target_label in sub_labels:
                location_histgram[f"{x}_{y}"] = 1 + location_histgram.get(f"{x}_{y}", 0)

        if (indx % (heatmap_delay * fps)) == 0:
            hm_points = []
            for loc, count in location_histgram.items():
                x, y = loc.split("_")
                hm_points.append([[int(x), int(y)], count])

            heatmap = generate_heatmap_points(frame, hm_points)
            yield heatmap

        await asyncio.sleep(1 / delay)


if __name__ == "__main__":
    # from tqdm import tqdm

    # ch03_20240522212448
    # ch03_20240522041053
    # ch13_20240522034400
    # ch13_20240522073829
    # ch13_20240522082610
    # ch11_20240522031430
    # ch16_20240522034137

    async def main():
        size = 1
        async for d in analytics(
            video_name="ch03_20240522000001.mp4",
            # start_frame=20 * 25,
            start_frame=0,
            skip_frame=3,
            show_frame=True,
            cashier_recognition=False,
        ):
            metadata = d["metadata"]
            # print(metadata)
            print(
                metadata["time_since_start"],
                "\n",
                metadata["frame_regions_count"],
                metadata["frame_labels_count"],
                "\n",
                metadata["time_cashier_missing"],
                metadata["time_cashier_available"],
                "\n",
                metadata["historic_region_total_count"],
                metadata["historic_region_unique_id_count"],
                metadata["historic_region_labels_unique_id_count"],
                "\n",
                metadata["historic_label_total_count"],
                metadata["historic_label_unique_id_count"],
                "\n",
                "Out Direction Hourly Count",
                metadata["output_dir_hourly_count"],
                "\n",
                # metadata["output_dir_hourly_count"], "\n",
                metadata["entrance_down_count"],
                metadata["entrance_inside_count"],
                "\n",
                metadata["cashier_time_count"],
            )

            frame = cv2.resize(d["frame"], dsize=None, fx=size, fy=size)

            cv2.imshow("image", frame)
            if cv2.waitKey(1) == ord("q"):
                break

    asyncio.run(main())

    # async def heatmap_main():
    #     size = 0.7
    #     async for heatmap in live_heatmap(
    #         video_name="ch16_20240522034137.mp4",
    #         heatmap_delay=10,
    #         cashier_recognition=True,
    #         target_label="cashier",
    #     ):
    #         frame = cv2.resize(heatmap, dsize=None, fx=size, fy=size)

    #         cv2.imshow("image", frame)
    #         if cv2.waitKey(1) == ord("q"):
    #             break

    # asyncio.run(heatmap_main())
