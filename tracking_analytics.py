import csv
import os
import datetime

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from config import DATA_PATH, DETECTION_MODEL_PATH, VIDEO_PATH, DEVICE
from object_recognition import ObjectRecognition
from objects_data import ObjectsHistoricData
from live_analytics import LiveAnalytics
from utils import calculate_frame_time, get_video_starttime, draw_on_frame, load_roi

model = YOLO(DETECTION_MODEL_PATH)
cashier_recog = ObjectRecognition("cashier_clf")

delay = 300

DATA_PATH = "data"

def react_xy_graph_data(data):
    output = {"Xaxis": [], "Yaxis": []}
    if data is not None:
        for row in data.iterrows():
            row = row[1]
            output["Xaxis"].append(row.iloc[1])
            output["Yaxis"].append(row.iloc[2])
    return output


def model_track_people(video_name, skip_frames=5):
    display_region = True

    base_video_name = video_name.split(".")[0]
    video_path = os.path.join(f"{VIDEO_PATH}/{video_name}")

    print(video_path)
    video_start_time = get_video_starttime(base_video_name)

    print(os.path.exists(video_path))
    cap = cv2.VideoCapture(video_path)
    print(cap)
    print(cap.isOpened())
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(fps, frame_count)
    print(width, height)

    # tracked_classes = [0]

    tracking_results = {"frame_number": [], "frame_time": [], "results": []}

    output_csv = f"{DATA_PATH}/{base_video_name}_raw_test.csv"

    csvfile = open(output_csv, "w")  # Open in append mode
    writer = csv.writer(csvfile)
    # Write header only once (assuming the CSV doesn't exist)
    if os.stat(output_csv).st_size == 0:
        writer.writerow(["frame_number", "frame_time", "results"])

    try:
        polygons = load_roi(base_video_name.split("_")[0])
    except FileNotFoundError as e:
        print(e)
        polygons = []

    hourly_count = None
    historic_data = ObjectsHistoricData(regions=polygons)
    live_analytics = LiveAnalytics()

    cashier_time_missing = 0
    cashier_time_available = 0

    min_count = 0
    indx = 0

    for frame_number in tqdm(range(int(frame_count))):
        success, frame = cap.read()

        if frame_number % skip_frames != 0:
            continue

        if frame is None:
            continue

        if not success:
            break

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

        results = model.track(
            frame,
            persist=True,
            # classes=tracked_classes,
            # augment=True,
            # tracker="botsort.yaml",
            tracker="bytetrack.yaml",
            device=DEVICE,
            verbose=False,
        )

        boxes = results[0].boxes.xywh.cpu().tolist()
        cls_list = results[0].boxes.cls.cpu().numpy()
        conf_list = results[0].boxes.conf.cpu().numpy()

        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        except TypeError:
            track_ids = [-1 for _ in boxes]
        except AttributeError:
            track_ids = [-1 for _ in boxes]

        results = []
        for box, track_id, obj_cls, conf in zip(boxes, track_ids, cls_list, conf_list):
            uid = track_id
            x, y, w, h = [int(v) for v in box]
            xi, yi = x - (w // 2), y - (h // 2)
            xf, yf = x + (w // 2), y + (h // 2)
            object_image = frame[yi:yf, xi:xf, :]
            obj_label = model.names[int(obj_cls)]

            sub_labels = []
            if False:
                sr, sl = cashier_recog.recognize(object_image)
            else:
                sl = ["other", 0.8]
            sub_labels.append(sl[0])
            results.append(
                {
                    "box": box,
                    "conf": conf,
                    "label": obj_label,
                    "sub_labels": sl[0],
                    "sub_labels_conf": float(sl[1]),
                    "id": track_id,
                }
            )

            if "not_human" in sub_labels or conf < 0.3 or obj_label != "person":
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

                conf_r = round(float(conf), 2)
                draw_on_frame(
                    frame,
                    [x, y, w, h],
                    str(conf_r),
                    f"{direction} {uid} - {' '.join(sub_labels)}",
                )

        tracking_results["frame_number"].append(frame_number)
        tracking_results["results"].append(results)

        frame_time_str, frame_time = calculate_frame_time(
            video_start_time, frame_number, fps
        )

        tracking_results["frame_time"].append(frame_time_str)

        # Write data to CSV after processing each frame
        writer.writerow([frame_number, frame_time_str, results])
        ## Analytics
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

        if display_region:
            for polygon in polygons:
                cv2.polylines(
                    frame,
                    [np.array(polygon["polygon"], dtype=np.int32)],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )

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

        frame = cv2.resize(frame, dsize=None, fx=0.8, fy=0.8)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)

        # print(
        #     {
        #         "frame_number": frame_number,
        #         "total_frames": frame_count,
        #         "frame_time": frame_time_str,
        #         "time_since_start": str(datetime.timedelta(seconds=indx / fps)),
        #         "total_people": historic_data.get_unique_ids(),  # useless
        #         "time_cashier_missing": cashier_time_missing_str,
        #         "time_cashier_available": cashier_time_available_str,
        #         "output_dir_hourly_count": output_dir_hourly_count,
        #         "output_fifteen_min_count": output_15m_count,
        #         "output_dir_fifteen_min_count": output_dir_15m_count,
        #         "total_visitors": historic_data.get_unique_visitors(
        #             skip_count=fps
        #         ),  # redundant
        #         "customers": frame_data["labels_count"].get("other", 0),  # redundant
        #         "counts": frame_data["labels_count"].get("other", 0),
        #         "historic_unique_label_count": live_analytics.column_unqiue_uid_count(
        #             "label", skip_count=fps * 2
        #         ),  # redundant
        #         "historic_unique_region_count": live_analytics.column_unqiue_uid_count(
        #             "region", skip_count=fps * 2
        #         ),  # redundant
        #         "roi_counter": frame_data["regions_count"],  # renamed
        #         "sub_labels_counter": frame_data["labels_count"],
        #         # new keys
        #         "frame_regions_count": frame_data["regions_count"],
        #         "frame_labels_count": frame_data["labels_count"],
        #         "historic_region_unique_id_count": live_analytics.column_unqiue_uid_count(
        #             "region", skip_count=fps * 2
        #         ),
        #         "historic_region_labels_unique_id_count": live_analytics.columns_unqiue_uid_count(
        #             "label", "region", skip_count=fps
        #         ),
        #         "historic_region_total_count": live_analytics.column_occurance(
        #             "region", fps=fps, convert_to_time=True
        #         ),
        #         "historic_label_unique_id_count": live_analytics.column_unqiue_uid_count(
        #             "label", skip_count=fps * 2
        #         ),
        #         "historic_label_total_count": live_analytics.column_occurance(
        #             "label", fps=fps, convert_to_time=True
        #         ),
        #         "entrance_down_count": entrance_down_count,
        #         "entrance_inside_count": entrance_inside_count,
        #         "cashier_time_count": cashier_time_count,
        #     }
        # )

    csvfile.close()
    tracking_results = pd.DataFrame(tracking_results)
    tracking_results.to_csv(f"{DATA_PATH}/{base_video_name}_test.csv", index=False)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # video_name = "ch16_20240522034137.mp4"

    video_name = "ch03_20240522000001.mp4"

    # video_path = "ch03_20240522000000.mp4"
    # video_path = "ch16_20240522000000.mp4"
    # video_name = 'ch16_20240522000000.mp4'
    # video_name = "ch11_20240522010854.mp4"

    # VIDEO_PATH = r"c:\Users\Administrator\Downloads"
    VIDEO_PATH = "."
    # video_name = "WIN_20240628123340.mp4"

    model_track_people(video_name, skip_frames=1)
