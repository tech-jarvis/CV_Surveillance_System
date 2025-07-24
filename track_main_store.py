import csv
import os

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from config import DATA_PATH, DETECTION_MODEL_PATH, VIDEO_PATH, DEVICE
from object_recognition import ObjectRecognition
from utils import calculate_frame_time, get_video_starttime

model = YOLO(DETECTION_MODEL_PATH)
cashier_recog = ObjectRecognition("cashier_clf")

VIDEO_PATH = "."



def model_track_people(video_name, skip_frames=5):
    base_video_name = video_name.split(".")[0]
    video_path = os.path.join(f"{VIDEO_PATH}/{video_name}")

    video_start_time = get_video_starttime(base_video_name)

    cap = cv2.VideoCapture(video_path)

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # tracked_classes = [0]

    tracking_results = {"frame_number": [], "frame_time": [], "results": []}

    output_csv = f"{DATA_PATH}/{base_video_name}_raw_v1.csv"

    csvfile = open(output_csv, "w")  # Open in append mode
    writer = csv.writer(csvfile)
    # Write header only once (assuming the CSV doesn't exist)
    if os.stat(output_csv).st_size == 0:
        writer.writerow(["frame_number", "frame_time", "results"])

    for frame_number in tqdm(range(int(frame_count))):
        success, frame = cap.read()

        if skip_frames > 0:
            if frame_number % skip_frames != 0:
                continue

        if frame is None:
            continue

        if not success:
            break

        results = model.track(
            frame,
            persist=True,
            # classes=tracked_classes,
            # augment=True,
            tracker="botsort.yaml",
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
            x, y, w, h = [int(v) for v in box]
            xi, yi = x - (w // 2), y - (h // 2)
            xf, yf = x + (w // 2), y + (h // 2)
            object_image = frame[yi:yf, xi:xf, :]

            obj_label = model.names[int(obj_cls)]
            sr, labels = cashier_recog.recognize(object_image)

            results.append(
                {
                    "box": box,
                    "conf": conf,
                    "label_id": obj_label,
                    "sub_labels": labels[0],
                    "sub_labels_conf": float(labels[1]),
                    "id": track_id,
                }
            )
        # print(results)

        tracking_results["frame_number"].append(frame_number)
        tracking_results["results"].append(results)

        frame_time_str, frame_time = calculate_frame_time(
            video_start_time, frame_number, fps
        )

        tracking_results["frame_time"].append(frame_time_str)

        # Write data to CSV after processing each frame
        writer.writerow([frame_number, frame_time_str, results])

    csvfile.close()
    tracking_results = pd.DataFrame(tracking_results)
    tracking_results.to_csv(f"{DATA_PATH}/{base_video_name}_v1.csv", index=False)

    cap.release()
    # cv2.destroyAllWindows()



if __name__ == "__main__":
    import os
    import time

    # video_name = "ch16_20240522034137.mp4"
    # video_name = "ch03_20240522000000.mp4"

    video_name = "ch03_20240522000001.mp4"

    # video_path = "ch03_20240522000000.mp4"
    # video_path = "ch16_20240522000000.mp4"
    # video_name = 'ch16_20240522000000.mp4'
    # video_name = "ch11_20240522010854.mp4"

    video_done_filepath = "data/videos_done.txt"

    videos_done = []
    with open(video_done_filepath) as fp:
        videos_done = [fline.strip() for fline in fp.readlines()]

    print(videos_done)

    # for video_name in os.listdir("//172.16.0.250/ids_kassa"):
    for video_name in os.listdir(VIDEO_PATH):
        # print(video_name)

    # for video_name in ["ch03_20240522111536.mp4", "ch16_20240522100101.mp4"]:
        if not video_name.endswith(".mp4"):
            continue
        if video_name in videos_done:
            print("ALREADY DONE", video_name)
            continue
            
        print(video_name)
        model_track_people(video_name, skip_frames=0)
        videos_done.append(video_name)
        with open(video_done_filepath, "a") as fp:
            fp.writelines([video_name+"\n"])
    
        time.sleep(1)
        # time.sleep(10)
