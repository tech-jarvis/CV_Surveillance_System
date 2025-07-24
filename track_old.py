import os
import cv2
import csv
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from utils import get_video_starttime, calculate_frame_time
from config import DETECTION_MODEL_PATH, DATA_PATH, VIDEO_PATH, DEVICE

model = YOLO(DETECTION_MODEL_PATH)

def model_track_people(video_name, skip_frames=5):

    base_video_name = video_name.split(".")[0]
    video_path = os.path.join(f"{VIDEO_PATH}/{video_name}")
    
    video_start_time = get_video_starttime(base_video_name)

    cap = cv2.VideoCapture(video_path)

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # tracked_classes = [0]

    tracking_results = {
        "frame_number": [],
        "frame_time": [],
        "results": []
    }

    output_csv = f"{DATA_PATH}/{base_video_name}_raw_v1.csv"

    csvfile = open(output_csv, "w", newline='')  # Open in append mode
    writer = csv.writer(csvfile)
    # Write header only once (assuming the CSV doesn't exist)
    if os.stat(output_csv).st_size == 0:
        writer.writerow(["frame_number", "frame_time", "results"])

    for frame_number in tqdm(range(int(frame_count))):
        success, frame = cap.read()
        
        if frame_number % skip_frames != 0:
            continue

        if frame is None:
            continue
        
        if not success:
            break

        results = model.track(frame,
                              persist=True,
                            #   classes=tracked_classes,
                            # augment=True,
                              tracker="botsort.yaml",
                              device=DEVICE,
                              verbose=False)

        boxes = results[0].boxes.xywh.cpu().tolist()
        cls_list = results[0].boxes.cls.cpu().numpy()
        conf_list = results[0].boxes.conf.cpu().numpy()

        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        except:
            track_ids = [-1 for _ in boxes]
        
        results = []
        for (box, track_id, obj_cls, conf) in zip(boxes, track_ids, cls_list, conf_list):
            results.append(
                {
                    "box": box,
                    "conf": conf,
                    "label_id": int(obj_cls),
                    "id": track_id
                }
            )

        # print(results)

        tracking_results["frame_number"].append(frame_number)
        tracking_results["results"].append(results)

        frame_time_str , frame_time = calculate_frame_time(video_start_time, frame_number, fps)

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

    # video_name = "ch11_20240522031430.mp4"

    # video_path = "ch03_20240522000000.mp4"
    # video_path = "ch16_20240522000000.mp4"
    # video_name = 'ch16_20240522000000.mp4'
    # video_name = "ch11_20240522010854.mp4"

    video_done_filepath = "data/videos_done.txt"

    videos_done = []
    with open(video_done_filepath) as fp:
        videos_done = [fline.strip() for fline in fp.readlines()]

    print(videos_done)

    try:
        while True:
            for video_name in os.listdir("//172.16.0.250/ids_kassa"):
                
                if not video_name.endswith(".mp4"):
                    continue
                if video_name in videos_done:
                    print("ALREADY DONE", video_name)
                    continue
                    
                print(video_name)
                model_track_people(video_name, skip_frames=3)

                with open(video_done_filepath, "a") as fp:
                    fp.writelines([video_name+"\n"])
            
                time.sleep(1)
            time.sleep(10)
    except Exception as e:
        print(e)
