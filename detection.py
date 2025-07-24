import cv2
from ultralytics import YOLO

from config import DETECTION_MODEL_PATH, VIDEO_PATH, DEVICE

model = YOLO(DETECTION_MODEL_PATH)

print(model.names)
model_labels = [v for k, v in model.names.items()]
print(model_labels)

LIVE_FRAMES = []


def detect_objects_video(video_source, detected_classes: list = None):
    cap = cv2.VideoCapture(video_source)

    frame_number = 0
    skip_frames = 5

    labels = []
    if detected_classes is None:
        labels = list(range(len(model_labels)))
    elif isinstance(detected_classes, list):
        for item in detected_classes:
            if item in model_labels:
                labels.append(model_labels.index(item))
            else:
                print("Invalid Label = ", item)
    while True:
        success, frame = cap.read()
        # print(success)

        frame_number += 1

        if frame_number % skip_frames != 0:
            continue

        if not success:
            break

        # results = model.track(LIVE_FRAMES.pop(0),
        results = model(
            frame,
            # conf=0.5,
            # persist=True,
            classes=labels,
            # conf=0.3,
            # iou=0.5,
            augment=True,
            device=DEVICE,
            # verbose=True
        )

        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    video_source = 0
    # video_source = f"{VIDEO_PATH}/ch23_20240522000000.mp4"
    # video_source = f"{VIDEO_PATH}/ch18_20240522000000.mp4"
    # video_source = f"{VIDEO_PATH}/ch23_20240522000533.mp4"
    # video_source = f"{VIDEO_PATH}/ch11_20240522010854.mp4"
    video_source = 'ch03_20240522000000.mp4'
    # cap = cv2.VideoCapture(video_source)

    detect_objects_video(video_source, ["person"])
