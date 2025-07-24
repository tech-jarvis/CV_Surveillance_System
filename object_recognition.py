from ultralytics import YOLO
from config import DEVICE


class ObjectRecognition:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.model = YOLO(f"models/{project_name}.pt")

        self.history = {v: {} for k, v in self.model.names.items()}

    @property
    def names(self):
        return self.model.names

    def recognize(self, image):
        result = self.model.predict(image, verbose=False, device=DEVICE)[0]

        label = result.names[result.probs.top1]
        if label == "other":
            label = "Customer"
        return result, [
            result.names[result.probs.top1],
            result.probs.top1conf.cpu().numpy(),
        ]

    def insert(self, uid, label):
        if uid not in self.history[label]:
            self.history[label][uid] = 0
        self.history[label][uid] += 1


if __name__ == "__main__":
    import os

    # import cv2

    object_recog = ObjectRecognition("cashier_clf")

    dataset_path = "images/people"
    output_folder_path = "images/output"

    print(object_recog.names)

    for fname in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, fname)

        result, object_label, object_conf = object_recog.recognize(image_path)

        # print(result[0])
        # print(result[0].names)
        # print(result[0].probs)
        print(object_label, object_conf)

        output_path = os.path.join(output_folder_path, object_label, fname)

        os.rename(image_path, output_path)
        # cv2.imshow("image", result[0].orig_img)
        # if cv2.waitKey(0) == ord("q"):
        # break
        break
