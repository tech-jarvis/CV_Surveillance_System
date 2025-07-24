from typing import List
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

# Load a pre-trained ResNet model for feature extraction
resnet = models.resnet50(pretrained=True)
resnet.eval()


# Define transformation for feature extraction
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def extract_features(person):
    img = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = preprocess(img_pil)
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        features = resnet(img_tensor).numpy()
    return features


def match_person(features_1, features_2):
    return np.linalg.norm(features_1 - features_2)

def create_images_similarity_matrix(frame1: List, frame2: List):
    mat = []

    for f1_obj in frame1:
        obj1_feature = extract_features(f1_obj)

        temp_mat = []
        for f2_obj in frame2:
            obj2_feature = extract_features(f2_obj)
            simi = match_person(obj1_feature, obj2_feature)
            temp_mat.append(simi)

        mat.append(temp_mat)

    return mat

if __name__ == "__main__":
    import os
    import tqdm

    cashier_path = "images/train/cashier"
    other_path = "images/train/other"

    cashiers_files = os.listdir("images/train/cashier")
    other_files = os.listdir("images/train/other")

    for fname in cashiers_files:
        image_1 = cv2.imread(os.path.join(cashier_path, fname))

        feature1 = extract_features(image_1)

        cv2.imshow("image 1", image_1)

        for fname_2 in tqdm.tqdm(cashiers_files):
            image_2 = cv2.imread(os.path.join(cashier_path, fname_2))

            feature2 = extract_features(image_2)

            simi = match_person(feature1, feature2)
            
            if simi > 80:
                print("\r", simi, end="")

                cv2.imshow("image 2", image_2)

                cv2.waitKey(1000)
