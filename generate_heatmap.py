import cv2
import numpy as np


def generate_heatmap_points(image, points=[]):
    # Create a heatmap
    heatmap = np.zeros_like(image, dtype=np.float32)

    min_samples = 16
    max_samples = min_samples*3

    for point_data in points:
        point = point_data[0]
        size = point_data[1]
        if size > max_samples:
            size = max_samples
        if size <= min_samples:
            continue
        cv2.circle(
            heatmap, point, size // min_samples, (255, 255, 255), -1
        )  # White circle at each point

    heatmap = cv2.GaussianBlur(
        heatmap, (51, 51), 10
    )  # Gaussian blur to spread intensity
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    heatmap = cv2.applyColorMap(
        heatmap.astype(np.uint8), cv2.COLORMAP_JET
    )  # Apply colormap

    image = cv2.addWeighted(image, 1, heatmap, 0.5, 0)

    image = image.astype("uint8")
    return image


def generate_heatmap(image_path: str):
    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_component = lab[:, :, 1]
    th = cv2.threshold(a_component, 140, 255, cv2.THRESH_BINARY)[1]

    blur = cv2.GaussianBlur(th, (13, 13), 11)

    heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)

    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

    cv2.imshow("image", super_imposed_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    points = [
        ((300, 500), 50),
        ((500, 900), 70),
        ((900, 600), 30),
        ((1200, 600), 30),
        ((1400, 900), 40),
        ((1500, 800), 30),
        ((1550, 650), 60),
        ((1650, 550), 20),
    ]

    image = cv2.imread("image.png")
    generate_heatmap_points(image, points)
