import os

VIDEO_PATH = "/mnt"
OUTPUT_PATH = "output"
# DEVICE = 0
DEVICE = 'cpu'
# DATA_PATH = "data"
DATA_PATH = "data2"
DATA_PATH = "data3"

MODELS_PATH = "models"
DETECTION_MODEL_NAME = "yolov8m.pt"
DETECTION_MODEL_PATH = os.path.join(MODELS_PATH, DETECTION_MODEL_NAME)

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", 'Friday', "Saturday", "Sunday"]
