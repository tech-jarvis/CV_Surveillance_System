import os

from config import DATA_PATH, OUTPUT_PATH

if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
