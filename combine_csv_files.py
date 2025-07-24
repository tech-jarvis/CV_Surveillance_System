import os

import pandas as pd

data_folder = "data"

target_camera = "ch03"

global_csv = None
last_filename = ""
for filename in sorted(os.listdir(data_folder)):
    if not filename.startswith(target_camera):
        continue

    if len(filename) == len("ch03_202405220.csv"):
        continue
    print(filename)
    last_filename = filename
    data = pd.read_csv(os.path.join(data_folder, filename))
    if global_csv is None:
        global_csv = data
    else:
        fn = global_csv["frame_number"].iloc[-1]
        data["frame_number"] = data["frame_number"].apply(lambda x: x + fn)
    # print(fn)

    global_csv = pd.concat([global_csv, data])

    # print(global_csv.tail())

print(last_filename)
cam, date = last_filename.split(".")[0].split("_")

# global_csv = global_csv.sort_values(["frame_time", "frame_number"])

global_csv.to_csv(
    os.path.join(data_folder, target_camera + "_" + date[:8] + ".csv"), index=False
)
