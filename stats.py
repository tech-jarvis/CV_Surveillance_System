import cv2
import pandas as pd

from config import DATA_PATH, VIDEO_PATH

VIDEO_PATH = '.'
DATA_PATH = 'data3'


def historic_stats(video_name: str):
    base_video_name = video_name.split(".")[0]

    print()
    frame_data = pd.read_csv(f"{DATA_PATH}/{base_video_name}_analytics.csv")
    frame_objects = pd.read_csv(f"{DATA_PATH}/{base_video_name}_frame_objects.csv")
    frame_region = pd.read_csv(f"{DATA_PATH}/{base_video_name}_frame_regions.csv")

    frame_data["frame_time"] = pd.to_datetime(frame_data["frame_time"])
    frame_data["hour"] = frame_data["frame_time"].dt.hour
    frame_data["time"] = frame_data["frame_time"].dt.time
    frame_data["secs"] = frame_data["frame_time"].dt.second

    # Define time intervals (30 minutes)
    # frame_data["time_interval_start"] = frame_data["frame_time"].dt.floor('30T')
    # frame_data["time_interval_end"] = frame_data["time_interval_start"] + pd.Timedelta(minutes=30)
    # time_range_people = frame_object_comb.groupby(["time_interval_start", "time_interval_end"])["user_id"].nunique().reset_index()
    frame_region_comb = frame_data.merge(frame_region, how="left")
    frame_object_comb = frame_data.merge(frame_objects, how="left")
    # replace user ids with less than 25 count user stays less than a second (model anomoly)
    frame_object_comb = pd.DataFrame(frame_object_comb)
    # Count the occurrences of each userid
    userid_counts = frame_object_comb["user_id"].value_counts()
    # Create a mask for values with count less than 25
    mask = frame_object_comb["user_id"].isin(userid_counts[userid_counts < 25].index)
    # Update user_id values where count is less than 25 to -1
    frame_object_comb.loc[mask, "user_id"] = -1
    time_interval_people = frame_object_comb.groupby("hour")["user_id"].nunique()
    time_interval_people = time_interval_people.reset_index()
    time_interval_cashier = (
        frame_object_comb[frame_object_comb["region"] == "cashier"]
        .groupby("hour")["user_id"]
        .nunique()
    )
    time_interval_cashier = time_interval_cashier.reset_index()
    time_interval_people["user_id"] = (
        time_interval_people["user_id"] - time_interval_cashier["user_id"]
    )

    # get peak hours
    all_hours = pd.DataFrame({"hour": range(0, 24)})

    # time_interval_cashier_secs = frame_object_comb.groupby("frame_time")["frame_number"].count()
    # time_interval_cashier_secs = time_interval_cashier_secs.reset_index()
    # time_interval_cashier_secs["hour"] = time_interval_cashier_secs["frame_time"].dt.hour
    # print(time_interval_cashier_secs.groupby("hour").count())

    # rechecking cashier based on frame
    time_interval_cashier_secs = (
        frame_object_comb[frame_object_comb["region"] == "cashier"]
        .groupby("frame_time")["frame_number"]
        .count()
    )
    # print(time_interval_cashier_secs)
    time_interval_cashier_secs = time_interval_cashier_secs.reset_index()
    # time_interval_cashier_secs["user_id"] = time_interval_cashier_secs["user_id"].apply(lambda x: 1 if x > 0 else 0)
    time_interval_cashier_secs["hour"] = time_interval_cashier_secs[
        "frame_time"
    ].dt.hour
    time_interval_cashier_secs = time_interval_cashier_secs.groupby("hour").count() / 60
    time_interval_cashier_secs.reset_index()
    time_interval_cashier_secs = all_hours.merge(
        time_interval_cashier_secs, on="hour", how="left"
    )
    time_interval_cashier_secs.fillna({"frame_number": 0}, inplace=True)
    # print(time_interval_cashier_secs["frame_number"].fillna(0,inplace=True))

    # time_interval_cashier_secs = frame_object_comb.groupby(["frame_time"])["frame_number"].count()
    # time_interval_cashier_secs = time_interval_cashier_secs.reset_index()
    # print(time_interval_cashier_secs)

    # time_interval_cashier_secs["hour"] = time_interval_cashier_secs["frame_time"].dt.hour

    # print(time_interval_cashier_secs.groupby("hour").count())

    # print(time_interval_cashier_secs[time_interval_cashier_secs["region"] == "cashier"].groupby("hour").count())
    # print(time_interval_cashier_secs[time_interval_cashier_secs["region"] == "customers"].groupby("hour").count())

    time_interval_cashier = all_hours.merge(
        time_interval_cashier, on="hour", how="left"
    )
    time_interval_cashier.fillna({"user_id": 0}, inplace=True)
    peak_hours_ = time_interval_people.sort_values("user_id", ascending=False)
    peak_hours = peak_hours_.head(1)
    off_peak_hour = peak_hours_.tail(1)
    time_interval_people = time_interval_people[["hour", "user_id"]]
    time_interval_people = all_hours.merge(time_interval_people, on="hour", how="left")
    time_interval_people.fillna({"user_id": 0}, inplace=True)
    grouped = frame_object_comb.groupby("hour")
    cashier_counts = grouped["region"].apply(lambda x: (x == "cashier").sum())
    # Find the hour with the minimum count of "cashier"
    hour_with_least_cashier = cashier_counts.idxmin()
    # # Count NaN values in the "region" column for each group
    # off_peak_hour = grouped['region'].apply(lambda x: x.isna().sum())
    # off_peak_hour = off_peak_hour.idxmax()
    customer_df = frame_object_comb[frame_objectq_comb["region"] == "customers"]
    # Step 2: Calculate the time difference for each customer
    time_spent = customer_df.groupby("user_id")["frame_time"].transform(
        lambda x: x.max() - x.min()
    )
    # Step 3: Compute the average time spent by customers
    average_time_spent = time_spent.mean().total_seconds()
    hours_ = int(average_time_spent // 3600)
    minutes_ = int((average_time_spent % 3600) // 60)
    seconds_ = int(average_time_spent % 60)
    average_time_spent = f"{hours_} hours, {minutes_} minutes, {seconds_} seconds"
    cashier_availability = time_interval_cashier_secs["frame_number"].to_list()
    return {
        "hourly_traffic": time_interval_people["user_id"].to_list(),
        "cashier_availability": [
            cashier_availability,
            [60 - x for x in cashier_availability if x != 0],
        ],
        "Peak_hour": int(peak_hours["hour"].values[0]),
        "Peak_hour_traffic": int(peak_hours["user_id"].values[0]),
        "Hour_with_least_cashier": int(hour_with_least_cashier),
        "Off_peak_hour": int(off_peak_hour["hour"].values[0]),
        "Off_peak_hour_traffic": int(off_peak_hour["user_id"].values[0]),
        "average_time_spent": average_time_spent,
    }
    # return {
    #     "hourly_traffic": time_interval_people,
    #     "Peak_hour": peak_hours,
    # }


def time_freq_heatmap(video_name: str, freq="15min"):
    base_video_name = video_name.split(".")[0]

    frame_data = pd.read_csv(f"{DATA_PATH}/{base_video_name}_analytics.csv")
    frame_objects = pd.read_csv(f"{DATA_PATH}/{base_video_name}_frame_objects.csv")

    frame_data["frame_time"] = pd.to_datetime(frame_data["frame_time"])

    frame_object_comb = frame_data.merge(frame_objects, how="left")

    visitors_data = frame_object_comb
    # visitors_data = frame_object_comb[frame_object_comb["region"] != "cashier"]
    visitors_data = visitors_data[~visitors_data["box_x"].isna()]

    visitors_data["box_x"] = visitors_data["box_x"].map(int)
    visitors_data["box_y"] = visitors_data["box_y"].map(int)

    visitors_data["box_x"] = visitors_data["box_x"].map(str)
    visitors_data["box_y"] = visitors_data["box_y"].map(str)

    visitors_data["location"] = visitors_data["box_x"] + "," + visitors_data["box_y"]

    visitors_data = visitors_data.groupby(pd.Grouper(key="frame_time", freq=freq))[
        "location"
    ].value_counts()
    visitors_data = visitors_data.reset_index()

    _output = {}
    for d in visitors_data.values:
        loc = d[1].split(",")

        if d[0] not in _output:
            _output[d[0]] = []
        _output[d[0]].append([[int(loc[0]), int(loc[1])], d[2]])

    ## create heatmap
    cap = cv2.VideoCapture(f"{VIDEO_PATH}/{video_name}")
    ret, frame = cap.read()

    cap.release()

    output_heatmaps = {}

    for data_time, points in _output.items():
        hm_image = generate_heatmap_points(frame, points)
        output_heatmaps[str(data_time)] = hm_image

    return {"heatmaps": output_heatmaps}


def time_span_heatmap(video_name: str, start_time, end_time):
    base_video_name = video_name.split(".")[0]

    frame_data = pd.read_csv(f"{DATA_PATH}/{base_video_name}_analytics.csv")
    frame_objects = pd.read_csv(f"{DATA_PATH}/{base_video_name}_frame_objects.csv")

    frame_data["frame_time"] = pd.to_datetime(frame_data["frame_time"])

    frame_object_comb = frame_data.merge(frame_objects, how="left")

    visitors_data = frame_object_comb
    visitors_data = visitors_data[
        (
            (visitors_data["frame_time"] >= pd.to_datetime(start_time))
            & (visitors_data["frame_time"] <= pd.to_datetime(end_time))
        )
    ]

    # visitors_data = frame_object_comb[frame_object_comb["region"] != "cashier"]
    visitors_data = visitors_data[~visitors_data["box_x"].isna()]

    visitors_data["box_x"] = visitors_data["box_x"].map(int)
    visitors_data["box_y"] = visitors_data["box_y"].map(int)

    visitors_data["box_x"] = visitors_data["box_x"].map(str)
    visitors_data["box_y"] = visitors_data["box_y"].map(str)

    visitors_data["location"] = visitors_data["box_x"] + "," + visitors_data["box_y"]

    visitors_data = visitors_data.groupby("location")["frame_number"].count()
    visitors_data = visitors_data.reset_index()

    points = []
    for d in visitors_data.values:
        loc = d[0].split(",")
        points.append([[int(loc[0]), int(loc[1])], d[1]])

    ## create heatmap
    cap = cv2.VideoCapture(f"{VIDEO_PATH}/{video_name}")
    ret, frame = cap.read()

    cap.release()

    output_heatmaps = {}

    hm_image = generate_heatmap_points(frame, points)

    return hm_image


if __name__ == "__main__":
    import json

    from generate_heatmap import generate_heatmap_points

    # result = historic_stats("ch16_20240522034137.mp4")
    result = historic_stats("ch03_20240522000001.mp4")
    print(result)
    print(json.dumps(result))

    # import os
    # print(os.listdir(VIDEO_PATH))

    # heatmap = time_span_heatmap("ch16_20240522034137.mp4",
    # "2024-05-22 03:00:00",
    # "2024-05-22 04:00:00")

    # cv2.imshow("image", heatmap)
    # cv2.waitKey(0)

    # freq = "1h"
    # output = time_freq_heatmap("ch16_20240522034137.mp4", freq=freq)

    # output_list = output["heatmaps"]

    # for key, heat in output_list.items():
    #     # print(points)
    #     cv2.imwrite(f"output/heatmap/{freq}_{key}.png", heat)
