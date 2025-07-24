import datetime
from typing import List

import numpy as np
import pandas as pd

from utils import insert_previous_hour_min


class LiveAnalytics:
    def __init__(self) -> None:
        self.df = pd.DataFrame(
            data={
                "frame_number": [],
                "frame_time": [],
                "uid": [],
                "x": [],
                "y": [],
                "w": [],
                "h": [],
                "region": [],
                "label": [],
                "video_direction": [],
            }
        )

    def add(
        self,
        frame_number,
        frame_time,
        uid,
        x,
        y,
        w,
        h,
        region: str = "",
        label: str = "",
        video_direction: str = "",
    ) -> None:
        self.df: pd.DataFrame = pd.concat(
            objs=[
                self.df,
                pd.DataFrame(
                    data={
                        "frame_number": [frame_number],
                        "frame_time": [frame_time],
                        "uid": [uid],
                        "x": [x],
                        "y": [y],
                        "w": [w],
                        "h": [h],
                        "region": [region],
                        "label": [label],
                        "video_direction": [video_direction],
                    }
                ),
            ],
            ignore_index=True,
        )

    def column_time_freq_occurance(
        self, column_name: str, freq: str, group_filter: str = None
    ):
        if len(self.df) > 0:
            temp = self.df[
                ((self.df[column_name] != group_filter) | (self.df[column_name] != ""))
            ]
            # temp = temp[temp[column_name] != ""]
            temp = (
                temp.groupby([pd.Grouper(key="frame_time", freq=freq), column_name])[
                    "frame_number"
                ]
                .count()
                .reset_index()
            )
            temp["frame_time"] = temp["frame_time"].apply(lambda x: str(x))
            return temp

    def column_time_freq_unqiue_uid_count(
        self, column_name: str, freq: str, skip_count: int = 1
    ):
        temp = self.df[
            self.df.groupby("uid")["uid"].transform("count") > skip_count
        ].copy()

        if len(temp) > 0:
            temp = (
                temp[temp[column_name] != ""]
                .groupby([pd.Grouper(key="frame_time", freq=freq), column_name])["uid"]
                .nunique()
                .reset_index()
            )
            temp["frame_time"] = temp["frame_time"].apply(lambda x: str(x))
            temp = temp.rename({"uid": "uid_count"})
            return temp

    def column_occurance(
        self, column_name: str, fps: int = 25, convert_to_time: bool = False
    ):
        if len(self.df) > 0:
            temp = (
                self.df[self.df[column_name] != ""]
                .groupby([column_name])["frame_number"]
                .count()
                .reset_index()
            )
            return self.convert_df_to_labels_value(temp, fps, convert_to_time)

    def column_unqiue_uid_count(self, column_name: str, skip_count: int = 1):
        temp = self.df[
            self.df.groupby("uid")["uid"].transform("count") > skip_count
        ].copy()

        if len(temp) > 0:
            temp = (
                temp[temp[column_name] != ""]
                .groupby([column_name])["uid"]
                .nunique()
                .reset_index()
            )
            return self.convert_df_to_labels_value(temp)

    def convert_df_to_labels_value(
        self, data_df, normalize=1, convert_to_time: bool = False
    ):
        output = {}
        for row in data_df.values:
            k, v = row
            output[k] = v / normalize
            if convert_to_time:
                output[k] = str(datetime.timedelta(seconds=int(output[k])))
        return output

    def columns_unqiue_uid_count(
        self, column_name_1: str, column_name_2: str, skip_count: int = 1
    ):
        temp = self.df[
            self.df.groupby("uid")["uid"].transform("count") > skip_count
        ].copy()

        if len(temp) > 0:
            temp = (
                temp[(temp[column_name_1] != "")]
                .groupby([column_name_1, column_name_2])["uid"]
                .nunique()
                .reset_index()
            )
            return self.convert_df_to_labels_value_v2(temp)


    def convert_df_to_labels_value_v2(
        self, data_df, normalize=1, convert_to_time: bool = False
    ):
        output = {}
        for row in data_df.values:
            k, k2, v = row
            if k2 == "":
                k2 = "None"
            if k not in output:
                output[k] = {}
            output[k][k2] = v / normalize
            if convert_to_time:
                output[k][k2] = str(datetime.timedelta(seconds=int(output[k])))
        return output

    def timely_count_day(
        self,
        current_time,
        group_by: str = "label",
        group_filter: str = None,
        freq: int = 15,
        skip_count: int = 25,
    ):
        filter_time = current_time.strftime("%Y-%m-%d")
        temp = self.column_time_freq_unqiue_uid_count(
            group_by, freq=f"{freq}min", skip_count=skip_count
        )
        if isinstance(temp, pd.DataFrame):
            temp = temp[
                pd.to_datetime(temp["frame_time"]) >= pd.to_datetime(filter_time)
            ]
            if group_filter is not None:
                temp = temp[temp[group_by] == group_filter]
            temp = temp.drop([group_by], axis=1)
            temp["frame_time"] = temp["frame_time"].apply(
                lambda x: pd.to_datetime(x).strftime("%H:%M")
            )

            miniutes = np.clip(current_time.minute + 15, 0, 59)
            time_index = insert_previous_hour_min(
                f"{current_time.hour}:{miniutes}",
                freq=freq,
            )
            time_index = pd.DataFrame(
                {"frame_time": time_index, "uid": [0 for _ in range(len(time_index))]}
            )
            time_index["frame_time"] = time_index["frame_time"].apply(
                lambda x: x.strftime("%H:%M")
            )

            if len(temp) > 0:
                temp = pd.concat([time_index, temp])
            else:
                temp = time_index
            temp = temp.reset_index()
            temp = temp.sort_values(by="frame_time")
        else:
            miniutes = np.clip(current_time.minute + 15, 0, 59)
            time_index = insert_previous_hour_min(
                f"{current_time.hour}:{miniutes}", freq=freq, exclude_last=False
            )
            time_index = pd.DataFrame(
                {"frame_time": time_index, "uid": [0 for _ in range(len(time_index))]}
            )
            time_index["frame_time"] = time_index["frame_time"].apply(
                lambda x: x.strftime("%H:%M")
            )

            temp = time_index.reset_index()

        return temp

    def timely_apearance(
        self,
        current_time,
        group_by: str = "label",
        group_filter: str = None,
        freq: int = 15,
        skip_count: int = 25,
    ):
        filter_time = current_time.strftime("%Y-%m-%d")
        temp = self.column_time_freq_occurance(
            group_by, freq=f"{freq}min", group_filter=group_filter
        )

        if isinstance(temp, pd.DataFrame) and len(temp) > 0:
            temp = temp[
                pd.to_datetime(temp["frame_time"]) >= pd.to_datetime(filter_time)
            ]
            if group_filter is not None:
                temp = temp[temp[group_by] == group_filter]
            temp = temp.drop([group_by], axis=1)
            temp["frame_time"] = temp["frame_time"].apply(
                lambda x: pd.to_datetime(x).strftime("%H:%M")
            )

            miniutes = np.clip(current_time.minute + 15, 0, 59)
            time_index = insert_previous_hour_min(
                f"{current_time.hour}:{miniutes}",
                freq=freq,
            )
            time_index = pd.DataFrame(
                {
                    "frame_time": time_index,
                    "frame_number": [0 for _ in range(len(time_index))],
                }
            )
            time_index["frame_time"] = time_index["frame_time"].apply(
                lambda x: x.strftime("%H:%M")
            )

            temp = pd.concat([time_index, temp])

            temp = temp.reset_index()
            temp = temp.sort_values(by="frame_time")
        else:
            miniutes = np.clip(current_time.minute + 15, 0, 59)
            time_index = insert_previous_hour_min(
                f"{current_time.hour}:{miniutes}", freq=freq, exclude_last=False
            )
            time_index = pd.DataFrame(
                {
                    "frame_time": time_index,
                    "frame_number": [0 for _ in range(len(time_index))],
                }
            )
            time_index["frame_time"] = time_index["frame_time"].apply(
                lambda x: x.strftime("%H:%M")
            )
            temp = time_index.reset_index()
        # try:
        #     temp = temp.drop(["index"], axis=1)
        # except IndexError:
        #     pass

        return temp

    def video_frame_time(
        self,
        current_time,
        freq: int = 15,
    ):
        filter_time = current_time.strftime("%Y-%m-%d")
        temp = self.df[pd.to_datetime(self.df["frame_time"]) > pd.to_datetime(filter_time)]

        if len(temp) == 0:
            return
        
        temp = temp.groupby([pd.Grouper(key="frame_time", freq=f"{freq}ME")])[
                "frame_number"
            ].count().reset_index()

        temp["frame_time"] = temp["frame_time"].apply(
            lambda x: pd.to_datetime(x).strftime("%H:%M")
        )

        return temp
