from typing import Dict

from utils import point_is_in_polygon


class ObjectTimeData:
    def __init__(
        self,
        time,
        frame_number,
        x,
        y,
        labels,
        regions
    ) -> None:
        self.time = time
        self.frame_number = frame_number
        self.x = x
        self.y = y
        self.regions = regions
        self.labels = labels

class ObjectData:
    def __init__(self, uid):
        self.uid = uid
        self.time_data = {}
        self.video_direction = ""

        self.location_histogram = {}

    def update_location(self, x, y, labels, regions, time, frame_number):
        self.location_histogram[f"{x}_{y}"] = 1 + self.location_histogram.get(
            f"{x}_{y}", 0
        )

        self.time_data[f"{frame_number}"] = ObjectTimeData(
            time, frame_number, x, y, labels, regions
        )

    def motion_direction(self, region: str = "", min_dir_samples=5):
        if region == "":
            region_data = [d for _, d in self.time_data.items()]
        else:
            region_data = [d for _, d in self.time_data.items() if region in d.regions]

        number_of_samples = len(region_data)
        if number_of_samples > min_dir_samples:
            if number_of_samples > 100:
                start_id = number_of_samples - 100
                end_id = number_of_samples - 1

                start_data = region_data[start_id]
                end_data = region_data[end_id]
            else:
                start_id = 0
                end_id = number_of_samples - 1

                start_data = region_data[start_id]
                end_data = region_data[end_id]

            y_diff = end_data.y - start_data.y
            # x_diff = end_data.x - start_data.x

            self.video_direction = (
                "D" if y_diff > 0 else "U" if y_diff < 0 else ""
            )

        else:
            self.video_direction = ""

        return self.video_direction


class ObjectsHistoricData:
    def __init__(self, regions):
        self.objects_list: Dict[str, ObjectData] = {}

        self.unique_uid_count: Dict[str, int] = {}

        self.location_histogram: Dict[str, int] = {}

        self.regions = regions
        self.unique_regions_count: Dict[str, Dict[str, int]] = {}

        for region in self.regions:
            if region["name"] not in self.unique_regions_count:
                self.unique_regions_count[region["name"]] = {}

        self.unique_labels_count = {}

    def insert(self, uid, x, y, w, h, label: str, sub_labels, time, frame_number):
        try:
            self.unique_uid_count[uid] = 1 + self.unique_uid_count.get(uid, 0)

            loc = f"{x},{y}"

            self.location_histogram[loc] = 1 + self.location_histogram.get(loc, 0)

            region_list = self.get_objects_region(uid, x, y)

            if label not in self.unique_labels_count:
                self.unique_labels_count[label] = {}

            all_labels = [label] + sub_labels
            self.unique_labels_count[label][uid] = 1 + self.unique_labels_count[label].get(
                uid, 0
            )

            if uid not in self.objects_list:
                self.objects_list[uid] = ObjectData(uid)
            else:
                self.objects_list[uid].update_location(
                    x, y, all_labels, region_list, time, frame_number
                )

            return region_list
        except Exception as e:
            print("History Insert",e)
            return []

    def get_objects_region(self, uid, x, y):
        inside_polygons = []
        for polygon in self.regions:
            region_name = polygon["name"]
            if point_is_in_polygon([x, y], polygon["polygon"]):
                inside_polygons.append(region_name)

                if uid not in self.unique_regions_count[region_name]:
                    self.unique_regions_count[region_name][uid] = 0

                self.unique_regions_count[region_name][uid] += 1

        return inside_polygons

    def get_unique_ids(self, skip_count: int = 0):
        return len([k for k, v in self.unique_uid_count.items() if v > skip_count])

    def get_unique_sub_type(self, sub_type_name, skip_count: int = 0):
        if sub_type_name not in self.unique_regions_count:
            return 0
        return len(
            [
                k
                for k, v in self.unique_regions_count[sub_type_name].items()
                if v > skip_count
            ]
        )

    def get_unique_visitors(self, skip_count: int = 0):
        cashier_count = self.get_unique_sub_type("cashier", skip_count=skip_count)
        unique_ids = self.get_unique_ids(skip_count=skip_count)
        return unique_ids - cashier_count

    def get_all_region_id_count(self, skip_count: int = 0):
        output = {}
        for k, v in self.unique_regions_count.items():
            output[k] = len([k2 for k2, v2 in v.items() if v2 > skip_count])
        return output

    def get_all_label_id_count(self, skip_count: int = 0):
        output = {}
        for k, v in self.unique_labels_count.items():
            output[k] = len([k2 for k2, v2 in v.items() if v2 > skip_count])

        return output
