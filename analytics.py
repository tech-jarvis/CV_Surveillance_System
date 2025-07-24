import datetime


class UniqueIDLabelCounter:
    def __init__(self,
                 label:str,
                 min_count:int=25) -> None:
        self.label = label
        self.unique_ids = []
        self.unique_ids_counts = {}
        self.count = 0
        self.min_count = min_count
        self.count_greater_than_min = 0

    def update(self, data):

        for obj in data:
            uid = obj["uid"]

            if self.label in obj["labels"]:
                if uid not in self.unique_ids:
                    self.unique_ids.append(uid)
                    self.unique_ids_counts[uid] = 1
                    self.count += 1
                elif uid in self.unique_ids_counts:
                    self.unique_ids_counts[uid] += 1

                    if self.unique_ids_counts[uid] == self.min_count:
                        self.count_greater_than_min += 1
                        del self.unique_ids_counts[uid]


class IntervalLabelCounter:
    def __init__(self,
                 label:str,
                 interval:int) -> None:

        self.interval = interval
        self.label = label

        self.last_time_index = None
        self.last_seen_count = 0
        self.last_not_seen_count = 0

        self.index_unit = "h" if (self.interval == 60) else "m" if (self.interval < 60) else ""

        self.data = {}

        self.setup_data()

    def setup_data(self):
        if self.index_unit == "h":
            for h in range(0, 24):
                current_time_index = f"{h:2}:00"
                self.data[current_time_index] = {}
                self.data[current_time_index]["available"] = 0
                self.data[current_time_index]["unavailable"] = 0

        elif self.index_unit == "m":
            current_time_index = f"{time.hour}:{time.minute}"
            for h in range(0, 24):
                for m in range(0, 60, self.interval):
                    current_time_index = f"{h:2}:{m:2}"
                    self.data[current_time_index] = {}
                    self.data[current_time_index]["available"] = 0
                    self.data[current_time_index]["unavailable"] = 0

    def time_to_index(self, time):
        if self.index_unit == "h":
            current_time_index = f"{time.hour}:00"
        elif self.index_unit == "m":
            current_time_index = f"{time.hour}:{time.minute}"
        return current_time_index

    def update(self, time, data):
        self.current_time_index = self.time_to_index(time)

        seen = False
        for obj in data:
            if self.label in obj["labels"]:
                seen = True
                break

        if seen:
            self.data[self.current_time_index]["available"] += 1
        else:
            self.data[self.current_time_index]["unavailable"] -= 1

    def seen(self, time):
        self.current_time_index = self.time_to_index(time)
        self.last_seen_count += 1

    def not_seen(self, time):
        self.current_time_index = self.time_to_index(time)
        self.last_not_seen_count += 1

    def get_count(self, normalize=1, format="2-axis"):
        output = {}

        if format == "2-axis":
            if "Xaxis" not in output:
                output["Xaxis"] = []
            if "Yaxis" not in output:
                output["Yaxis"] = []

        for k,v in self.data.items():
            if format == "2-axis":
                output["Xaxis"].append(v["available"]/normalize)
                output["Yaxis"].append(v["unavailable"]/normalize)
            else:
                output[k] = {"available": 0, "unavailable": 0}
                output[k]["available"] = output[k]["available"]/normalize
                output[k]["unavailable"] = output[k]["unavailable"]/normalize
        return output


class UniqueIDIntervalLabelCounter:
    def __init__(self,
                 label: str,
                 interval: int,
                 min_count: int = 25) -> None:
        self.label = label
        self.interval = interval
        self.min_count = min_count

        # Unique ID tracking
        self.unique_ids = []
        self.unique_ids_counts = {}
        self.count = 0
        self.count_greater_than_min = 0

        # Interval tracking
        self.index_unit = "h" if (self.interval == 60) else "m" if (self.interval < 60) else ""
        self.data = {}

        self.setup_data()

    def setup_data(self):
        if self.index_unit == "h":
            for h in range(0, 24):
                current_time_index = f"{h:02}:00"
                self.data[current_time_index] = {"unique_ids": {}}
        elif self.index_unit == "m":
            for h in range(0, 24):
                for m in range(0, 60, self.interval):
                    current_time_index = f"{h:02}:{m:02}"
                    self.data[current_time_index] = {"unique_ids": {}}

    def time_to_index(self, time):
        if self.index_unit == "h":
            current_time_index = f"{time.hour:02}:00"
        elif self.index_unit == "m":
            current_time_index = f"{time.hour:02}:{time.minute // self.interval * self.interval:02}"
        return current_time_index

    def update(self, time, data):
        self.current_time_index = self.time_to_index(time)

        # if self.current_time_index not in self.data:
        #     self.data[self.current_time_index] = {"unique_ids": {}}

        for obj in data:
            uid = obj["uid"]
            if self.label in obj["labels"]:
                # Record the unique ID for this interval and count its occurrences
                if uid not in self.data[self.current_time_index]["unique_ids"]:
                    self.data[self.current_time_index]["unique_ids"][uid] = 1
                else:
                    self.data[self.current_time_index]["unique_ids"][uid] += 1

    def get_count(self, normalize=1, format="2-axis"):
        # Normalize data if needed
        output = {}
        for k, v in self.data.items():
            output[k] = len([ki for ki, vi in v["unique_ids"].items() if vi > self.min_count])

        if format == "2-axis":
            output_temp = {"Xaxis": [], "Yaxis": []}
            for dt, value in output.items():
                output_temp["Xaxis"].append(dt)
                output_temp["Yaxis"].append(value)
            output = output_temp

        return output


if __name__ == "__main__":

    import datetime
    import time

    direction_counter = IntervalLabelCounter(
        label="cashier",
        interval=60
        )

    for _ in range(10):
        direction_counter.update(
            datetime.datetime.now(),
            data=[
                {
                    "labels": ["cashier"]
                }
                ]
        )
        print(direction_counter.get_count())
