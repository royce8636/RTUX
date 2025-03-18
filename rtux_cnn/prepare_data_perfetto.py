import os
import json
import re
import argparse
import math
import sys
import yaml
from collections import defaultdict
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
import pprint
import itertools

class DataResolver:
    def __init__(self, directory):
        self.directory = directory
        self.diff = 0
        print(f"Data Resolver started: {self.directory}")
        self.read_sync(self.directory + "sync_diff.txt")
        print(f"Diff: {self.diff}")
        
        self.dirs = self.get_subdirectories()

        self.process_directories()
        print("Perfetto json files generated")

    def is_integer(self, s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def get_subdirectories(self):
        entries = os.listdir(self.directory)
        return sorted([d for d in entries if os.path.isdir(os.path.join(self.directory, d)) and self.is_integer(d)],
                      key=lambda x: int(x))

    def read_sync(self, file):
        with open(file, "r") as f:
            data = json.load(f)
        self.diff = data['time_diff']

    def process_directories(self):
        with ThreadPoolExecutor() as executor:
            executor.map(self.process_dir, [os.path.join(self.directory, d) for d in self.dirs])

    def process_dir(self, directory):
        directory = directory.rstrip('/')
        rep_index = os.path.basename(directory)

        yaml_file = f"{directory}/{rep_index}_simplified_updated.yaml"
        with open(yaml_file, "r") as f:
            yaml_data = yaml.safe_load(f)

        summary_dict = {}
        rtux_events = defaultdict(list)
        min_frame_count, max_frame_count = 0, 0
        for event in yaml_data:
            event_id = event['EventID']
            event_type = event['EventType']
            metrics = event['Metrics']
            queue_index = event['QueueIndex']

            if event_type == 'am_start':
                am_start = metrics['LINUX_MONOTONIC']
                summary_dict[f"{event_id}_am_command_{queue_index}"] = {
                    "CLOCK_REALTIME": float(am_start) - self.diff,
                    "Name": f"am command input ({event_id}:{queue_index})",
                    "frame_count": metrics['FRAME'],
                }
            elif event_type == 'writer':
                rtux_events[f"{event_id}_{queue_index}"].append({
                    "Name": f"{event['Detail'] if isinstance(event['Detail'], str) else (' ').join(event['Detail'])} ({event_id}:{queue_index})",
                    "Level": event['Level'],
                    "CLOCK_REALTIME": metrics['LINUX_MONOTONIC'] - self.diff,
                    "frame_count": metrics['FRAME'],
                    "fps": metrics.get('FPS', 0),
                    "total_time": metrics.get('LoadTime', 0),
                    "queue_index": queue_index
                })
            if metrics['FRAME'] > max_frame_count:
                max_frame_count = metrics['FRAME']
            if min_frame_count == 0 or metrics['FRAME'] < min_frame_count:
                min_frame_count = metrics['FRAME']

        for key, events in rtux_events.items():
            summary_dict[key] = events[-1]

        # pprint.pprint(summary_dict)
        log_dir = os.path.dirname(directory)
        # print(f"Min frame count: {min_frame_count}, Max frame count: {max_frame_count}")

        sorted_items = sorted(summary_dict.items(), key=lambda item: item[1].get('frame_count', float('inf')))
        pprint.pprint(sorted_items)
        rounded_info = {}
        # for (key, value), next_item in itertools.zip_longest(summary_dict.items(), list(summary_dict.items())[1:]):
        for (key, value), next_item in itertools.zip_longest(sorted_items, sorted_items[1:]):
            # if "total_time" not in value or "frame_count" not in value:
            #     continue
            # if value["total_time"] < 0:
                # continue
                
            start_time = value["CLOCK_REALTIME"]
            # fps = value["fps"] if value["fps"] > 0 else 120
            fps = 120
            frame_count = value["frame_count"]
            if next_item:
                next_key, next_value = next_item
                next_frame_count = next_value["frame_count"] if "frame_count" in next_value else frame_count + 1200
            else:
                next_frame_count = frame_count + 1200

            print(f"Start time: {start_time}, FPS: {fps}, End time: {start_time + (1/fps * frame_count)}")
            print(f"frame_count: {frame_count}, next_frame_count: {next_frame_count}")

            photo_time_diff = 1 / fps
            photo_time = start_time
            for i in range(frame_count, next_frame_count):
                rounded_photo_time = round((photo_time * 100) // 1 / 100, 2)
                photo_path = os.path.join(log_dir, "all_log/all_frames", f"{i}.jpg")
                rounded_info.setdefault(rounded_photo_time, {})[photo_path] = photo_time
                photo_time += photo_time_diff
        
        print(f"Finished processing photo info. Number of items in rounded_info: {len(rounded_info)}")

        sorted_rounded_info = dict(sorted(rounded_info.items()))
        keys = list(sorted_rounded_info.keys())
        min_key, max_key = keys[0], keys[-1]
        # print(f"Min key: {min_key}, Max key: {max_key}")
        min_step = 0.01

        incremental_dict = {}
        for k in sorted_rounded_info.keys():
            rounded_k = round(float(k), 2)
            if rounded_k not in incremental_dict:
                incremental_dict[rounded_k] = {}
            incremental_dict[rounded_k].update(sorted_rounded_info[k])

        hashed_json = {
            "summary": summary_dict,
            "rounded_photo_info": incremental_dict,
            "hash": {"min_key": float(min_key)*1e9, "min_step": float(min_step)*1e9, "formula": "(target_value - min_key) / min_step"}
        }

        with open(f"{directory}/all_summary_hashed.json", "w") as f:
            json.dump(hashed_json, f, indent=4)



if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('-d', "--directory", type=str, help="directory to resolve")
    args = arg.parse_args()
    dr = DataResolver(args.directory)