import yaml
import argparse
from collections import deque
import os
import sys
from collections import defaultdict
import json
import statistics
import csv
from pprint import pprint
import logging
import shutil

class EventParser:
    def __init__(self, yaml_file, queue_file=None, task_sqeuence=None):

        self.all = []
        if not queue_file and not task_sqeuence:
            print("Queue file or task sequence must be provided.")
            return
        if task_sqeuence:
            self.task_sequence = task_sqeuence
        elif queue_file:
            self.task_sequence = self.parse_task_sequence(queue_file)

        with open(yaml_file, 'r') as file:
            input_data = yaml.safe_load(file)

        input_data = self.remove_duplicate_logs(input_data)

        simplified_log = self.simplify_log(input_data)

        output_path = yaml_file.replace('.yaml', '_simplified.yaml')
        with open(output_path, 'w') as file:
            yaml.dump(simplified_log, file, default_flow_style=False)
        
        logging.info("Simplified log has been written to 'simplified_log.yaml'")
        logging.info(f"Number of events in simplified log: {len(simplified_log)}")

    def parse_task_sequence(self, queue_file):
        sequence = []
        line_number = 0
        with open(queue_file, 'r') as file:
            for line in file:
                parts = line.strip().split(': ', 1)
                if len(parts) == 2:
                    event_type, task = parts
                    if "echo" not in task and "force-stop" not in task:
                        sequence.append((line_number, event_type, task))
                    if event_type == '0' or 'am_start' in task:
                        self.all.append({f"{task}": []})
                    else:
                        self.all.append(None)
                    line_number += 1
        return sequence


    def remove_duplicate_logs(self, input_data):
        result = []
        current_block = []
        event_dict = defaultdict(lambda: float('inf'))

        def process_block():
            nonlocal current_block, event_dict
            for event in current_block:
                key = (event['EventType'], event['Detail'])
                if event['Metrics']['LINUX_MONOTONIC'] < event_dict[key]:
                    event_dict[key] = event['Metrics']['LINUX_MONOTONIC']

            for event in current_block:
                key = (event['EventType'], event['Detail'])
                if event['Metrics']['LINUX_MONOTONIC'] == event_dict[key]:
                    result.append(event)

            current_block = []
            event_dict.clear()

        for event in input_data:
            if event['Status'] in ['Touch', 'AppStart']:
                if current_block:
                    process_block()
                result.append(event)
            else:
                current_block.append(event)

        if current_block:
            process_block()

        return result

    def simplify_log(self, input_yaml):
        simplified_events = []
        task_queue = deque(self.task_sequence)
        frame_events = {}
        
        input_yaml.sort(key=lambda event: event['Metrics']['FRAME'])
        
        for event in input_yaml:
            frame = event['Metrics']['FRAME']
            if frame not in frame_events:
                frame_events[frame] = []
            frame_events[frame].append(event)

        for frame in reversed(sorted(frame_events.keys())):
            events = frame_events[frame]
            events.sort(key=lambda e: (e['QueueIndex']))

            for event in reversed(events):
                event_type, detail, queue_index = event['EventType'], event['Detail'], event['QueueIndex']
                cur_task_line, cur_task_type, cur_task_detail = task_queue[-1]

                next_task_line, next_task_type, next_task_detail = task_queue[-2] if len(task_queue) > 1 else (None, None, None)

                logging.debug(f"Cur Task: {cur_task_detail} ({cur_task_line})")
                logging.debug(f"Cur Event: {detail} ({event['EventID']})")
                qts = 3
                if event_type == 'am_start':
                    task = detail.split('/')[0]
                    if task_queue and cur_task_detail.split()[-1].startswith(task) and queue_index in range(cur_task_line - qts, cur_task_line + qts+1):
                        logging.debug(f"ADDING ({event['EventID']}) (task {cur_task_detail})")
                        event['QueueIndex'] = cur_task_line
                        simplified_events.insert(0, event)
                        task_queue.pop()
                    elif len(task_queue) > 1 and next_task_detail.split()[-1].startswith(task) and queue_index in range(cur_task_line - qts, cur_task_line + qts+1):
                        logging.debug(f"ADDING ({event['EventID']}) (task {next_task_detail})")
                        event['QueueIndex'] = next_task_line
                        simplified_events.insert(0, event)
                        task_queue.pop()
                        task_queue.pop()
                elif event_type == 'writer':
                    if isinstance(detail, list) and len(detail) > 3 and detail[0] == '/bin/bash':
                        detail = ' '.join(detail)
                        if task_queue and cur_task_type == '1' and cur_task_detail in detail:
                            logging.debug(f"ADDING ({event['EventID']}) (task {cur_task_detail})")
                            event['QueueIndex'] = cur_task_line
                            simplified_events.insert(0, event)
                            task_queue.pop()
                    elif isinstance(detail, str):
                        if task_queue and cur_task_type == '0':
                            if cur_task_detail in detail and queue_index in range(cur_task_line - qts, cur_task_line + qts+1):
                                logging.debug(f"ADDING ({event['EventID']}) (task {cur_task_detail})")
                                event['QueueIndex'] = cur_task_line
                                simplified_events.insert(0, event)
                                task_queue.pop()
                            elif len(task_queue) > 1 and next_task_detail in detail and queue_index in range(cur_task_line - qts, cur_task_line + qts+1):
                                logging.debug(f"ADDING ({event['EventID']}) (task {next_task_detail})")
                                event['QueueIndex'] = next_task_line
                                simplified_events.insert(0, event)
                                task_queue.pop()
                                task_queue.pop()
                logging.debug('\n')
                if not task_queue:
                    break

            if not task_queue:
                break
        return simplified_events


class DataParser():
    def __init__(self, directory, diff):
        self.file_path = f"{directory}"
        self.diff = diff
        self.all = {}
        self.all_andtimes = {}
        events = self.load_yaml(self.file_path)
        load_times = self.calculate_load_times(events)
        fps_values = self.calculate_fps(events)
        
        self.update_events(events, load_times, fps_values)
        
        self.save_yaml(f"{self.file_path}", events)

    def load_yaml(self, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data

    def save_yaml(self, file_path, data):
        file_path = file_path.replace('.yaml', '_updated.yaml')
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file, sort_keys=False)

    def divide_events(self, file_path):
        events = self.load_yaml(file_path)
        info_events = []
        other_events = []
        
        for event in events:
            if event.get('Level') == 'INFO':
                info_events.append(event)
            else:
                other_events.append(event)
        
        self.save_yaml(self.info_file_path, info_events)
        self.save_yaml(self.others_file_path, other_events)

    def calculate_load_times(self, events):
        load_times = []
        previous_time = None
        
        for event in events:
            if previous_time is not None:
                load_time = event['Metrics']['LINUX_MONOTONIC'] - previous_time
                load_times.append({
                    'EventID': event['EventID'],
                    'LoadTime': load_time,
                })
                if isinstance(event['Detail'], str):
                    key = f"{event['QueueIndex']}_{event['Detail']}"
                    if key not in self.all:
                        self.all[key] = []
                    self.all[key].append(load_time)
            else:
                load_times.append({
                    'EventID': event['EventID'],
                    'LoadTime': 0,
                })
                if isinstance(event['Detail'], str):
                    key = f"{event['QueueIndex']}_{event['Detail']}"
                    if key not in self.all:
                        self.all[key] = []
                    self.all[key].append(0)
            previous_time = event['Metrics']['LINUX_MONOTONIC']
        
        return load_times
    
    def calculate_fps(self, events):
        fps_values = []
        for i, event in enumerate(events):
            if i == 0:
                continue
            prev_event = events[i - 1]
            elapsed_time = event['Metrics']['LINUX_MONOTONIC'] - prev_event['Metrics']['LINUX_MONOTONIC']
            frame_difference = event['Metrics']['FRAME'] - prev_event['Metrics']['FRAME']
            try:
                fps = frame_difference / elapsed_time
            except ZeroDivisionError:
                fps = 120
            fps_values.append({
                'EventID': event['EventID'],
                'FPS': fps
            })
        return fps_values

    def update_events(self, events, load_times, fps_values):
        load_time_dict = {item['EventID']: item['LoadTime'] for item in load_times}
        fps_dict = {item['EventID']: item['FPS'] for item in fps_values}
        
        for event in events:
            event_id = event['EventID']
            if event_id in load_time_dict:
                event['Metrics']['LoadTime'] = load_time_dict[event_id]
            if event_id in fps_dict:
                event['Metrics']['FPS'] = fps_dict[event_id]
            event['Metrics']['Android_BootTime'] = event['Metrics']['LINUX_MONOTONIC'] - self.diff
            key = f"{event['QueueIndex']}_{event['Detail']}"
            self.all_andtimes[key] = event['Metrics']['Android_BootTime']


def process_directory(directory, queue_file, diff):
    all_times = []
    all_elapsed_times = []
    all_boot_times = []
    stages = set()

    for d in sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and d.isdigit()]):
        yaml_file = f"{directory}/{d}/{d}.yaml"
        if os.path.exists(yaml_file) and os.path.exists(f"{directory}/{d}/perfetto_trace_{d}"):
            EventParser(yaml_file, queue_file)
            dp = DataParser(f"{directory}/{d}/{d}_simplified.yaml", diff)
            
            run_times = {}
            run_elapsed = {}
            run_boot = {}
            
            for event in dp.events:
                if isinstance(event['Detail'], str):
                    key = event['Detail']
                    stages.add(key)
                    run_times[key] = event['Metrics']['LoadTime']
                    run_elapsed[key] = event['Metrics'].get('ELAPSED', 0)
                    run_boot[key] = event['Metrics']['Android_BootTime']
            
            all_times.append(run_times)
            all_elapsed_times.append(run_elapsed)
            all_boot_times.append(run_boot)

    stages = sorted(list(stages))

    csv_file = f"{directory}/stages_android_boottime.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(['Run'] + stages)
        
        for i, run_data in enumerate(all_boot_times, 1):
            row = [i]
            for stage in stages:
                row.append(run_data.get(stage, 'N/A'))
            writer.writerow(row)

    print(f"CSV file has been created at: {csv_file}")

    merged_dict = defaultdict(list)
    for d in all_times:
        for key, value in d.items():
            merged_dict[key].append(value)

    with open(f"{directory}/all_times.json", 'w') as file:
        json.dump(dict(merged_dict), file, indent=4)

    summary = {key: {
        'min': min(filter(None, values)),
        'max': max(filter(None, values)),
        'avg': statistics.mean(filter(None, values)),
        'median': statistics.median(filter(None, values))
    } for key, values in merged_dict.items() if any(values)}

    with open(f"{directory}/summary.json", 'w') as file:
        json.dump(summary, file, indent=4)

    for key, values in merged_dict.items():
        avg = summary[key]['avg']
        outliers = [(i, x) for i, x in enumerate(values) if x and x > 2 * avg]
        if outliers:
            print(f"Outliers for {key}: {outliers}")

def user_confirm(prompt="Do you want to continue? [y/n]: "):
    while True:
        response = input(prompt).lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False

def copy_filtered_files(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    for root, dirs, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        dst_path = os.path.join(dst, rel_path)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        for file in files:
            if not (file.endswith('.png') or file.endswith('.jpg') or file.startswith('perfetto_trace_')):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_path, file)
                shutil.copy2(src_file, dst_file)  # copy2 preserves metadata
                # print(f"Copied {src_file} to {dst_file}")

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('-d', "--directory", required=True, type=str, help="directory to resolve")
    arg.add_argument('-q', "--queue", required=True,type=str, help="queue file directory")
    arg.add_argument('-t', "--threshold", type=int, default=3, help="minimum number of events for a valid run")
    arg.add_argument('-v', "--verbose", action='store_true', help="verbose mode")
    arg.add_argument('-b', "--backup", action='store_true', help="Create backup with barebone files")
    args = arg.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not os.path.exists(args.queue):
        print("Queue file does not exist.")
        sys.exit(1)

    with open(args.directory + "sync_diff.txt", 'r') as f:
        rdata = json.load(f)
    diff = rdata['time_diff']
    print(f"Sync diff: {diff}")

    entries = os.listdir(args.directory)
    dirs = sorted([d for d in entries if os.path.isdir(os.path.join(args.directory, d)) and is_integer(d)],
            key=lambda x: int(x))

    if args.backup:
        copy_filtered_files(args.directory, f"{args.directory.strip('/')}_original")

    all_times = []
    all_android_boottime = []
    abnormal_info = {}
    new_index = 0
    for d in dirs:
        if "abnormal_" in d:
            abnormal_info[d] = []
            continue
        files = os.listdir(f"{args.directory}/{d}")
        if "abnormal.txt" in files:
            with open(f"{args.directory}/{d}/abnormal.txt", 'r') as f:
                data = f.readlines()
            abnormal_info[d] = data
            os.system(f"mv {args.directory}/{d} {args.directory}/abnormal_{d}")
            continue
        else:
            org_ind = int(d)
            if org_ind != new_index:
                if os.path.exists(f"{args.directory}/{new_index}"):
                    print(f"Warning: Directory {new_index} already exists.")
                    if not user_confirm():
                        print("Exiting...")
                        sys.exit(1)
                if os.path.exists(f"{args.directory}/{d}/{d}.yaml"):
                    os.rename(f"{args.directory}/{d}/{d}.yaml", f"{args.directory}/{d}/{new_index}.yaml")
                if os.path.exists(f"{args.directory}/{d}/perfetto_trace_{d}"):
                    os.rename(f"{args.directory}/{d}/perfetto_trace_{d}", f"{args.directory}/{d}/perfetto_trace_{new_index}")
                if os.path.exists(f"{args.directory}/{d}"):
                    os.rename(f"{args.directory}/{d}", f"{args.directory}/{new_index}")

                d = str(new_index)
            new_index += 1
        logging.debug(f"Processing directory {d}")

        # if os.path.exists(f"{args.directory}/{d}/{d}.yaml") and os.path.exists(f"{args.directory}/{d}/perfetto_trace_{d}"):
        if os.path.exists(f"{args.directory}/{d}/{d}.yaml"):
            EventParser(f"{args.directory}/{d}/{d}.yaml", args.queue)
            DP = DataParser(f"{args.directory}/{d}/{d}_simplified.yaml", diff)
            # print(DP.all)
            if len(DP.all) < args.threshold:
                print(f"Warning: Run {d} has fewer than {args.threshold} events. Setting all values to null.")
                all_times.append({})
                all_android_boottime.append({})
            else:
                all_times.append(DP.all)
                all_android_boottime.append(DP.all_andtimes)
        else:
            print(f"Warning: Run {d} does not have the required files.")
            print(f"{args.directory}/{d}/{d}.yaml: {os.path.exists(f'{args.directory}/{d}/{d}.yaml')}, {os.path.exists(f'{args.directory}/{d}/perfetto_trace_{d}')}: {os.path.exists(f'{args.directory}/{d}/perfetto_trace_{d}')}")
        logging.debug("\n\n")
        
    print(len(abnormal_info), abnormal_info.keys())

    all_keys = set()
    i = 0
    for d in all_times:
        all_keys.update(d.keys())
        i += 1
    all_keys_sorted = sorted(list(all_keys), key=lambda x: int(x.split('_')[0]))

    print(all_keys_sorted)

    csv_file = f"{args.directory}/all_boottime.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Run'] + all_keys_sorted)
        for i, d in enumerate(all_android_boottime):
            row = [i]
            for key in all_keys_sorted:
                row.append(d.get(key, 'N/A'))
            writer.writerow(row)

    csv_file = f"{args.directory}/all_loadtimes.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Run'] + all_keys_sorted)
        for i, d in enumerate(all_times):
            row = [i]
            for key in all_keys_sorted:
                row.append(d.get(key, 'N/A'))
            writer.writerow(row)

    merged_dict = defaultdict(list)
    for d in all_times:
        if not d:
            continue
        for key in all_keys:
            if key in d:
                merged_dict[key].extend(d[key])
            else:
                merged_dict[key].extend([None] * len(d[next(iter(d))]))

    merged_dict = dict(merged_dict)

    with open(f"{args.directory}/all_times.json", 'w') as file:
        json.dump(merged_dict, file, indent=4)


    summary = {}
    for key, value in merged_dict.items():
        valid_values = [v for v in value if v is not None]
        if valid_values:
            summary[key] = {
                'min': min(valid_values),
                'max': max(valid_values),
                'avg': statistics.mean(valid_values),
                'median': statistics.median(valid_values)
            }
        else:
            summary[key] = {
                'min': None,
                'max': None,
                'avg': None,
                'median': None
            }
    
    with open(f"{args.directory}/summary.json", 'w') as file:
        json.dump(summary, file, indent=4)
    print(f"Summary has been written to '{args.directory}/summary.json'")