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
from utils import CustomPrint, TensorRTModel  # Assuming `utils` is available
import tensorflow as tf
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil


def load_model(model_name):
    key = model_name
    path = f"dataset/{key.split('_')[0]}/models/{key}"
    if not os.path.exists(path):
        print(f"Model {key} does not exist!\n"
              f"Please check if the model {key} exists in the right path:\n"
              f"{path}")
        exit(1)
    tensor_engine = TensorRTModel.from_file(path)
    print(f"{key}: done!")
    return tensor_engine

class FrameParser:
    def __init__(self, file_lst, log_dir, model_paths):
        self.model_paths = model_paths
        self.files = file_lst
        self.log_dir = log_dir
        self.starts = [self.get_starts(yaml_file) for yaml_file in self.files]
        self.starts.append(self.get_largest_int_file(log_dir))
        files = os.listdir(f"{log_dir}/all_log/all_frames")
        last_frame = int(files[-1].split('.')[0])
        self.all = {}
        self.all_andtimes = {}
        self.time_pairs, self.frame_pairs = self.parse_frames()
        # self.frame_pairs = [x.append(i) for i, x in enumerate(self.frame_pairs)]
        self.frame_pairs = [x + [i] for i, x in enumerate(self.frame_pairs)]
        print(self.frame_pairs)
        # self.calculate_times()

        for pair in self.frame_pairs:
            result = self.detect_frames(pair)
            print(result)

        # with ProcessPoolExecutor(max_workers=1) as executor:
        #     futures = [executor.submit(self.detect_frames, pair) for pair in self.frame_pairs]
        #     for future in as_completed(futures):
        #         result = future.result()
                
        print(result)

    def preprocess_image(self, file_path):
        img = cv2.imread(file_path)
        img = tf.image.resize(img, (128, 128))
        img = img[:, :, 0]
        img = tf.cast(img, dtype=tf.float32) / 255.0
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
        return img
        
    # def detect_frames(self, pair):
    #     models = {model_name: load_model(model_name) for model_name in self.model_paths}
    #     start_frame, end_frame, ind = pair
    #     file_format = f"{self.log_dir}/all_log/all_frames/"
    #     image_paths = [f"{file_format}/{frame}.jpg" for frame in range(start_frame, end_frame)]
        
    #     destination_dir = f"{self.log_dir}/{ind}/."
    #     os.makedirs(destination_dir, exist_ok=True)  # Create the destination directory if it doesn't exist
        
    #     i = 0
    #     for image_path in image_paths:
    #         i += 1
    #         if os.path.exists(image_path) and i % 10 == 0:  # Check if the file exists before copying
    #             shutil.copy(image_path, destination_dir)
        
    def detect_frames(self, pair):
        models = {model_name: load_model(model_name) for model_name in self.model_paths}
        start_frame, end_frame, ind = pair
        file_format = f"{self.log_dir}/all_log/all_frames/"
        image_paths = [f"{file_format}/{frame}.jpg" for frame in range(start_frame, end_frame)]
        
        images = [self.preprocess_image(path) for path in image_paths]
        
        detection_results = []

        model_ind = 0
        for i, image in enumerate(images):
            model = list(models.values())[model_ind]
            result = model.predict(image)
            if result > 0.9:
                print(f"Frame {start_frame + i} detected as {model_ind}")
                os.system(f"cp {file_format}/{start_frame + i}.jpg {self.log_dir}/{ind}/{model_ind}")
                detection_results.append((start_frame + i, model_ind))
                model_ind += 1
                if model_ind == len(models):
                    break
        return detection_results

        # for model_name, model in self.models.items():
        #     print(f"Running model {model_name}")
        #     for i, image in enumerate(images):
        #         result = model.predict(image)
        #         if result > 0.9:
        #             print(f"Frame {start_frame + i} detected as {model_name}")
        #             os.system(f"cp {file_format}/{start_frame + i}.jpg {self.log_dir}/{ind}/{model_name}")
        #             detection_results.append((start_frame + i, model_name))
                    
        # return detection_results
        
    def calculate_times(self):
        loadtimes = [end - start for start, end in self.time_pairs]
        print(f"Load times: {loadtimes}")
        print(f"Average load time: {statistics.mean(loadtimes)}")
        
    def parse_frames(self):
        file_format = f"{self.log_dir}/all_log/all_frames/"
        time_pairs = [(self.starts[i][1], self.starts[i+1][1]) for i in range(len(self.starts) - 1)]
        frame_pairs = [[self.starts[i][0], self.starts[i+1][0]] for i in range(len(self.starts) - 1)]
        return time_pairs, frame_pairs

    def get_starts(self, yaml_file):
        with open(yaml_file, 'r') as file:
            input_data = yaml.safe_load(file)
        return (input_data[0]['Metrics']['FRAME'], input_data[0]['Metrics']['LINUX_MONOTONIC'])
        
    def get_largest_int_file(self, log_dir):
        with open(self.files[-1], 'r') as file:
            input_data = yaml.safe_load(file)
        frame, monotonic = input_data[-1]['Metrics']['FRAME'], input_data[-1]['Metrics']['LINUX_MONOTONIC']
        directory = f"{log_dir}/all_log/all_frames"
        largest_int = -1
        largest_file = None
        for file_name in os.scandir(directory):
            if file_name.is_file() and file_name.name.endswith('.jpg'):
                try:
                    file_int = int(file_name.name.split('.')[0])
                    if file_int > largest_int:
                        largest_int = file_int
                        largest_file = file_name.name
                except ValueError:
                    continue
        frame_diff = largest_int - frame
        time = monotonic + frame_diff/120
        return (int(largest_int), time)


        
def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('-d', "--directory", type=str, help="directory to resolve")
    arg.add_argument('-q', "--queue", type=str, help="queue file directory")
    arg.add_argument('-t', "--threshold", type=int, default=3, help="minimum number of events for a valid run")
    arg.add_argument('-v', "--verbose", action='store_true', help="verbose mode")
    args = arg.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    with open(args.queue, 'r') as f:
        queue = f.readlines()
        
    detection_tasks = [x.split(':')[-1].rstrip().lstrip() for x in queue if '0:' in x] 
    
    entries = os.listdir(args.directory)
    dirs = sorted([d for d in entries if os.path.isdir(os.path.join(args.directory, d)) and is_integer(d)],
            key=lambda x: int(x))

    all_lmk = []
    all_times = []
    all_android_boottime = []
    abnormal_info = {}
    new_index = 0
    file_lst = []
    for d in dirs:
        # if os.path.exists(f"{args.directory}/{d}/{d}.yaml") and os.path.exists(f"{args.directory}/{d}/perfetto_trace_{d}"):
        if os.path.exists(f"{args.directory}/{d}/{d}.yaml"):
            # EventParser(f"{args.directory}/{d}/{d}.yaml", args.queue)
            # FrameParser(f"{args.directory}/{d}/{d}.yaml")
            file_lst.append(f"{args.directory}/{d}/{d}.yaml")
        else:
            print(f"Warning: Run {d} does not have the required files.")
            print(f"{args.directory}/{d}/{d}.yaml: {os.path.exists(f'{args.directory}/{d}/{d}.yaml')}, {os.path.exists(f'{args.directory}/{d}/perfetto_trace_{d}')}: {os.path.exists(f'{args.directory}/{d}/perfetto_trace_{d}')}")
        logging.debug("\n\n")
    
    print(f"Total runs: {len(file_lst)}")
    # FrameParser(file_lst, args.directory, models)
    FrameParser(file_lst, args.directory, detection_tasks)
