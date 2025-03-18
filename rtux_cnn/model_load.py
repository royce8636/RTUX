import cv2
import tensorflow as tf
import json
import os
import time

from rtux_cnn.augmenter import Augmenter
from rtux_cnn.utils import CustomPrint, print, TensorRTModel
from rtux_cnn.config import Config

shape = 128


class ModelLoader:
    def __init__(self, queue=None, file=None):
        self.config = Config()
        self.device = self.config.device
        self.augmenter = Augmenter()
        if queue is not None:
            self.queue = queue
        elif file is not None:
            self.queue = self.load_file(file)
        self.dep_dict = {}

    def find_before(self):
        dep_dict = {}
        prev_value = None

        for item in self.queue:
            key = next(iter(item))
            value = item[key]
            if key == '0':
                if value not in dep_dict:
                    dep_dict[value] = []
                if prev_value is not None:
                    if prev_value not in dep_dict[value]:
                        dep_dict[value].append(prev_value)
                else:
                    dep_dict[value].append(self.device)
                prev_value = value
            elif self.device in value:
                prev_value = self.device
        print(dep_dict)
        return dep_dict

    def load_file(self, filename):
        queue = []
        with open(filename) as f:
            for line in f:
                line = line.strip()
                line = line.split('#', 1)[0].strip()
                if line:
                    lst = line.split(':')
                    queue.append({lst[0].strip(): lst[1].strip()})
        print(f"Loaded {len(queue)} commands from {filename}")
        return queue

    def load_model(self, main_path):
        models = {}
        for item in self.queue:
            key = next(iter(item))
            value = item[key]
            if key == '0':
                if value not in models:
                    models[value] = None
            if key == 'register':
                value = value.split(' ')[0]
                if value not in models:
                    models[value] = None
        for key in models:
            path = f"{main_path}/{key.split('_')[0]}/models/{key}"
            if not os.path.exists(path):
                print(f"Model {key} does not exist!\n"
                      f"Please check if the model {key} exists in the right path:\n"
                      f"{path}")
                exit(1)
            tensor_engine = TensorRTModel.from_file(path)
            models[key] = tensor_engine
            print(f"{key}: done!")
        return models

if __name__ == "__main__":
    queue = []
    file = "testing/supercell_one.txt"
    loader = ModelLoader(queue)
    loader.queue = loader.load_file(file)
    loader.dep_dict = loader.find_before()
    loader.train_dependency(loader.dep_dict, 'dataset')
