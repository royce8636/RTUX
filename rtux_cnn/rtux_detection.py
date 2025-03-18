import functools
import json
import subprocess
import threading
import copy
import cv2
from imutils.video import FPS
import numpy as np
import time
import tensorflow as tf
import queue
import yaml
import os

from rtux_cnn.calibrator import Calibrator
from rtux_cnn.utils import print
from rtux_cnn.config import Config

def apply_decorators(cls):
    cls.writer = cls.increment_counter(cls.writer)
    return cls

@apply_decorators
class RtuxDetect:

    count = 0

    def __init__(self, directory, exp_shape, notify_callback):
        self.config = Config()
        self.cam = self.config.get_camera()
        self.x_adj, self.y_adj, self.width, self.height = 0, 0, 0, 0
        self.done = False
        self.log_thread = None
        self.count = -1
        self.prev_app = None
        self.done = False
        self.shape = exp_shape
        self.ready = False
        self.detected = True
        self.kill = False
        self.saving_thread = None
        self.calib = self.config.get_calib()
        self.cv_window = self.config.get_cv_window()
        self.backup_img = []
        self.lock = threading.Lock()
        self.vid_queue = queue.Queue()
        self.vid_thread_stop = False
        self.unregister = False
        self.image_queue = []
        self.img2save = []
        self.notify_callback = notify_callback  
        self.stop_event = False
        self.ad_thread = None

        self.write_queue = queue.Queue()
        self.writer_thread = None
        self.file_lock = threading.Lock()
        self.yaml_lock = threading.Lock()

    def process_img(self, img):
        img = tf.image.resize(img, (self.shape[2], self.shape[1]))
        img = img[:, :, 0]
        img = tf.cast(img, dtype=tf.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
        return img

    def set_cam(self, device):
        calibration = Calibrator()
        if self.calib is None:
            calibration.set_position()
            calibration.match_br()
            self.width, self.height, self.x_adj, self.y_adj = calibration.phone_width, calibration.phone_height, calibration.x_adj, calibration.y_adj
        else:
            with open(self.calib, "r") as f:
                parsed_data = json.load(f)
                br = parsed_data[0]["BRIGHTNESS"]
                cr = parsed_data[0]["CONTRAST"]
                exp = parsed_data[0]["EXPOSURE"]
                calibration.set_cam_val(br, cr, exp, self.cam)
                self.width, self.height = parsed_data[1]["width"], parsed_data[1]["height"]
                self.x_adj, self.y_adj = parsed_data[1]["x_adj"], parsed_data[1]["y_adj"]

    def register_ad_watcher(self, model, action):
        reg_time = time.time()
        start_frame = self.count
        pressed = time.time()
        while self.unregister is False:
            img2check = self.image_queue[-1]
    
            pred_val = self.models[model].predict(img2check)
            if pred_val >= 0.95:
                if time.time() - pressed < 2:
                    continue
                pressed = time.time()
                print(f"Ad detected: {pred_val} | {self.end_time - self.start_time} | {self.count}")
                print(f" Doing action: {action}")
                if '/bin/bash' not in action:
                    action.insert(0, '/bin/bash')
                action.append(self.config.touchscreen)
                if self.config.device not in action:
                    action.append(self.config.device)
                print(action)
                self.shell_running = subprocess.Popen(action)
                self.shell_running.wait()

        print("Ending ad watcher")
        print(f"Ad watcher time: {time.time() - reg_time}")
        print(f"Ad Watcher Frames: {self.count - start_frame}")
        print(f"Ad Watcher FPS: {(self.count - start_frame) / (time.time() - reg_time)}")
        return

    def frame_grabber(self):
        print("Frame grabber started")
        fps = FPS().start()
        tpf = 1/120
        
        while self.cam.isOpened():
            if self.kill:
                self.kill = False
                break
            frame_grab_time = time.time()
            ret, frame = self.cam.read() 
            if ret is False:
                continue
            frame = frame[self.y_adj: self.height + self.y_adj,
                        self.x_adj: self.width + self.x_adj]
            self.img2save.append(frame[:, :, 0])
            if self.cv_window is not None and self.count % 2 == 0:
                    cv2.imshow(self.cv_window, frame)

            if self.count % 240 == 0:
                threading.Thread(target=self.saving, args=(copy.copy(self.img2save),)).start()
                if isinstance(self.config.log_all, int) and not isinstance(self.config.log_all, bool):
                    self.image_queue[:self.count - self.config.log_all] = [None] * (self.count - self.config.log_all)
                    self.img2save[:self.count - self.config.log_all] = [None] * (self.count - self.config.log_all)
                else:
                    self.image_queue[:self.count - 10] = [None] * (self.count - 10)
                    self.img2save[:self.count - 10] = [None] * (self.count - 10)
            frame = self.process_img(frame)
            self.image_queue.append(frame)
            self.count += 1
            fps.update()
            time_to_sleep = tpf - (time.time() - frame_grab_time + 0.0001)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
        fps.stop()
        print(f"Frame grabber Ended. \nFPS: {fps.fps()}")
        return
    
    def detect(self, thread_ind):
        local_count = 0
        last_frame = None
        last_state = False
        last_model = ""
        while not self.stop_event:
            start_time = time.time()
            with self.config.current_tasks.lock:
                event = self.config.current_tasks.queue[thread_ind]

            if last_frame == len(self.image_queue):
                time.sleep(0.001)
                continue
            last_frame = len(self.image_queue)

            if len(self.image_queue) == 0 or event is None:
                time.sleep(0.001)
                local_count = 0
                continue
            local_count += 1
            current_count = self.count
            frame = self.image_queue[current_count]
            
            boottime = time.clock_gettime(time.CLOCK_BOOTTIME)
            try:
                model_name = list(event.keys())[0]
            except AttributeError:
                continue
            pred_val = self.config.models[model_name].predict(frame)

            if model_name != last_model:
                changed_time = time.clock_gettime(time.CLOCK_BOOTTIME)
                last_state = self.config.current_tasks.get_state(thread_ind)
            last_model = model_name
                        
            if self.count % 120 == 0:
                print(f"Thread {thread_ind}: {model_name} {pred_val} ({current_count})")

            if pred_val >= 0.90 and last_state is False:
                print(f"Thread {thread_ind}: {model_name} detected: {pred_val} ({current_count})")
                threading.Thread(target=self.notify_callback, args=(thread_ind, model_name, current_count, boottime, changed_time, self.config._ind, True)).start()
                last_state = True
                print(f"Thread {thread_ind}: {model_name} detectction notify done")
            elif pred_val < 0.85 and last_state is True:
                print(f"Thread {thread_ind}: {model_name} not detected: {pred_val} ({current_count})")
                if thread_ind == 1:
                    self.config.current_tasks.update_to_false(thread_ind, True)
                else:
                    self.config.current_tasks.update_to_false(thread_ind, False)
                last_state = False

            if time.time() - start_time < 0.0001:
                time.sleep(0.0001 - (time.time() - start_time))
        print(f"Thread {thread_ind} ended")
    
    @staticmethod
    def increment_counter(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tag = f"{RtuxDetect.count}: {func.__name__}"
            RtuxDetect.count += 1
            kwargs['tag'] = tag
            return func(*args, **kwargs)

        return wrapper

    def read_yaml_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                file_contents = file.read()
                try:
                    data = yaml.safe_load(file_contents)
                    return data if data is not None else []
                except yaml.YAMLError as e:
                    print(f"Error reading YAML file: {e}")
                    print("Contents of the YAML file:")
                    print(file_contents)

                    with open(file_path + ".error", 'w') as error_file:
                        error_file.write(file_contents)
                    return []
        except FileNotFoundError:
            return []

    def start_threads(self):
        self.writer_thread = threading.Thread(target=self._writer_thread)
        self.writer_thread.start()

    def writer(self, model_name, img_lst, load_time, ind, level, status, elapsed, queue_ind, event_type=None, **kwargs):
        task = {
            'event_type': event_type,
            'model_name': model_name,
            'img_lst': img_lst,
            'load_time': load_time,
            'ind': ind,
            'level': level,
            'status': status,
            'elapsed': elapsed,
            'queue_ind': queue_ind,
            'kwargs': kwargs
        }
        self.write_queue.put(task)
        
    
    def _writer_thread(self):
        while True:
            try:
                task = self.write_queue.get(timeout=5)
                if task is None:
                    break
                self._process_write_task(task)
                self.write_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print("Exception in _writer_thread")


    def _process_write_task(self, task):
        model_name = task['model_name']
        ind = task['ind']
        level = task['level']
        status = task['status']
        load_time = task['load_time']
        elapsed = task['elapsed']
        img_lst = task['img_lst']
        queue_ind = task['queue_ind']
        tag = task['kwargs'].get('tag', 'default_tag')
        event_type = task['event_type'] if task['event_type'] is not None else tag.split(':')[1].strip()

        event = {
            'EventID': int(tag.split(':')[0]),
            'EventType': event_type,
            'Detail': model_name,
            'QueueIndex': queue_ind,
            'Level': level,
            'Status': status,
            'Metrics': {
                'LINUX_MONOTONIC': load_time,
                'FRAME': max(0, ind),
                'ELAPSED': elapsed
            }
        }

        with self.yaml_lock:
            existing_data = self._read_yaml_file(self.config.log_yaml)
            existing_data.append(event)
            self._write_yaml_file(self.config.log_yaml, existing_data)

        log_entry = f"{tag} | {model_name} | load_time: {str(load_time)}\n" \
                    f"{tag} | {model_name} | frame_count: {ind}\n"
        
        with self.file_lock:
            with open(self.config.log_file, 'a') as f:
                f.write(log_entry)

        if img_lst is None:
            return

        try:
            cv2.imwrite(f"{self.config.log_path}/{ind}_{model_name}_{status}.png", img_lst[ind])
        except IndexError:
            print(f"detected ind is {ind}, list is {len(img_lst)}")
        except cv2.error as e:
            os.system(f"cp {self.config.log_path}/all_frames/{ind}.jpg {self.config.log_path}/{ind}_{model_name}_{status}.png")

        try:
            cv2.imwrite(f"{self.config.rep_log_path}/{ind}_{model_name}_{status}.png", img_lst[ind])
        except IndexError:
            print(f"detected ind is {ind}, list is {len(img_lst)}")
        except cv2.error as e:
            os.system(f"cp {self.config.log_path}/all_frames/{ind}.jpg {self.config.rep_log_path}/{ind}_{model_name}_{status}.png")


        if isinstance(self.config.log_all, int):
            for i in range(ind - self.config.log_all, ind + 1):
                if img_lst[i] is not None:
                    cv2.imwrite(f"{self.config.log_path}/all_frames/{i}.jpg", img_lst[i])


    def _read_yaml_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file) or []
        except FileNotFoundError:
            return []

    def _write_yaml_file(self, file_path, data):
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file, sort_keys=False)

    def saving(self, img_lst):
        if isinstance(self.config.log_all, bool) and self.config.log_all is True:
            for i in range(len(img_lst)):
                if img_lst[i] is None:
                    continue
                if not cv2.imwrite(f"{self.config.log_path}/all_frames/{i}.jpg", img_lst[i]):
                    print(f"Error saving frame {i} please check your settings")
        return

    def save_vid(self):
        while True:
            if self.vid_thread_stop:
                break
            try:
                if self.lock.locked():
                    continue
                (vid_frame, timestamp) = self.vid_queue.get(timeout=1)
                if vid_frame is not None:
                    cv2.putText(vid_frame, str(timestamp), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.out.write(vid_frame)
            except (queue.Empty):
                break
            except Exception as e:
                print(e)
                pass
