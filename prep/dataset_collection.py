import cv2
import os
import time
import sys
import argparse
import json
import random
import glob
from pathlib import Path
import shutil
import subprocess
import re
import threading
from imutils.video import FPS
import copy
import numpy as np


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from rtux_cnn.calibrator import Calibrator
from rtux_cnn.config import Config
from rtux_cnn.utils import get_adb_devices
from prep.dataset_json_writer import count_sequential_false_images, check_all_false
import prep.create_test

MIN_BRIGHTNESS = 5
MAX_BRIGHTNESS = 130


class TrueDataset:
    def __init__(self):
        self.brightness = 10
        self.contrast = 50
        self.saturation = 64
        self.gamma = 100
        self.exposure_absolute = 100
        self.cam_val_arg = None
        self.x_adj, self.y_adj, self.width, self.height = 0, 0, 0, 0
        self.dir = None
        self.save_dir = None
        self.landscape = False
        self.augmented = {}
        self.stopped = False
        self.count = 0
        self.status = None
        self.device = None

    def display_vid(self, app, action, tf, device):
        self.device = device
        print(f"Displaying video for {app}, {action}/vid.avi")
        # if self.landscape is True:
            # os.system(f'adb -s {device} shell settings put system accelerometer_rotation 0')
            # os.system(f'adb -s {device} shell settings put system user_rotation 1')
        # else:
            # os.system(f'adb -s {device} shell settings put system accelerometer_rotation 1')
        os.system(f"adb -s {device} shell am force-stop com.example.vlcrtux")
        os.system(f"adb -s {device} shell am start -n com.example.vlcrtux/.MainActivity -d file:///mnt/sdcard/temp_pic/{app}/{action}/{tf}/vid.avi")
        print(f"adb -s {device} shell am start -n com.example.vlcrtux/.MainActivity -d file:///mnt/sdcard/temp_pic/{app}/{action}/{tf}/vid.avi")
        time.sleep(1)

    def change_settings(self, cam, start_num, device):
        # os.system(f"adb -s {device} shell settings put system screen_brightness {MAX_BRIGHTNESS}")

        for _ in range(5):
            ret, frame = cam.read()

        ret, frame = cam.read()  # flush first image in buffer
        first = True
        pill2kill = threading.Event()
        checker_thread = threading.Thread(target=self.vid_end_checker, args=(pill2kill,))
        checker_thread.start() 
        total_count = start_num
        img_lst = []

        x_min, x_max = -1 * (self.width // 15), (self.width // 15)
        y_min, y_max = -1 * (self.height // 15), (self.height // 15)

        print(f"x range: {self.width // 15}, y range: {self.height // 15}")
        target_total = 200000
        print(f"Target Total Images: {target_total}")
        
        fps_lst = []
        while True:
            if "true" in self.save_dir and total_count > target_total:
                self.saving_thread = threading.Thread(target=self.photo_writer, args=(self.save_dir, copy.copy(img_lst))).start()
                print("[change_settings]: Finished collecting images")
                break
            elif "false" in self.save_dir and total_count > 250000:
                self.saving_thread = threading.Thread(target=self.photo_writer, args=(self.save_dir, copy.copy(img_lst))).start()
                print("[change_settings]: Too many images taken, stopping")
                break

            if first:
                print("[change_settings]: Waiting for video to be ready (first)")
                while self.status != "ready":
                    time.sleep(0.5)
                first = False

            os.system(f"adb -s {device} shell am broadcast -a com.example.vlcrtux.action.PLAY -n com.example.vlcrtux/.ControlReceiver")
            time.sleep(0.5)
            fps = FPS().start()
            print(f"Playing video")
            while True:
                if total_count % 120 == 0:  # flush images every 120 frames
                    self.saving_thread = threading.Thread(target=self.photo_writer, args=(self.save_dir, copy.copy(img_lst))).start()
                    img_lst[:total_count - 1] = [None] * (total_count - 1) 

                if self.status == "stopped":
                    print(f"Video has stopped: {total_count}")
                    self.status = None
                    break
                ret, frame = cam.read()
                if ret is True:
                    if np.isnan(frame).any() or frame.size==0:
                        continue
                    x_off, y_off = random.randint(x_min, x_max), random.randint(y_min, y_max)
                    frame = frame[(self.y_adj+y_off): (self.height+self.y_adj+y_off), (self.x_adj+x_off): (self.width + self.x_adj+x_off)]
                    frame = frame[:, :, 0]
                    br = frame.mean()
                    if br < 10:
                        print(f"Low brightness: {br}")
                        continue
                    img_lst.append(frame)
                    total_count += 1
                    fps.update()
            fps.stop()
            fps_lst.append(fps.fps())
            

        self.status = "done"    
        pill2kill.set()
        print(f"[change_settings]: Approx. FPS: {np.mean(fps_lst)}")
        print("\n")
        checker_thread.join()
        self.status = None


    def vid_end_checker(self, stop_event):
        vidinfo_loc = "/data/data/com.example.vlcrtux/files/videoplayback.txt"
        while not stop_event.wait(1):
            try:
                result = subprocess.run(['adb', '-s', self.device, 'shell', 'cat', vidinfo_loc], capture_output=True, text=True)
                output = result.stdout.strip().split('\n')
                if "stopped" in output[0]:
                    self.status = "stopped"
                elif "playing" in output[0]:
                    print("Video is playing")
                    self.status = "playing"
                elif "ready" in output[0]:
                    print("Video is ready")
                    self.status = "ready"
            except FileNotFoundError:
                print("Error: Could not get contents of videoplayback.txt")
                time.sleep(2)
        print("Exiting vid_end_checker")
        return
    
    def photo_writer(self, file_des, img_lst):
        for i in range(len(img_lst)):
            if img_lst[i] is not None and img_lst[i].size != 0:
                cv2.imwrite(f"{self.save_dir}/{i}.jpg", img_lst[i])
        self.saving_thread = None


MIN_CAMBR = 20
MAX_CAMBR = 40

if __name__ == '__main__':
    data = TrueDataset()
    num = 3  # Total number of pictures to take

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, default='dataset',
                        help='Directory to save the photos')
    parser.add_argument('-D', '--device', type=str, default=None,
                        help='serial number of the device to pass arguments to')
    parser.add_argument('-c', '--camera', type=int, default=0,
                        help='Specify which camera to take video from (num from /dev/video)')
    parser.add_argument('-C', '--calibrate', type=str, required=True,
                        help='Calibration file from br_calib.py')
    parser.add_argument('-g', '--game', type=str, required=True,
                        help='Which game to test')
    parser.add_argument('-a', '--append', action="store_true",
                        help='Whether to append images if there are existing photos ')
    parser.add_argument('-l', '--landscape', action='store_true',
                        help='Whether to collect data in a random fashion')
    parser.add_argument('-s', '--start', type=int, default=0,
                        help='Starting Index if to be specified')
    parser.add_argument('-e', '--end', type=int, default=None,
                        help='Ending Index if to be specified')
    parser.add_argument('-r', '--resume', action="store_true",
                        help="Resume from the number of saved images")
    args = vars(parser.parse_args())

    if args['device'] is None:
        print(f"Device serial not provided")

    abs_start = time.time()


    if not os.path.exists(args['directory']):
        print(f"Directory {args['directory']} does not exist")
        sys.exit(1)
    else:
        args['directory'] = os.path.join(os.getcwd(), args['directory'])
        print(f"Using directory {args['directory']}")

    """Setup device, camera, and calibration settings"""
    device = f"{get_adb_devices()}" if args['device'] is None else f"{args['device']}"
    print("Using device: " + device)

    """ Check if app is installed"""
    result = subprocess.run(["adb", "-s", device, "shell", "pm", "list", "packages", "com.example.vlcrtux"], stdout=subprocess.PIPE, text=True)
    if "com.example.vlcrtux" not in result.stdout:
        print("Please install com.example.vlcrtux on the device to continue")
        sys.exit(1)

    config = Config(cap_ind=args['camera'], device=device, cv_window=False, calib=args['calibrate'])

    cap = config.get_camera()
    #print webcam current fps
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
    cap.set(cv2.CAP_PROP_FPS, 120)

    os.system(f"./turnon.sh {device}")
    # max_br = config.get_max_br()
    MAX_BRIGHTNESS = 500

    # os.system(f'adb -s {device} shell settings put system screen_brightness {int(max_br)}')

    calibration = Calibrator()
    calib = config.get_calib()
    if calib is None:
        calibration.set_position()
        calibration.match_br()
        width, height, x_adj, y_adj = calibration.w, calibration.h, calibration.x_adj, calibration.y_adj
    else:
        with open(calib, 'r') as f:
            parsed_data = json.load(f)
            br = parsed_data[0]["BRIGHTNESS"]
            cr = parsed_data[0]["CONTRAST"]
            exp = parsed_data[0]["EXPOSURE"]
            calibration.set_cam_val(br, cr, exp)
            width, height = parsed_data[1]["width"], parsed_data[1]["height"]
            x_adj, y_adj = parsed_data[1]["x_adj"], parsed_data[1]["y_adj"]

    data.width, data.height, data.x_adj, data.y_adj = width, height, x_adj, y_adj
    print(f"data: {data.width}, {data.height}, {data.x_adj}, {data.y_adj}")

    landscape = ["csr", "asphalt", "GenshinImpact", "clashofclan", "marvel", "kabam", "CookieKingdom", "pvz"]

    "TRUE IMAGES"
    targ_app = args['game']
    if targ_app in landscape:
        data.landscape = True
    else:

        if args['landscape'] is True:
            data.landscape = True
        else:
            data.landscape = False

    """Create directories for each app process (action) and true/false images"""
    screen_path = f"{args['directory']}/{targ_app}/screen"
    indexes = [int(x) for x in os.listdir(screen_path)]
    indexes.sort()
    if args['start'] != 0:
        indexes = [x for x in indexes if x >= args['start']]
    if args['end'] is not None:
        indexes = [x for x in indexes if x <= args['end']]

    ind_lst = []
    for entry in indexes:
        full_path = os.path.join(screen_path, str(entry))
        if os.path.isdir(full_path):
            ind_lst.append(full_path)
    ind_lst.sort(key=lambda x: int(x.split('/')[-1]))
    print(ind_lst)

    data.dir = args['directory']
    existing_lst = [[1, 1] for _ in range(len(ind_lst))]
    
    for i in range(len(ind_lst)):
        ind = ind_lst[i].split('/')[-1]
        true_dir = f"{data.dir}/{targ_app}/{ind}/true"
        os.makedirs(true_dir, exist_ok=True)

        false_dir = f"{data.dir}/{targ_app}/{ind}/false"
        os.makedirs(false_dir, exist_ok=True)

        if args['resume'] is True:
            true_cnt, false_cnt = len(glob.glob(f"{true_dir}/*.jpg")), len(glob.glob(f"{false_dir}/*.jpg"))
            print(f"True: {true_cnt}, False: {false_cnt}")
            true_cnt, false_cnt = max(1, true_cnt), max(1, false_cnt)
            existing_lst[i] = [true_cnt, false_cnt]
        print(f"Creating sub directory {data.dir}/{targ_app}/{ind}/")

    print("existing_lst: ", existing_lst)

    for i in range(len(ind_lst)):
        ind = ind_lst[i].split('/')[-1]
        try:
            os.makedirs(f"{data.dir}/{targ_app}/{ind}/true", exist_ok=True)
            os.makedirs(f"{data.dir}/{targ_app}/{ind}/false", exist_ok=True)
            print(f"Creating sub directory {data.dir}/{targ_app}/{ind}/")
        except FileExistsError:
            print(f"Saving in an existing sub directory {data.dir}/{targ_app}/{ind}")


        for true_false in ['true', 'false']:
            existing_img = existing_lst[i][0] if true_false == 'true' else existing_lst[i][1]
            print(f"Resuming from {existing_img} images for {true_false}")
            # screen_images = glob.glob(f"{ind_lst[ind]}/{true_false}/*.jpg")
            screen_vid = f"{ind_lst[i]}/{true_false}.avi"
            data.save_dir = f"{data.dir}/{targ_app}/{ind}/{true_false}"  # Directory for each app process (action)
            vid = f"{ind_lst[i]}/{true_false}/vid.avi"
            thm = f"{ind_lst[i]}/{true_false}/thumbnail.bmp"
            # check if folder exists in adb
            os.system(f"adb -s {device} shell mkdir -p /mnt/sdcard/temp_pic/{targ_app}/{ind}/{true_false}")
            os.system(f"adb -s {device} push {vid} /mnt/sdcard/temp_pic/{targ_app}/{ind}/{true_false}/vid.avi") # video
            os.system(f"adb -s {device} push {thm} /mnt/sdcard/temp_pic/{targ_app}/{ind}/{true_false}/thumbnail.bmp") # thumbnail
            data.display_vid(targ_app, ind, true_false, device)
            data.change_settings(cap, existing_img, device)
            # os.system(f"adb -s {device} shell input keyevent 500 500")

    print("Finished collecting images")
    print(f"Full running time: {time.time() - abs_start}")
    os.system(f"adb -s {device} shell am force-stop com.example.vlcrtux")

    root_directory = args['directory']

    false_counts = check_all_false(root_directory)
    
    re_arg = fr"^{root_directory}/[^/]"
    false_counts = {dir: count for dir, count in false_counts.items() if re.match(re_arg, dir)
                    and "remove" not in dir}

    output_file = f"{root_directory}/dataset_false.json"  
    with open(output_file, "w") as f:
        json.dump(false_counts, f, indent=4)

    print(f"Data written to {output_file}")
    print("Creating test dataset")
    prep.create_test.main(parser.parse_args())
    # os.system(f"./dataset/create_test.sh {args['game']} {len(ind_lst) - 1}")


    # for i in range(0, len(ind_lst), 2):
    #     os.system(f"./trainer.sh dataset {args['game']} {i} {i+1} hdk.txt")