import argparse
import subprocess
import shlex
import os
import cv2
from PIL import Image, ImageDraw
import numpy as np
import time
import json

try:
    from rtux_cnn.utils import CustomPrint, print
    from rtux_cnn.config import Config
except ImportError:
    from utils import CustomPrint, print
    from config import Config


class Calibrator:
    def __init__(self):
        self.config = Config()
        self.MIN_BR, self.MAX_BR = -64, 64
        self.MIN_CR, self.MAX_CR = 0, 64
        self.wedge_steps = 10
        self.cap = self.config.get_camera()
        self.cv_window = self.config.get_cv_window()
        # max_brightness = int(os.popen(f'adb {self.DEVICE_SERIAL} shell cat '
        #                               f'/sys/class/backlight/panel0-backlight/max_brightness').read().strip())
        # max_br = self.config.get_max_br()
        # print(max_br)
        # os.system(f'adb -s {self.config.device} shell settings put system screen_brightness {(max_br // 3) * 2}')

        # Get the device's screen size
        size = os.popen(
            f'adb -s {self.config.device} shell wm size | grep "Physical" | grep -o "[0-9]*x[0-9]\+" | head -n '
            f'1').read().strip()
        self.phone_width = int(size.split('x')[0])
        self.phone_height = int(size.split('x')[1])

        self.x_adj, self.y_adj, self.w, self.h = 0, 0, 0, 0

    def show_image_photos(self, image):

        os.system(f"adb -s {self.config.device} shell am start -n com.example.vlcrtux/.MainActivity")
        if "white" in image:
            print(f"adb -s {self.config.device} shell am broadcast -a com.example.vlcrtux.action.WHITE")
            os.system(f"adb -s {self.config.device} shell am broadcast -a com.example.vlcrtux.action.WHITE")
        elif "step" in image:
            print(f"adb -s {self.config.device} shell am broadcast -a com.example.vlcrtux.action.STEP")
            os.system(f"adb -s {self.config.device} shell am broadcast -a com.example.vlcrtux.action.STEP")

    def create_step_wedge(self, size):
        size = (self.phone_width, self.phone_height)
        image = Image.new("L", size)  # Create a new grayscale image
        draw = ImageDraw.Draw(image)

        step_size = size[1] // self.wedge_steps  # Height of each step

        for i in range(self.wedge_steps):
            y1 = i * step_size
            y2 = (i + 1) * step_size
            draw.rectangle([(0, y1), (size[0], y2)], fill=i * (256 // (self.wedge_steps - 1)))

        image.save("step_wedge.png")
        subprocess.run(shlex.split(f'adb -s {self.config.device} push step_wedge.png /sdcard/step_wedge.png'))

    def set_cam_val(self, BR, CR, EXP, cap=None):
        if cap is None:
            cap = self.cap
        cap.set(cv2.CAP_PROP_BRIGHTNESS, BR)
        cap.set(cv2.CAP_PROP_CONTRAST, CR)
        cap.set(cv2.CAP_PROP_SATURATION, 64)
        cap.set(cv2.CAP_PROP_HUE, 0)
        cap.set(cv2.CAP_PROP_GAMMA, 100)
        cap.set(cv2.CAP_PROP_GAIN, 0)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 6500)
        cap.set(cv2.CAP_PROP_SHARPNESS, 3)
        cap.set(cv2.CAP_PROP_BACKLIGHT, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, EXP)

    def find_phone(self, frame_img):
        """Find the phone inside the screen to cut the frames"""
        frame_img = cv2.bilateralFilter(frame_img, 11, 17, 17)
        edged = cv2.Canny(frame_img, 30, 200)
        cnts, heirarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        screenCnt = None
        # iterate over contours and find which satisfy some conditions
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # change value of 0.02
            x, y, w, h = cv2.boundingRect(approx)
            # x_adj, y_adj, width, height = x, y, w, h
            if h >= 15 and len(approx) == 4:
                screenCnt = approx
                break
        if screenCnt is not None:
            x, y, w, h = cv2.boundingRect(screenCnt)
            # cv2.rectangle(frame_img, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Draw rectangle
            # cv2.drawContours(frame_img, [screenCnt], -1, (255, 0, 0), 3)  # Draw contour
            return x, y, w, h
        else:
            return None

    def set_position(self):
        self.show_image_photos("white.png")

        # Initial camera set
        self.set_cam_val(-30, 32, 15, self.cap)

        # Find phone
        ratio = self.phone_height / self.phone_width
        print(f"RATIO: {ratio}")
        x_adj, y_adj, w, h = 0, 0, 0, 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                new = self.find_phone(frame)
                if new is None:
                    continue
                x, y, w, h = new
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Draw rectangle
                if self.cv_window is not None:
                    cv2.imshow(self.cv_window, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                try:
                    x_adj, y_adj, w, h = new
                    if (ratio - 0.1) < (w / h) < (ratio + 0.1) and (w * h > 1280 * 800 * 0.2):
                        print(f"FINAL SCREEN: {w} x {h}, x_adj: {x_adj}, y_adj: {y_adj} ratio: {w / h}")
                        break
                    else:
                        print(f"SCREEN: {w} x {h}, ratio: {w / h} | {bool((ratio - 0.1) < (w / h) < (ratio + 0.1))}, {bool(w * h > 1280 * 800 * 0.2)}")
                except (ZeroDivisionError, TypeError) as e:
                    pass

        self.x_adj, self.y_adj, self.w, self.h = x_adj, y_adj, w, h
        return x_adj, y_adj, w, h

    def match_br(self):
        # Open the step wedge image on the phone in Google Photos
        # self.create_step_wedge((self.phone_width, self.phone_height))
        self.show_image_photos('step_wedge.png')
        time.sleep(2)
        # Initial settings
        self.set_cam_val(0, 32, 15, self.cap)
        target_val = [i * (256 // (self.wedge_steps - 1)) for i in range(10)]
        target_val = target_val[::-1]
        EXPOSURE, BRIGHTNESS, CONTRAST = 32, 15, 15
        change_val = []
        # Keep matching brightness
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                org_frame = frame.copy()
                frame = frame[self.y_adj:self.y_adj + self.h, self.x_adj:self.x_adj + self.w]
                frame = frame[:, :, 0]
                step_size = self.w // 10
                means = []
                for i in range(self.wedge_steps):
                    x1 = i * step_size
                    x2 = (i + 1) * step_size
                    means.append(np.mean(frame[:, x1:x2]))

                if self.cv_window is not None:
                    cv2.imshow(self.cv_window, frame)
                diff = np.array(target_val) - np.array(means)
                mean_diff = np.mean(diff)
                rmse = np.sqrt(np.mean(diff ** 2))

                change_val.append((BRIGHTNESS, CONTRAST, EXPOSURE))

                prev = (BRIGHTNESS, CONTRAST)
                BRIGHTNESS = int(rmse / 256 * (self.MAX_BR - self.MIN_BR))
                # CONTRAST = int((np.std(diff) / 256) * (self.MAX_CR - self.MIN_CR))
                if (BRIGHTNESS, CONTRAST) == (prev[0], prev[1]):
                    if (mean_diff) > 10:
                        EXPOSURE += 1
                    elif (mean_diff) < -10:
                        EXPOSURE -= 1
                CONTRAST = 32
                count = change_val.count((BRIGHTNESS, CONTRAST, EXPOSURE))  # Count the occurrences of the values
                # count = change_val.count((BRIGHTNESS, CONTRAST))  # Count the occurrences of the values
                if count > 6:
                    print(f"FINAL:\n"
                          f"Brightness: {BRIGHTNESS}, Contrast: {CONTRAST}, Exposure: {EXPOSURE}\n"
                          f"width: {self.w}, height: {self.h}, x: {self.x_adj}, y: {self.y_adj}")
                    break
                print(f"BRIGHTNESS: {BRIGHTNESS}, CONTRAST: {CONTRAST}, EXP: {EXPOSURE}, count: {count}", end='\r')
                self.set_cam_val(BRIGHTNESS, CONTRAST, EXPOSURE, self.cap)
                # cv2.imwrite("final_frame.png", frame)
                # cv2.imwrite("final_org.png", org_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("WRONG EXIT: EXITED FROM WINDOW")
                    return None
        return BRIGHTNESS, CONTRAST, EXPOSURE


    def json_writer(self, BRIGHTNESS, CONTRAST, EXPOSURE, file):
        data = []
        with open(f'{file}', 'w') as f:
            data.append({"BRIGHTNESS": BRIGHTNESS, "CONTRAST": CONTRAST, "EXPOSURE": EXPOSURE})
            data.append({"width": self.w, "height": self.h, "x_adj": self.x_adj, "y_adj": self.y_adj})
            json.dump(data, f, indent=4)


def get_adb_devices():
    try:
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        output = result.stdout.strip().split('\n')
        devices = []
        for line in output[1:]:
            if 'device' in line:
                device_info = line.split('\t')
                devices.append(device_info[0])
        if len(devices) == 0:
            print("No adb devices found")
            exit(1)
        elif len(devices) > 1:
            print(f"Multiple adb devices found, using the first device {devices[0]}\n"
                  f"Provide the device with -D to use an other device")
        return devices[0]
    except FileNotFoundError:
        print("No adb devices found")
        exit(1)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('-D', '--device', type=str, default=None,
                     help='Device serial number')
    arg.add_argument('-c', '--camera', type=str, default=0,
                     help='Camera to use')
    arg.add_argument('-o', '--output', type=str, required=True,
                     help='Output file name')
    args = vars(arg.parse_args())

    # cap = cv2.VideoCapture(int(args['camera']))
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    # cap.set(cv2.CAP_PROP_FPS, 120)
    #
    # cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('cam', 1280, 800)

    device = f"{get_adb_devices()}" if args['device'] is None else f"{args['device']}"
    os.system(f"./turnon.sh {device}")
    os.system(f"./reset_cam.sh {args['camera']} {device}")

    config = Config(device=device, cap_ind=int(args['camera']), cv_window=True)

    calibration = Calibrator()
    calibration.set_position()
    br, cr, exp = calibration.match_br()
    calibration.json_writer(br, cr, exp, args['output'])
