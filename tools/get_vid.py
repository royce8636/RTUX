import argparse
import json
import os
import subprocess
import sys
import cv2
import threading
import fcntl
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(f"{project_root}")

from rtux_cnn.config import Config
from rtux_cnn.calibrator import Calibrator


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


def cmd_recv():
    global signal
    fl = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
    fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, fl | os.O_NONBLOCK)
    try:
        while True:
            if signal is None:
                return
            try:
                stdin = sys.stdin.read()
                if "\n" in stdin or "\r" in stdin:
                    signal = True
                    print("Executing next script")
            except (IOError, TypeError, IndexError) as e:
                pass
                time.sleep(0.1)
    except ValueError:
        print("ValueError: idk why this happens")
        exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--device', type=str, default=None,
                        help='serial number of the device to pass arguments to')
    parser.add_argument('-c', '--camera', type=int, default=0,
                        help='Specify which camera to take video from (num from /dev/video)')
    parser.add_argument('-C', '--calibrate', type=str, default=None,
                        help='Calibration file from br_calib.py')
    parser.add_argument('-p', '--path', type=str, default="usage_pattern",
                        help='Where usage pattern scripts are at (default is usage_pattern)')
    parser.add_argument('-d', '--directory', type=str, default="dataset",
                        help='Directory with all saved things')
    parser.add_argument('-g', '--game', type=str, default=None, required=True,
                        help='which game to test')
    parser.add_argument('-a', '--adb', type=str, default=None, required=True,
                        help='The opening activity of ADB is required to open the app!')
    args = vars(parser.parse_args())

    device = f"{get_adb_devices()}" if args['device'] is None else f"{args['device']}"
    print("Using device: " + device)

    config = Config(cap_ind=args['camera'], device=device, calib=args['calibrate'], cv_window=False)

    os.system(f"./turnon.sh {device}")
    # max_br = Config().get_max_br()
    # os.system(f'adb -s {device} shell settings put system screen_brightness {int(max_br)}')

    cam = config.get_camera()

    calib = Calibrator()
    if args['calibrate'] is None:
        calib.set_position()
        calib.match_br()
        width, height, x_adj, y_adj = calib.phone_width, calib.phone_height, calib.x_adj, calib.y_adj
    else:
        with open(args['calibrate'], 'r') as f:
            parsed_data = json.load(f)
            br = parsed_data[0]["BRIGHTNESS"]
            cr = parsed_data[0]["CONTRAST"]
            exp = parsed_data[0]["EXPOSURE"]
            calib.set_cam_val(br, cr, exp, cam)
            width, height = parsed_data[1]["width"], parsed_data[1]["height"]
            x_adj, y_adj = parsed_data[1]["x_adj"], parsed_data[1]["y_adj"]

    us_path = f"{args['path']}/{args['game']}"

    os.makedirs(us_path, exist_ok=True)

    if not os.path.exists(us_path):
        print(f"Usage pattern for {args['game']} not found:\n"
              f"    {us_path} not found")
        exit(0)

    try:
        shell_scripts = [os.path.join(us_path, file) for file in os.listdir(us_path) if file.endswith('.sh')]
        shell_scripts = sorted(shell_scripts, key=lambda x: int(x.split('.')[-2][-1]))
    except ValueError:
        print(f"Shell files in {us_path} are not named correctly:\n"
              f"Should be named as '{args['game']}_1.sh', '{args['game']}_2.sh', '{args['game']}_3.sh', etc.")
        exit(0)

    shell_scripts.insert(0, f"adb -s {device} shell am start -W {args['adb']}")
    print(shell_scripts)
    os.system(f"adb -s {device} shell am force-stop {args['adb'].split('/')[0]}")

    signal = False
    cmd_thread = threading.Thread(target=cmd_recv)
    cmd_thread.start()

    out = cv2.VideoWriter(f"{args['directory']}/{args['game']}/{args['game']}_encoded.avi",
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 120, (width, height))

    while cam.isOpened():
        ret, frame = cam.read()
        if ret is True:
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            if signal is True:
                signal = False
                try:
                    script_to_run = shell_scripts.pop(0)
                    threading.Thread(target=lambda: os.system(script_to_run)).start()
                except IndexError:
                    print("No more scripts to run, ending video capture")
                    signal = None
                    cmd_thread.join()
                    break
            frame = frame[y_adj: height + y_adj, x_adj: width + x_adj]
            out.write(frame)
            cv2.imshow("cam", frame)
        else:
            break

    print("Saving video...")
    out.release()
    cam.release()
    cv2.destroyAllWindows()
    print("Exiting Program")
    sys.exit(0)
